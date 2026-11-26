"""CrowdGuard video pipeline data contract.

Writes output/timeline.csv (sec, zone, count, speed, counterflow, chaos, risk),
output/alerts.json (id, type, severity, sec, zone, what, why_flagged,
evidence_frames, recommended_action, metrics, explanation_source),
output/forecast.json (text, seconds_to_high, trend, safest_zone),
output/frames/sec_N.jpg, and a browser-ready output/annotated.mp4 or
output/annotated.webm. If re-encoding fails, output/annotated-original.mp4 is
kept and the frontend falls back to the keyframes. All measurements are
computed from supplied video frames; this module has no simulated data path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("CROWDGUARD_ROOT", Path(__file__).resolve().parent.parent)).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crowdguard.alerts import evaluate_alerts
from crowdguard.anomaly import build_forecast, learn_baseline, score_risk
from crowdguard.detector_tracker import DetectorTracker
from crowdguard.explain import explain_alert
from crowdguard.memory import MemoryStore
from crowdguard.metrics import TIMELINE_COLUMNS, PerSecondAggregator, frame_zone_metrics
from crowdguard.zones import ZoneGrid

ProgressCallback = Callable[[float, str], None]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config: dict[str, Any] | None = None, config_path: str | Path | None = None) -> dict[str, Any]:
    package_config = Path(__file__).with_name("config.json")
    merged: dict[str, Any] = {}
    for path in (package_config, Path(config_path) if config_path else PROJECT_ROOT / "config.json"):
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Could not read config file {path}: {error}") from error
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file must contain a JSON object: {path}")
            merged = _deep_merge(merged, loaded)
    if config:
        if not isinstance(config, dict):
            raise TypeError("config must be a dictionary or None")
        merged = _deep_merge(merged, config)
    return merged


def _notify(callback: ProgressCallback | None, fraction: float, message: str) -> None:
    if callback is not None:
        callback(float(min(1.0, max(0.0, fraction))), str(message))


def _clear_output(output_dir: Path, project_root: Path) -> None:
    resolved_root = project_root.resolve()
    if output_dir.resolve().parent != resolved_root or output_dir.name != "output":
        raise ValueError("Refusing to clear an output directory outside the project root.")
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _open_video_writer(cv2: Any, path: Path, fps: float, size: tuple[int, int]):
    errors = []
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
        errors.append(codec)
    raise RuntimeError(f"Could not create MP4 writer; tried codecs: {', '.join(errors)}")


def _draw_grid(cv2: Any, frame: Any, zones: ZoneGrid) -> None:
    height, width = frame.shape[:2]
    color = (90, 190, 185)
    for column in range(1, zones.columns):
        x = int(width * column / zones.columns)
        cv2.line(frame, (x, 0), (x, height), color, 1)
    for row in range(1, zones.rows):
        y = int(height * row / zones.rows)
        cv2.line(frame, (0, y), (width, y), color, 1)
    for zone in zones.zones:
        x = int(width * (zone.column + 0.04) / zones.columns)
        y = int(height * (zone.row + 0.10) / zones.rows)
        cv2.putText(frame, zone.name[:24], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _draw_detections(cv2: Any, frame: Any, detections: list[Any]) -> None:
    class_names = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        color = (80, 210, 135) if detection.class_id == 0 else (40, 190, 245)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        track = "?" if detection.track_id is None else str(detection.track_id)
        label = f"{class_names.get(detection.class_id, 'object')} #{track}"
        cv2.putText(frame, label, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_risk_bar(cv2: Any, frame: Any, risk: float, low_threshold: float, high_threshold: float) -> None:
    height, width = frame.shape[:2]
    risk = float(min(100.0, max(0.0, risk)))
    color = (70, 190, 105) if risk < low_threshold else (55, 195, 235) if risk <= high_threshold else (65, 75, 235)
    left, top = 14, 14
    bar_width = min(300, max(100, width - 28))
    cv2.rectangle(frame, (left, top), (left + bar_width, top + 20), (35, 38, 40), -1)
    cv2.rectangle(frame, (left, top), (left + int(bar_width * risk / 100.0), top + 20), color, -1)
    cv2.rectangle(frame, (left, top), (left + bar_width, top + 20), (235, 240, 240), 1)
    cv2.putText(frame, f"RISK {risk:.0f}/100", (left + 6, top + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _write_keyframe(cv2: Any, frame: Any, path: Path) -> None:
    height, width = frame.shape[:2]
    resized_height = max(1, int(round(height * 640 / max(1, width))))
    resized = cv2.resize(frame, (640, resized_height), interpolation=cv2.INTER_AREA if width > 640 else cv2.INTER_LINEAR)
    if not cv2.imwrite(str(path), resized):
        raise RuntimeError(f"Could not write keyframe: {path}")


def _render_risk_video(cv2: Any, base_path: Path, final_path: Path, fps: float, size: tuple[int, int], risks: dict[int, float], config: dict[str, Any]) -> str:
    source = cv2.VideoCapture(str(base_path))
    if not source.isOpened():
        raise RuntimeError("Could not reopen the temporary annotated video to add risk overlays.")
    writer, codec = _open_video_writer(cv2, final_path, fps, size)
    thresholds = config.get("thresholds", {})
    low = float(thresholds.get("low_risk", 40))
    high = float(thresholds.get("high_risk", 70))
    frame_index = 0
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                break
            sec = int(frame_index / fps)
            _draw_risk_bar(cv2, frame, risks.get(sec, 0.0), low, high)
            writer.write(frame)
            frame_index += 1
    finally:
        source.release()
        writer.release()
    if frame_index == 0:
        raise RuntimeError("The annotated video contained no readable frames.")
    return codec


def _reencode_for_browser(
    cv2: Any,
    original_path: Path,
    output_dir: Path,
    timeout_seconds: int = 180,
) -> tuple[Path | None, str | None]:
    """Publish a browser-friendly video, keeping the OpenCV original on failure."""
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        if not ffmpeg or not Path(ffmpeg).is_file():
            raise FileNotFoundError("imageio-ffmpeg did not provide an executable")
    except Exception as error:
        print(
            f"WARNING: browser video re-encode unavailable ({error}); keeping {original_path.name}; app will use keyframes.",
            file=sys.stderr,
        )
        return None, None

    attempts = [
        ("libx264", "annotated.mp4", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]),
        ("libvpx-vp9", "annotated.webm", ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "4", "-b:v", "1M", "-pix_fmt", "yuv420p"]),
        ("mpeg4", "annotated.mp4", ["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]),
    ]
    failures: list[str] = []
    for codec, final_name, codec_args in attempts:
        final_path = output_dir / final_name
        temporary_path = output_dir / f".annotated-{codec}-temporary{Path(final_name).suffix}"
        temporary_path.unlink(missing_ok=True)
        command = [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(original_path), "-map", "0:v:0", "-an",
            *codec_args, str(temporary_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=max(1, int(timeout_seconds)),
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or f"ffmpeg exited {result.returncode}").strip().splitlines()
                failures.append(f"{codec}: {detail[-1] if detail else 'encoding failed'}")
                continue
            if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                failures.append(f"{codec}: ffmpeg produced no output")
                continue
            capture = cv2.VideoCapture(str(temporary_path))
            opened = capture.isOpened()
            has_frame = False
            if opened:
                ok, _ = capture.read()
                has_frame = bool(ok)
            capture.release()
            if not opened or not has_frame:
                failures.append(f"{codec}: output failed OpenCV decode verification")
                continue

            final_path.unlink(missing_ok=True)
            os.replace(temporary_path, final_path)
            other_final = output_dir / ("annotated.webm" if final_name.endswith(".mp4") else "annotated.mp4")
            try:
                other_final.unlink(missing_ok=True)
            except OSError:
                pass
            try:
                original_path.unlink(missing_ok=True)
            except OSError as error:
                print(f"WARNING: encoded video is ready, but could not delete {original_path.name}: {error}", file=sys.stderr)
            return final_path, codec
        except Exception as error:
            failures.append(f"{codec}: {error}")
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as error:
                failures.append(f"{codec}: could not remove temporary file: {error}")

    details = "; ".join(failures) or "no encoder succeeded"
    print(
        f"WARNING: browser video re-encode failed ({details}); keeping {original_path.name}; app will use keyframes.",
        file=sys.stderr,
    )
    return None, None


def run_pipeline(
    video_path: str | Path,
    config: dict[str, Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Analyze a real video and write all frontend-consumed output artifacts."""
    source_path = Path(video_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source_path}")
    settings = load_config(config)
    output_dir = PROJECT_ROOT / "output"
    _clear_output(output_dir, PROJECT_ROOT)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required. Install opencv-python from requirements.txt.") from error

    _notify(progress_callback, 0.01, "Opening source video")
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video: {source_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    frame_count_hint = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video reports invalid frame dimensions.")
    expected_frames = max(0, frame_count_hint)
    requested_skip = max(1, int(settings.get("frame_skip", 1)))
    frame_skip = min(requested_skip, max(1, int(fps)))

    _notify(progress_callback, 0.03, "Loading YOLO tracker")
    detector = DetectorTracker(settings)
    zones = ZoneGrid(settings)
    memory = MemoryStore(
        memory_seconds=float(settings.get("memory_seconds", 60)),
        speed_window_seconds=float(settings.get("speed_window_seconds", 1.0)),
        stationary_speed_px_s=float(settings.get("thresholds", {}).get("stationary_speed_px_s", 8)),
    )
    aggregator = PerSecondAggregator(zones.width)
    observations_by_second: dict[int, list[Any]] = {}
    last_detections: list[Any] = []
    keyframes_written: set[int] = set()
    base_video = output_dir / "annotated-base.mp4"
    writer, _ = _open_video_writer(cv2, base_video, fps, (width, height))
    frame_index = 0
    inference_count = 0
    _notify(progress_callback, 0.05, "Detecting and tracking video frames")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            sec = int(frame_index / fps)
            if frame_index % frame_skip == 0:
                detections = detector.track(frame)
                observations = []
                outflow: dict[int, int] = defaultdict(int)
                for detection in detections:
                    center_x = (detection.bbox[0] + detection.bbox[2]) / 2.0
                    center_y = (detection.bbox[1] + detection.bbox[3]) / 2.0
                    zone_index = zones.zone_index(center_x, center_y, width, height)
                    observation = memory.observe(detection, frame_index / fps, zone_index)
                    observations.append(observation)
                    if observation.class_id == 0 and observation.previous_zone is not None and observation.previous_zone != observation.zone:
                        outflow[observation.previous_zone] += 1
                memory.prune(frame_index / fps)
                per_zone = frame_zone_metrics(observations, zones.width, float(settings.get("thresholds", {}).get("min_move_px_s", 12)))
                aggregator.add(sec, per_zone, outflow)
                observations_by_second[sec] = observations
                last_detections = detections
                inference_count += 1

            if sec not in keyframes_written:
                _write_keyframe(cv2, frame, frame_dir / f"sec_{sec}.jpg")
                keyframes_written.add(sec)
            annotated = frame.copy()
            _draw_grid(cv2, annotated, zones)
            _draw_detections(cv2, annotated, last_detections)
            writer.write(annotated)
            frame_index += 1
            if frame_index % 15 == 0:
                fraction = 0.05 + 0.72 * (frame_index / expected_frames) if expected_frames else 0.05 + 0.72 * (1.0 - math.exp(-frame_index / max(1.0, fps * 20)))
                _notify(progress_callback, min(0.78, fraction), f"Processed {frame_index} source frames")
    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        raise RuntimeError("The source video contains no readable frames.")
    if inference_count == 0:
        raise RuntimeError("No source frames were sent to the detector.")
    total_seconds = int(math.ceil(frame_index / fps))
    rows = aggregator.finalize(total_seconds)

    requested_baseline = max(1, int(settings.get("baseline_seconds", 30)))
    if total_seconds < requested_baseline:
        effective_baseline = max(1, total_seconds // 3)
        print(
            f"Video length ({total_seconds} s) is shorter than the {requested_baseline} s baseline; "
            f"using the first {effective_baseline} s as baseline.",
            file=sys.stderr,
        )
    else:
        effective_baseline = requested_baseline
    settings["_effective_baseline_seconds"] = effective_baseline
    baseline = learn_baseline(rows, zones.width, effective_baseline)
    score_risk(rows, baseline, effective_baseline, settings.get("weights"))
    high_threshold = float(settings.get("thresholds", {}).get("high_risk", 70))
    forecast = build_forecast(rows, zones, high_threshold)

    management: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        zone = zones.zone(int(row["zone"]))
        outflow = float(row.get("outflow_per_sec", 0))
        wait = float(row["count"]) / outflow if zone.is_gate and outflow > 0 else None
        management[(int(row["sec"]), int(row["zone"]))] = {
            "capacity_percent": round(float(row["count"]) / zone.capacity * 100.0, 2),
            "outflow_per_sec": round(outflow, 2),
            "queue_wait_estimate_seconds": None if wait is None else round(wait, 2),
            "safest_zone": forecast["safest_zone"],
        }

    def evidence_frame_for_sec(sec: int) -> str:
        candidates = [int(sec), int(sec) - 1, int(sec) + 1, int(sec) - 2, int(sec) + 2]
        for candidate in candidates:
            path = frame_dir / f"sec_{candidate}.jpg"
            if path.is_file():
                return path.relative_to(PROJECT_ROOT).as_posix()
        return ""

    alerts = evaluate_alerts(
        rows=rows,
        observations_by_second=observations_by_second,
        zones=zones,
        baseline=baseline,
        management=management,
        frame_size=(width, height),
        config=settings,
        evidence_frame_for_sec=evidence_frame_for_sec,
    )
    for alert in alerts:
        sec = int(alert["sec"])
        evidence_paths = []
        for candidate in (sec, sec - 1, sec + 1, sec - 2, sec + 2):
            path = frame_dir / f"sec_{candidate}.jpg"
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            if path.is_file() and relative not in evidence_paths:
                evidence_paths.append(relative)
            if len(evidence_paths) == 3:
                break
        alert["evidence_frames"] = evidence_paths
        full_paths = [str(PROJECT_ROOT / path) for path in evidence_paths]
        explanation, source = explain_alert(alert, full_paths, settings)
        alert["why_flagged"] = explanation
        alert["explanation_source"] = source

    timeline_path = output_dir / "timeline.csv"
    pd.DataFrame([{key: row[key] for key in TIMELINE_COLUMNS} for row in rows], columns=TIMELINE_COLUMNS).to_csv(timeline_path, index=False)
    (output_dir / "alerts.json").write_text(json.dumps(alerts, indent=2, allow_nan=False), encoding="utf-8")
    (output_dir / "forecast.json").write_text(json.dumps(forecast, indent=2, allow_nan=False), encoding="utf-8")

    risks_by_second = {sec: 0.0 for sec in range(total_seconds)}
    for row in rows:
        sec = int(row["sec"])
        risks_by_second[sec] = max(risks_by_second[sec], float(row["risk"]))
    annotated_original = output_dir / "annotated-original.mp4"
    _notify(progress_callback, 0.82, "Rendering risk overlay")
    opencv_codec = _render_risk_video(cv2, base_video, annotated_original, fps, (width, height), risks_by_second, settings)
    base_video.unlink(missing_ok=True)
    _notify(progress_callback, 0.88, "Re-encoding browser video")
    browser_video, browser_codec = _reencode_for_browser(cv2, annotated_original, output_dir)
    _notify(progress_callback, 1.0, "CrowdGuard outputs are ready" if browser_video else "Keyframes are ready; video re-encode failed")

    top_risks = sorted(((sec, risk) for sec, risk in risks_by_second.items()), key=lambda pair: pair[1], reverse=True)[:5]
    return {
        "video": str(source_path),
        "frames_processed": frame_index,
        "inference_frames": inference_count,
        "unique_tracks": len(memory.unique_tracks),
        "seconds": total_seconds,
        "zones": zones.width,
        "baseline_seconds": effective_baseline,
        "top_risky_seconds": [{"sec": sec, "risk": round(risk, 2)} for sec, risk in top_risks],
        "alert_counts": {kind: sum(alert["type"] == kind for alert in alerts) for kind in sorted({alert["type"] for alert in alerts})},
        "annotated_codec": browser_codec or f"opencv-{opencv_codec}; not browser-verified",
        "annotated_video": str(browser_video) if browser_video else None,
        "annotated_original": str(annotated_original) if annotated_original.is_file() else None,
        "output_dir": str(output_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CrowdGuard on a real video file.")
    parser.add_argument("video", help="Path to a real input video")
    parser.add_argument("--config", help="Optional JSON config override")
    args = parser.parse_args(argv)
    config = None
    if args.config:
        try:
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"Could not read config file: {error}")
    try:
        summary = run_pipeline(args.video, config=config, progress_callback=lambda fraction, message: print(f"[{fraction:5.1%}] {message}"))
    except Exception as error:
        parser.exit(1, f"CrowdGuard pipeline failed: {error}\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
