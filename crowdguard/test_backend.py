from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from lap import lapjv

from crowdguard.alerts import AlertManager
from crowdguard.anomaly import build_forecast, learn_baseline, score_risk
from crowdguard.detector_tracker import Detection
from crowdguard.memory import MemoryStore, TrackObservation
from crowdguard.metrics import TIMELINE_COLUMNS, frame_zone_metrics
from crowdguard.pipeline import PROJECT_ROOT, _clear_output, _reencode_for_browser, run_pipeline
from crowdguard.zones import ZoneGrid


ALERT_FIELDS = {
    "id", "type", "severity", "sec", "zone", "what", "why_flagged",
    "evidence_frames", "recommended_action", "metrics", "explanation_source",
}


def validate_outputs(output_dir: Path, expected_zone_count: int, source_video: Path | None = None) -> list[dict[str, Any]]:
    checks = []
    timeline_path = output_dir / "timeline.csv"
    alerts_path = output_dir / "alerts.json"
    forecast_path = output_dir / "forecast.json"
    video_path = next(
        (path for path in (output_dir / "annotated.mp4", output_dir / "annotated.webm") if path.is_file()),
        output_dir / "annotated.mp4",
    )
    frames_dir = output_dir / "frames"

    timeline_valid = False
    try:
        timeline = pd.read_csv(timeline_path)
        numeric = timeline[TIMELINE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        seconds = sorted(timeline["sec"].astype(int).unique().tolist())
        timeline_valid = (
            list(timeline.columns) == TIMELINE_COLUMNS
            and not timeline.empty
            and np.isfinite(numeric.to_numpy(dtype=float)).all()
            and timeline["zone"].between(0, expected_zone_count - 1).all()
            and timeline["risk"].between(0, 100).all()
            and not timeline.duplicated(["sec", "zone"]).any()
            and all(len(timeline[timeline.sec == sec]) == expected_zone_count for sec in seconds)
        )
    except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError):
        timeline = pd.DataFrame()
    checks.append({"file": str(timeline_path), "exists": timeline_path.is_file(), "valid": bool(timeline_valid), "detail": "columns, numeric values and one row per second/zone"})

    alerts_valid = False
    try:
        alerts = json.loads(alerts_path.read_text(encoding="utf-8"))
        alerts_valid = isinstance(alerts, list) and all(
            isinstance(item, dict)
            and ALERT_FIELDS.issubset(item)
            and item["severity"] in {"Low", "Medium", "High"}
            and isinstance(item["evidence_frames"], list)
            and all(
                isinstance(frame_path, str)
                and (Path(frame_path) if Path(frame_path).is_absolute() else PROJECT_ROOT / frame_path).is_file()
                for frame_path in item["evidence_frames"]
            )
            and isinstance(item["metrics"], dict)
            and item["explanation_source"] in {"vision_ai", "template"}
            for item in alerts
        )
    except (OSError, json.JSONDecodeError):
        alerts = []
    checks.append({"file": str(alerts_path), "exists": alerts_path.is_file(), "valid": bool(alerts_valid), "detail": "JSON list and alert fields"})

    forecast_valid = False
    try:
        forecast = json.loads(forecast_path.read_text(encoding="utf-8"))
        forecast_valid = (
            isinstance(forecast, dict)
            and {"text", "seconds_to_high", "trend", "safest_zone"}.issubset(forecast)
            and forecast["trend"] in {"rising", "falling", "stable"}
            and (forecast["seconds_to_high"] is None or isinstance(forecast["seconds_to_high"], (int, float)))
        )
    except (OSError, json.JSONDecodeError):
        forecast = {}
    checks.append({"file": str(forecast_path), "exists": forecast_path.is_file(), "valid": bool(forecast_valid), "detail": "forecast JSON keys and types"})

    frame_paths = sorted(frames_dir.glob("sec_*.jpg")) if frames_dir.is_dir() else []
    frames_valid = bool(frame_paths) and all(path.stat().st_size > 0 for path in frame_paths)
    if frames_valid:
        try:
            import cv2
            frames_valid = all(cv2.imread(str(path)) is not None for path in frame_paths)
        except ImportError:
            frames_valid = False
    checks.append({"file": str(frames_dir / "sec_N.jpg"), "exists": bool(frame_paths), "valid": bool(frames_valid), "detail": f"{len(frame_paths)} readable keyframes"})

    video_valid = False
    video_detail = "video opens, yields a frame, and uses a browser-playable codec"
    if video_path.is_file() and video_path.stat().st_size > 0:
        try:
            import cv2
            capture = cv2.VideoCapture(str(video_path))
            video_valid = capture.isOpened()
            if video_valid:
                output_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
                codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)).lower()
                ok, _ = capture.read()
                if video_path.suffix.lower() == ".webm":
                    codec_valid = codec in {"vp80", "vp90", "vp8 ", "vp9 "}
                else:
                    codec_valid = codec in {"avc1", "h264", "mp4v"}
                count_valid = True
                if source_video is not None and source_video.is_file():
                    source = cv2.VideoCapture(str(source_video))
                    source_frames = int(source.get(cv2.CAP_PROP_FRAME_COUNT))
                    source.release()
                    count_valid = source_frames <= 0 or abs(output_frames - source_frames) <= 1
                video_valid = bool(ok) and codec_valid and count_valid
                video_detail = f"codec={codec or 'unknown'}, frames={output_frames}, codec/count verified"
            capture.release()
        except ImportError:
            video_valid = False
    checks.append({"file": str(video_path), "exists": video_path.is_file(), "valid": bool(video_valid), "detail": video_detail})
    return checks


class BackendUnitTests(unittest.TestCase):
    def test_scipy_lap_cost_limit_preserves_alternative_valid_matches(self):
        total, row_to_column, column_to_row = lapjv(
            np.asarray([[1.0, 2.0], [2.0, 100.0]]),
            extend_cost=True,
            cost_limit=3.0,
        )
        self.assertEqual(total, 4.0)
        self.assertEqual(row_to_column.tolist(), [1, 0])
        self.assertEqual(column_to_row.tolist(), [1, 0])

    def test_grid_and_restricted_polygon(self):
        zones = ZoneGrid({
            "grid": {"rows": 2, "columns": 2},
            "restricted_polygons": [{"zone": 0, "points": [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]}],
        })
        self.assertEqual(zones.zone_index(25, 25, 100, 100), 0)
        self.assertEqual(zones.zone_index(75, 75, 100, 100), 3)
        self.assertTrue(zones.is_restricted(25, 25, 100, 100, 0))
        self.assertFalse(zones.is_restricted(75, 75, 100, 100, 3))

    def test_speed_uses_elapsed_video_time_and_missing_ids_are_safe(self):
        memory = MemoryStore(speed_window_seconds=1.0)
        first = memory.observe(Detection(0, 7, 0.9, (0, 0, 10, 10)), 0.0, 0)
        second = memory.observe(Detection(0, 7, 0.9, (10, 0, 20, 10)), 0.5, 0)
        untracked = memory.observe(Detection(0, None, 0.8, (0, 0, 10, 10)), 0.5, 0)
        self.assertEqual(first.speed_px_s, 0.0)
        self.assertAlmostEqual(second.speed_px_s, 20.0)
        self.assertIsNone(untracked.track_id)
        self.assertEqual(untracked.speed_px_s, 0.0)

    def test_counterflow_and_chaos_follow_unit_directions(self):
        observations = [
            TrackObservation(index, 0, (0, 0, 1, 1), (0, 0), 0, 20, direction, 0, 0, None)
            for index, direction in enumerate([(1, 0), (1, 0), (1, 0), (-1, 0)])
        ]
        result = frame_zone_metrics(observations, 1, min_move_px_s=10)[0]
        self.assertEqual(result["count"], 4)
        self.assertAlmostEqual(result["counterflow"], 0.25)
        self.assertAlmostEqual(result["chaos"], 0.5)

    def test_baseline_risk_and_std_floors(self):
        rows = [
            {"sec": 0, "zone": 0, "count": 10, "speed": 20, "counterflow": 0, "chaos": 0, "risk": 0},
            {"sec": 1, "zone": 0, "count": 10, "speed": 20, "counterflow": 0, "chaos": 0, "risk": 0},
            {"sec": 2, "zone": 0, "count": 40, "speed": 0, "counterflow": 1, "chaos": 1, "risk": 0},
        ]
        baseline = learn_baseline(rows, 1, 2)
        self.assertEqual(baseline[0]["count_std"], 1.0)
        self.assertEqual(baseline[0]["speed_std"], 5.0)
        scored = score_risk(rows, baseline, 2)
        self.assertEqual(scored[0]["risk"], 0.0)
        self.assertEqual(scored[1]["risk"], 0.0)
        self.assertGreater(scored[2]["risk"], 0.0)
        self.assertLessEqual(scored[2]["risk"], 100.0)

    def test_alert_merge_within_ten_seconds(self):
        alerts = AlertManager(10)
        first = alerts.add("Crush Risk", "High", 12, 4, "observed", "measured", "output/frames/sec_12.jpg", {"risk": 80.0})
        alerts.add("Crush Risk", "High", 20, 4, "observed again", "measured again", "output/frames/sec_20.jpg", {"risk": 82.0})
        self.assertEqual(len(alerts.output()), 1)
        self.assertEqual(first["sec"], 12)
        self.assertEqual(len(first["evidence_frames"]), 2)
        self.assertEqual(first["metrics"]["risk"], 82.0)

    def test_forecast_uses_recent_risk_and_capacity_density_for_exit(self):
        zones = ZoneGrid({
            "grid": {"rows": 1, "columns": 2},
            "capacities": [100, 200],
        })
        rows = [
            {"sec": 0, "zone": 0, "count": 30, "risk": 10},
            {"sec": 0, "zone": 1, "count": 40, "risk": 10},
            {"sec": 1, "zone": 0, "count": 40, "risk": 20},
            {"sec": 1, "zone": 1, "count": 60, "risk": 20},
            {"sec": 2, "zone": 0, "count": 50, "risk": 40},
            {"sec": 2, "zone": 1, "count": 60, "risk": 40},
        ]
        forecast = build_forecast(rows, zones, high_threshold=70)
        self.assertEqual(forecast["trend"], "rising")
        self.assertGreater(forecast["seconds_to_high"], 0)
        self.assertEqual(forecast["safest_zone"]["zone"], 1)

    def test_pipeline_clears_only_the_configured_output_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "output"
            frames = output / "frames"
            frames.mkdir(parents=True)
            (output / "old.csv").write_text("old", encoding="utf-8")
            (frames / "old.jpg").write_bytes(b"old")
            _clear_output(output, root)
            self.assertEqual(list(output.iterdir()), [])
            with self.assertRaises(ValueError):
                _clear_output(root / "outside", root / "different-root")

    def test_ffmpeg_failure_keeps_original_and_returns_for_keyframe_fallback(self):
        import cv2

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            original = output / "annotated-original.mp4"
            original.write_bytes(b"not a valid video stream")
            warning = io.StringIO()
            with contextlib.redirect_stderr(warning):
                final_path, codec = _reencode_for_browser(cv2, original, output, timeout_seconds=15)
            self.assertIsNone(final_path)
            self.assertIsNone(codec)
            self.assertTrue(original.is_file())
            self.assertIn("app will use keyframes", warning.getvalue())
            self.assertFalse((output / "annotated.mp4").exists())
            self.assertFalse((output / "annotated.webm").exists())
            self.assertEqual(list(output.glob(".annotated-*-temporary.*")), [])


def run_video_test(video_path: str | Path, config_path: str | Path | None = None) -> int:
    config = None
    if config_path:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    summary = run_pipeline(video_path, config=config)
    print(f"Unique tracks: {summary['unique_tracks']}")
    print("Top 5 riskiest seconds:")
    for item in summary["top_risky_seconds"]:
        print(f"  sec {item['sec']}: risk {item['risk']:.2f}")
    print("Alert counts by type:")
    for alert_type, count in sorted(summary["alert_counts"].items()):
        print(f"  {alert_type}: {count}")
    checks = validate_outputs(Path(summary["output_dir"]), int(summary["zones"]), Path(summary["video"]))
    print("Output validation:")
    for check in checks:
        print(f"  {'PASS' if check['exists'] and check['valid'] else 'FAIL'} {check['file']}: {check['detail']}")
    return 0 if all(check["exists"] and check["valid"] for check in checks) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run CrowdGuard and validate its generated outputs.")
    parser.add_argument("video", help="Path to a real input video")
    parser.add_argument("--config", help="Optional JSON config override")
    args = parser.parse_args(argv)
    try:
        return run_video_test(args.video, args.config)
    except Exception as error:
        parser.exit(1, f"Backend test could not run: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
