from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import Any

import numpy as np


DEFAULT_WEIGHTS = {"density": 0.35, "speed_drop": 0.25, "counterflow": 0.25, "chaos": 0.15}


def learn_baseline(rows: list[dict[str, Any]], zone_count: int, baseline_seconds: int) -> dict[int, dict[str, float]]:
    baseline_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["sec"]) < int(baseline_seconds):
            baseline_rows[int(row["zone"])].append(row)
    stats = {}
    for zone in range(zone_count):
        zone_rows = baseline_rows.get(zone, [])
        values = {}
        for field, floor in (("count", 1.0), ("speed", 5.0), ("counterflow", 0.01), ("chaos", 0.01)):
            samples = np.asarray([float(row[field]) for row in zone_rows], dtype=float)
            values[f"{field}_mean"] = float(np.mean(samples)) if len(samples) else 0.0
            values[f"{field}_std"] = max(float(np.std(samples)) if len(samples) else 0.0, floor)
        stats[zone] = values
    return stats


def score_risk(
    rows: list[dict[str, Any]],
    baseline: dict[int, dict[str, float]],
    baseline_seconds: int,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    configured = dict(DEFAULT_WEIGHTS)
    if weights:
        configured.update({key: float(value) for key, value in weights.items() if key in configured})
    for key in configured:
        configured[key] = max(0.0, configured[key])
    for row in rows:
        sec, zone = int(row["sec"]), int(row["zone"])
        if sec < int(baseline_seconds):
            row["risk"] = 0.0
            continue
        stats = baseline.get(zone, {})
        density_z = np.clip((float(row["count"]) - stats.get("count_mean", 0.0)) / (3.0 * max(1.0, stats.get("count_std", 1.0))), 0.0, 1.0)
        speed_drop_z = np.clip((stats.get("speed_mean", 0.0) - float(row["speed"])) / (3.0 * max(5.0, stats.get("speed_std", 5.0))), 0.0, 1.0)
        counterflow_scaled = np.clip(float(row["counterflow"]), 0.0, 1.0)
        chaos = np.clip(float(row["chaos"]), 0.0, 1.0)
        raw_score = 100.0 * (
            configured["density"] * density_z
            + configured["speed_drop"] * speed_drop_z
            + configured["counterflow"] * counterflow_scaled
            + configured["chaos"] * chaos
        )
        row["risk"] = float(np.clip(raw_score, 0.0, 100.0))
    return rows


def build_forecast(
    rows: list[dict[str, Any]],
    zones: Any,
    high_threshold: float = 70.0,
    history_seconds: int = 30,
) -> dict[str, Any]:
    per_second: dict[int, float] = {}
    latest_counts: dict[int, float] = {}
    latest_sec = max((int(row["sec"]) for row in rows), default=-1)
    for row in rows:
        sec = int(row["sec"])
        per_second[sec] = max(per_second.get(sec, 0.0), float(row["risk"]))
        if sec == latest_sec:
            latest_counts[int(row["zone"])] = float(row["count"])
    seconds = sorted(per_second)
    recent = seconds[-max(2, int(history_seconds)):]
    slope = 0.0
    if len(recent) >= 2:
        slope = float(np.polyfit(np.asarray(recent, dtype=float), np.asarray([per_second[sec] for sec in recent]), 1)[0])
    epsilon = 0.05
    trend = "rising" if slope > epsilon else "falling" if slope < -epsilon else "stable"
    current_risk = per_second[seconds[-1]] if seconds else 0.0
    seconds_to_high = None
    if current_risk >= high_threshold:
        seconds_to_high = 0
        text = "Risk is currently HIGH."
    elif trend == "rising" and slope > 0:
        seconds_to_high = int(ceil((high_threshold - current_risk) / slope))
        text = f"Risk may reach HIGH in ~{seconds_to_high} seconds."
    elif trend == "falling":
        text = "Risk trend is falling."
    else:
        text = "Risk trend is stable."

    safest_zone = None
    if latest_counts:
        safest_zone = min(latest_counts, key=lambda index: latest_counts[index] / zones.zone(index).capacity)
    safest = None if safest_zone is None else {"zone": int(safest_zone), "name": zones.zone(safest_zone).name}
    return {
        "text": text,
        "seconds_to_high": seconds_to_high,
        "trend": trend,
        "safest_zone": safest,
    }
