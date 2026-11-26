from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .memory import TrackObservation


TIMELINE_COLUMNS = ["sec", "zone", "count", "speed", "counterflow", "chaos", "risk"]


def frame_zone_metrics(
    observations: Iterable[TrackObservation],
    zone_count: int,
    min_move_px_s: float = 12.0,
) -> list[dict[str, float | int]]:
    """Calculate occupancy and movement direction statistics for one frame."""
    grouped: dict[int, list[TrackObservation]] = defaultdict(list)
    for observation in observations:
        if observation.class_id == 0 and 0 <= observation.zone < zone_count:
            grouped[observation.zone].append(observation)

    metrics = []
    for zone in range(zone_count):
        people = grouped[zone]
        moving = [person for person in people if person.speed_px_s > min_move_px_s and person.direction is not None]
        mean_direction = np.zeros(2, dtype=float)
        if moving:
            mean_direction = np.mean(np.asarray([person.direction for person in moving], dtype=float), axis=0)
        mean_length = float(np.linalg.norm(mean_direction))
        counterflow = 0.0
        chaos = 0.0
        if moving:
            if mean_length > 1e-9:
                mean_unit = mean_direction / mean_length
                opposing = sum(float(np.dot(person.direction, mean_unit)) < -0.5 for person in moving)
                counterflow = opposing / len(moving)
            chaos = float(np.clip(1.0 - mean_length, 0.0, 1.0))
        metrics.append({
            "zone": zone,
            "count": len(people),
            "speed": float(np.mean([person.speed_px_s for person in people])) if people else 0.0,
            "counterflow": float(np.clip(counterflow, 0.0, 1.0)),
            "chaos": chaos,
        })
    return metrics


class PerSecondAggregator:
    """Average frame snapshots into exactly one record per second and zone."""

    def __init__(self, zone_count: int):
        self.zone_count = int(zone_count)
        self._sums: dict[int, dict[int, dict[str, float]]] = {}
        self._samples: dict[int, int] = defaultdict(int)
        self._outflow: dict[tuple[int, int], int] = defaultdict(int)

    def add(self, sec: int, metrics: list[dict[str, float | int]], outflow_by_zone: dict[int, int] | None = None) -> None:
        sec = int(sec)
        self._samples[sec] += 1
        bucket = self._sums.setdefault(sec, {zone: {"count": 0.0, "speed": 0.0, "counterflow": 0.0, "chaos": 0.0} for zone in range(self.zone_count)})
        for item in metrics:
            zone = int(item["zone"])
            for field in ("count", "speed", "counterflow", "chaos"):
                bucket[zone][field] += float(item[field])
        for zone, count in (outflow_by_zone or {}).items():
            self._outflow[(sec, int(zone))] += int(count)

    def finalize(self, total_seconds: int) -> list[dict[str, float | int]]:
        rows = []
        for sec in range(max(0, int(total_seconds))):
            samples = self._samples.get(sec, 0)
            bucket = self._sums.get(sec)
            for zone in range(self.zone_count):
                values = {"count": 0.0, "speed": 0.0, "counterflow": 0.0, "chaos": 0.0}
                if samples and bucket:
                    values = {field: bucket[zone][field] / samples for field in values}
                rows.append({
                    "sec": sec,
                    "zone": zone,
                    "count": int(round(values["count"])),
                    "speed": float(values["speed"]),
                    "counterflow": float(values["counterflow"]),
                    "chaos": float(values["chaos"]),
                    "risk": 0.0,
                    "outflow_per_sec": int(self._outflow.get((sec, zone), 0)),
                })
        return rows
