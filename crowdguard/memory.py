from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Any


@dataclass(frozen=True)
class TrackObservation:
    track_id: int | None
    class_id: int
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    zone: int
    speed_px_s: float
    direction: tuple[float, float] | None
    stationary_seconds: float
    confinement_radius_px: float
    previous_zone: int | None


@dataclass
class _TrackHistory:
    class_id: int
    points: deque[tuple[float, float, float, int]]
    last_seen: float
    stationary_since: float | None = None
    stationary_anchor: tuple[float, float] | None = None
    previous_zone: int | None = None


class MemoryStore:
    """Short-lived motion history; IDs are detector track IDs, never identities."""

    def __init__(self, memory_seconds: float = 60.0, speed_window_seconds: float = 1.0, stationary_speed_px_s: float = 8.0):
        self.memory_seconds = max(1.0, float(memory_seconds))
        self.speed_window_seconds = max(0.2, float(speed_window_seconds))
        self.stationary_speed_px_s = max(0.0, float(stationary_speed_px_s))
        self._tracks: dict[tuple[int, int], _TrackHistory] = {}
        self.unique_tracks: set[tuple[int, int]] = set()
        self.zone_history: dict[int, deque[dict[str, float]]] = {}

    def observe(self, detection: Any, timestamp: float, zone: int) -> TrackObservation:
        x1, y1, x2, y2 = (float(value) for value in detection.bbox)
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        class_id = int(detection.class_id)
        track_id = detection.track_id
        if track_id is None:
            return TrackObservation(None, class_id, (x1, y1, x2, y2), center, int(zone), 0.0, None, 0.0, 0.0, None)

        track_id = int(track_id)
        key = (class_id, track_id)
        self.unique_tracks.add(key)
        history = self._tracks.get(key)
        if history is None:
            history = _TrackHistory(class_id, deque(), float(timestamp))
            self._tracks[key] = history
        previous_zone = history.points[-1][3] if history.points else None
        history.last_seen = float(timestamp)
        history.points.append((float(timestamp), center[0], center[1], int(zone)))
        cutoff = float(timestamp) - self.memory_seconds
        while history.points and history.points[0][0] < cutoff:
            history.points.popleft()

        window_start = float(timestamp) - self.speed_window_seconds
        window_points = [point for point in history.points if point[0] >= window_start]
        if len(window_points) < 2:
            window_points = list(history.points)[-2:]
        speed = 0.0
        direction = None
        if len(window_points) >= 2:
            first, last = window_points[0], window_points[-1]
            elapsed = last[0] - first[0]
            if elapsed > 0.0:
                dx = (last[1] - first[1]) / elapsed
                dy = (last[2] - first[2]) / elapsed
                speed = hypot(dx, dy)
                if speed > 0:
                    direction = (dx / speed, dy / speed)

        if speed <= self.stationary_speed_px_s:
            if history.stationary_since is None:
                history.stationary_since = float(timestamp)
                history.stationary_anchor = center
        else:
            history.stationary_since = None
            history.stationary_anchor = None
        stationary_seconds = 0.0 if history.stationary_since is None else max(0.0, float(timestamp) - history.stationary_since)
        anchor = history.stationary_anchor or center
        confinement = max((hypot(point[1] - anchor[0], point[2] - anchor[1]) for point in history.points if point[0] >= (history.stationary_since or timestamp)), default=0.0)

        return TrackObservation(
            track_id=track_id,
            class_id=class_id,
            bbox=(x1, y1, x2, y2),
            center=center,
            zone=int(zone),
            speed_px_s=float(speed),
            direction=direction,
            stationary_seconds=stationary_seconds,
            confinement_radius_px=float(confinement),
            previous_zone=previous_zone,
        )

    def record_zone_metrics(self, zone: int, timestamp: float, metrics: dict[str, float]) -> None:
        history = self.zone_history.setdefault(int(zone), deque(maxlen=60))
        history.append({"timestamp": float(timestamp), **{key: float(value) for key, value in metrics.items()}})

    def prune(self, timestamp: float) -> None:
        stale = [key for key, value in self._tracks.items() if float(timestamp) - value.last_seen > self.memory_seconds]
        for key in stale:
            del self._tracks[key]
