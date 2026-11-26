from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Zone:
    index: int
    row: int
    column: int
    name: str
    capacity: int
    is_gate: bool = False
    is_sensitive: bool = False


class ZoneGrid:
    """Assign pixel centers to configured grid cells and restricted polygons."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        grid = config.get("grid", {})
        self.rows = max(1, int(grid.get("rows", config.get("rows", 3))))
        self.columns = max(1, int(grid.get("columns", grid.get("cols", config.get("columns", 3)))))
        self.width = self.rows * self.columns
        names = config.get("zone_names", [])
        capacities = config.get("capacities", config.get("zone_capacities", []))
        zone_settings = config.get("zones", [])
        self.zones: list[Zone] = []
        for index in range(self.width):
            row, column = divmod(index, self.columns)
            item = zone_settings[index] if isinstance(zone_settings, list) and index < len(zone_settings) else {}
            if not isinstance(item, dict):
                item = {}
            name = item.get("name")
            if not name and isinstance(names, list) and index < len(names):
                name = names[index]
            if not name and isinstance(names, dict):
                name = names.get(str(index), names.get(str(index + 1)))
            capacity = item.get("capacity")
            if capacity is None and isinstance(capacities, list) and index < len(capacities):
                capacity = capacities[index]
            if capacity is None and isinstance(capacities, dict):
                capacity = capacities.get(str(index), capacities.get(str(index + 1)))
            self.zones.append(Zone(
                index=index,
                row=row,
                column=column,
                name=str(name or f"Zone {index + 1}"),
                capacity=max(1, int(capacity or 150)),
                is_gate=bool(item.get("gate", item.get("is_gate", False))),
                is_sensitive=bool(item.get("sensitive", item.get("is_sensitive", False))),
            ))

        self._restricted: list[tuple[int | None, list[tuple[float, float]]]] = []
        for item in config.get("restricted_polygons", []) or []:
            if isinstance(item, dict):
                points = item.get("points", item.get("polygon", []))
                zone = item.get("zone")
                zone = None if zone is None else int(zone)
            else:
                points, zone = item, None
            polygon = self._clean_polygon(points)
            if len(polygon) >= 3:
                self._restricted.append((zone, polygon))
        for zone_index, item in enumerate(zone_settings if isinstance(zone_settings, list) else []):
            if not isinstance(item, dict):
                continue
            for points in item.get("restricted_polygons", []) or []:
                polygon = self._clean_polygon(points)
                if len(polygon) >= 3:
                    self._restricted.append((zone_index, polygon))

    @staticmethod
    def _clean_polygon(points: Any) -> list[tuple[float, float]]:
        cleaned = []
        if not isinstance(points, (list, tuple)):
            return cleaned
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    cleaned.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    continue
        return cleaned

    def zone_index(self, x: float, y: float, frame_width: int, frame_height: int) -> int:
        row = min(self.rows - 1, max(0, int(float(y) * self.rows / max(1, frame_height))))
        column = min(self.columns - 1, max(0, int(float(x) * self.columns / max(1, frame_width))))
        return row * self.columns + column

    def zone(self, index: int) -> Zone:
        return self.zones[int(index)]

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
        inside = False
        previous_x, previous_y = polygon[-1]
        for current_x, current_y in polygon:
            crosses = (current_y > y) != (previous_y > y)
            if crosses:
                edge_x = (previous_x - current_x) * (y - current_y) / (previous_y - current_y) + current_x
                if x < edge_x:
                    inside = not inside
            previous_x, previous_y = current_x, current_y
        return inside

    def is_restricted(self, x: float, y: float, frame_width: int, frame_height: int, zone_index: int | None = None) -> bool:
        for polygon_zone, polygon in self._restricted:
            if polygon_zone is not None and zone_index is not None and polygon_zone != zone_index:
                continue
            if polygon and max(abs(value) for point in polygon for value in point) <= 1:
                scaled = [(px * frame_width, py * frame_height) for px, py in polygon]
            else:
                scaled = polygon
            if self._point_in_polygon(float(x), float(y), scaled):
                return True
        return False

    def restricted_polygon_count(self) -> int:
        return len(self._restricted)
