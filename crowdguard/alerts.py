from __future__ import annotations

from collections import defaultdict
from math import hypot
from typing import Any


OBJECT_CLASS_NAMES = {24: "backpack", 26: "handbag", 28: "suitcase"}
ACTIONS = {
    "Crush Risk": "Pause entry and open another gate.",
    "Abandoned Object": "Send staff to inspect the item and keep the crowd away.",
    "Loitering": "Send staff to check the sensitive area.",
    "Restricted Zone Entry": "Pause access and send staff to the restricted area.",
    "Panic Running": "Open another gate and slow incoming entry.",
    "Wrong-Way Movement": "Redirect the flow and pause entry at the affected gate.",
    "Fallen Person": "Send staff and create space around the person.",
}


def _base_metrics(row: dict[str, Any], management: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = {
        "count": int(row["count"]),
        "speed_px_s": round(float(row["speed"]), 2),
        "counterflow": round(float(row["counterflow"]), 3),
        "chaos": round(float(row["chaos"]), 3),
        "risk": round(float(row["risk"]), 2),
    }
    if management:
        metrics.update(management)
    return metrics


class AlertManager:
    """Merge repeated same-type/same-zone observations inside a time window."""

    def __init__(self, merge_seconds: int = 10):
        self.merge_seconds = max(0, int(merge_seconds))
        self.alerts: list[dict[str, Any]] = []
        self._last_by_key: dict[tuple[str, int], dict[str, Any]] = {}

    def add(self, alert_type: str, severity: str, sec: int, zone: int, what: str, why: str, evidence_frame: str, metrics: dict[str, Any]) -> dict[str, Any]:
        key = (alert_type, int(zone))
        previous = self._last_by_key.get(key)
        if previous is not None and int(sec) - int(previous["_last_seen_sec"]) <= self.merge_seconds:
            previous["_last_seen_sec"] = int(sec)
            if evidence_frame and evidence_frame not in previous["evidence_frames"]:
                previous["evidence_frames"].append(evidence_frame)
                previous["evidence_frames"] = previous["evidence_frames"][-3:]
            previous["metrics"].update(metrics)
            return previous

        alert = {
            "id": f"{alert_type.lower().replace(' ', '-')}-{int(sec)}-{int(zone)}",
            "type": alert_type,
            "severity": severity,
            "sec": int(sec),
            "zone": int(zone),
            "what": what,
            "why_flagged": why,
            "evidence_frames": [evidence_frame] if evidence_frame else [],
            "recommended_action": ACTIONS[alert_type],
            "metrics": dict(metrics),
            "explanation_source": "template",
            "_last_seen_sec": int(sec),
        }
        self.alerts.append(alert)
        self._last_by_key[key] = alert
        return alert

    def output(self) -> list[dict[str, Any]]:
        return [{key: value for key, value in alert.items() if not key.startswith("_")} for alert in self.alerts]


def evaluate_alerts(
    rows: list[dict[str, Any]],
    observations_by_second: dict[int, list[Any]],
    zones: Any,
    baseline: dict[int, dict[str, float]],
    management: dict[tuple[int, int], dict[str, Any]],
    frame_size: tuple[int, int],
    config: dict[str, Any],
    evidence_frame_for_sec,
) -> list[dict[str, Any]]:
    thresholds = config.get("thresholds", {})
    thresholds = thresholds if isinstance(thresholds, dict) else {}
    def setting(key: str, default: float) -> float:
        try:
            return float(thresholds.get(key, default))
        except (TypeError, ValueError):
            return default

    abandoned_seconds = setting("abandoned_seconds", config.get("abandoned_seconds", 60))
    clearance_px = setting("person_clearance_px", config.get("person_clearance_px", 120))
    loiter_seconds = setting("loiter_seconds", config.get("loiter_seconds", 30))
    loiter_radius_px = setting("loiter_radius_px", config.get("loiter_radius_px", 50))
    min_move = setting("min_move_px_s", 12)
    high_risk_threshold = setting("high_risk", 70)
    crowd_high_z = setting("crowd_high_z", 1.5)
    speed_drop_z = setting("speed_drop_z", 1.0)
    high_counterflow = setting("counterflow_high", 0.35)
    high_chaos = setting("chaos_high", 0.55)
    panic_multiple = setting("panic_speed_multiple", 1.8)
    panic_min_tracks = int(setting("panic_min_tracks", 3))
    wrong_way_seconds = int(setting("wrong_way_seconds", 3))
    fallen_seconds = setting("fallen_still_seconds", 5)
    fallen_speed = setting("fallen_speed_px_s", min_move)
    restricted_high_risk = setting("restricted_high_risk", 85)
    baseline_seconds = int(config.get("_effective_baseline_seconds", config.get("baseline_seconds", 30)))
    width, height = frame_size
    manager = AlertManager(int(setting("merge_seconds", 10)))
    wrong_way_run: dict[int, int] = defaultdict(int)

    for row in rows:
        sec, zone_index = int(row["sec"]), int(row["zone"])
        zone = zones.zone(zone_index)
        stats = baseline.get(zone_index, {})
        count = float(row["count"])
        speed = float(row["speed"])
        counterflow = float(row["counterflow"])
        chaos = float(row["chaos"])
        risk = float(row["risk"])
        facts = _base_metrics(row, management.get((sec, zone_index)))
        evidence = evidence_frame_for_sec(sec)
        people = [item for item in observations_by_second.get(sec, []) if item.class_id == 0 and item.zone == zone_index]

        density_high = count >= stats.get("count_mean", 0.0) + crowd_high_z * max(1.0, stats.get("count_std", 1.0))
        speed_dropped = speed <= stats.get("speed_mean", 0.0) - speed_drop_z * max(5.0, stats.get("speed_std", 5.0))
        flow_or_chaos = counterflow >= high_counterflow or chaos >= high_chaos
        if sec >= baseline_seconds and density_high and speed_dropped and flow_or_chaos:
            severity = "High" if risk >= high_risk_threshold else "Medium"
            manager.add(
                "Crush Risk", severity, sec, zone_index,
                f"Measured {int(count)} people in {zone.name}; mean movement speed was {speed:.1f} px/s.",
                f"Measured counterflow was {counterflow:.2f}, chaos {chaos:.2f}, and risk {risk:.1f}; density, speed, and movement indicators crossed configured warning conditions.",
                evidence, facts,
            )

        fast_tracks = [person for person in people if person.speed_px_s > max(min_move, stats.get("speed_mean", 0.0) * panic_multiple)]
        if sec >= baseline_seconds and len(fast_tracks) >= panic_min_tracks:
            manager.add(
                "Panic Running", "High", sec, zone_index,
                f"{len(fast_tracks)} measured tracks exceeded the zone's running-speed condition among {len(people)} people.",
                f"Mean zone speed was {speed:.1f} px/s, with measured chaos {chaos:.2f} and risk {risk:.1f}.",
                evidence, {**facts, "fast_tracks": len(fast_tracks), "fastest_speed_px_s": round(max(item.speed_px_s for item in fast_tracks), 2)},
            )

        if zone.is_gate and sec >= baseline_seconds and counterflow >= high_counterflow:
            wrong_way_run[zone_index] += 1
        else:
            wrong_way_run[zone_index] = 0
        if zone.is_gate and wrong_way_run[zone_index] >= wrong_way_seconds:
            manager.add(
                "Wrong-Way Movement", "Medium", sec, zone_index,
                f"Measured counterflow reached {counterflow:.2f} at gate {zone.name}.",
                f"Counterflow remained above its configured condition for {wrong_way_run[zone_index]} measured seconds; zone risk was {risk:.1f}.",
                evidence, {**facts, "sustained_seconds": wrong_way_run[zone_index]},
            )

        persons = [item for item in observations_by_second.get(sec, []) if item.class_id == 0]
        for item in observations_by_second.get(sec, []):
            if item.class_id in OBJECT_CLASS_NAMES and item.stationary_seconds >= abandoned_seconds:
                nearest = min((hypot(item.center[0] - person.center[0], item.center[1] - person.center[1]) for person in persons), default=float("inf"))
                if nearest > clearance_px:
                    object_metrics = {
                        **facts,
                        "object_class_id": int(item.class_id),
                        "stationary_seconds": round(float(item.stationary_seconds), 2),
                        "nearest_person_distance_px": None if nearest == float("inf") else round(nearest, 2),
                    }
                    nearest_text = "no person detection" if nearest == float("inf") else f"nearest detected person was {nearest:.1f} px away"
                    manager.add(
                        "Abandoned Object", "Medium", sec, zone_index,
                        f"A tracked {OBJECT_CLASS_NAMES[item.class_id]} remained stationary for {item.stationary_seconds:.1f} s.",
                        f"The nearest detected person was {nearest_text}; measured zone occupancy was {int(count)} and risk was {risk:.1f}.",
                        evidence, object_metrics,
                    )

            if item.class_id != 0:
                continue
            if zone.is_sensitive and item.stationary_seconds >= loiter_seconds and item.confinement_radius_px <= loiter_radius_px:
                severity = "Medium" if risk >= high_risk_threshold else "Low"
                manager.add(
                    "Loitering", severity, sec, zone_index,
                    f"A tracked person remained within {item.confinement_radius_px:.1f} px of a local area for {item.stationary_seconds:.1f} s.",
                    f"The zone is configured as sensitive; measured occupancy was {int(count)} and zone risk was {risk:.1f}.",
                    evidence, {**facts, "stationary_seconds": round(float(item.stationary_seconds), 2), "confinement_radius_px": round(float(item.confinement_radius_px), 2)},
                )

            if zones.is_restricted(item.center[0], item.center[1], width, height, zone_index):
                severity = "High" if risk >= restricted_high_risk else "Medium"
                manager.add(
                    "Restricted Zone Entry", severity, sec, zone_index,
                    f"A tracked person center was measured inside a configured restricted polygon in {zone.name}.",
                    f"Measured zone occupancy was {int(count)}, mean speed {speed:.1f} px/s, and risk {risk:.1f}.",
                    evidence, {**facts, "restricted_polygon_count": zones.restricted_polygon_count()},
                )

            box_width = max(0.0, float(item.bbox[2] - item.bbox[0]))
            box_height = max(0.0, float(item.bbox[3] - item.bbox[1]))
            box_ratio = box_width / max(1.0, box_height)
            if box_ratio > 1.2 and item.speed_px_s <= fallen_speed and item.stationary_seconds >= fallen_seconds:
                manager.add(
                    "Fallen Person", "Medium", sec, zone_index,
                    f"A tracked person had a measured box width-to-height ratio of {box_ratio:.2f} and remained nearly still for {item.stationary_seconds:.1f} s.",
                    f"Measured zone occupancy was {int(count)} and zone risk was {risk:.1f}; this is an operator-review indicator, not a confirmed event.",
                    evidence, {**facts, "box_width_px": round(box_width, 2), "box_height_px": round(box_height, 2), "box_ratio": round(box_ratio, 3), "stationary_seconds": round(float(item.stationary_seconds), 2)},
                )

    return sorted(manager.output(), key=lambda item: (item["sec"], item["zone"], item["type"]))
