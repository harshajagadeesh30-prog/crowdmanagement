from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PERSON_CLASS = 0
OBJECT_CLASSES = {24, 26, 28}
TRACKED_CLASSES = [PERSON_CLASS, *sorted(OBJECT_CLASSES)]


@dataclass(frozen=True)
class Detection:
    class_id: int
    track_id: int | None
    confidence: float
    bbox: tuple[float, float, float, float]


class DetectorTracker:
    """Thin YOLO tracking adapter for people and carried/left objects."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        model_name = str(config.get("model", "yolov8n.pt"))
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is required for video analysis. Install dependencies with: "
                "python -m pip install -r requirements.txt"
            ) from error
        self.model = YOLO(model_name)
        self.tracker = str(config.get("tracker", "bytetrack.yaml"))
        self.device = config.get("device")

    def track(self, frame: Any) -> list[Detection]:
        options: dict[str, Any] = {
            "source": frame,
            "persist": True,
            "classes": TRACKED_CLASSES,
            "verbose": False,
            "tracker": self.tracker,
        }
        if self.device:
            options["device"] = self.device
        results = self.model.track(**options)
        detections: list[Detection] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.xyxy is None:
                continue
            coordinates = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
            confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else [0.0] * len(coordinates)
            identifiers = None
            if boxes.id is not None:
                identifiers = boxes.id.cpu().numpy().astype(int).tolist()
            for index, box in enumerate(coordinates):
                class_id = int(classes[index]) if index < len(classes) else -1
                if class_id not in TRACKED_CLASSES:
                    continue
                track_id = int(identifiers[index]) if identifiers is not None and index < len(identifiers) else None
                detections.append(Detection(
                    class_id=class_id,
                    track_id=track_id,
                    confidence=float(confidences[index]) if index < len(confidences) else 0.0,
                    bbox=tuple(float(value) for value in box[:4]),
                ))
        return detections
