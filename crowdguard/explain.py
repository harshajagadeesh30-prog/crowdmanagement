from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Any


def template_explanation(alert: dict[str, Any]) -> str:
    return " ".join(part.strip() for part in (
        str(alert.get("what", "")),
        str(alert.get("why_flagged", "")),
        f"Recommended action: {alert.get('recommended_action', 'Review the area with staff.')}",
    ) if part.strip())


def _numeric_tokens(value: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", value))


def explain_alert(alert: dict[str, Any], evidence_frames: list[str], config: dict[str, Any] | None = None) -> tuple[str, str]:
    """Ask Claude for evidence-grounded text; always retain a local template fallback."""
    fallback = template_explanation(alert)
    if alert.get("severity") not in {"High", "Medium"} or len(evidence_frames) < 2:
        return fallback, "template"
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return fallback, "template"

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        content: list[dict[str, Any]] = []
        for frame_path in evidence_frames[:3]:
            path = Path(frame_path)
            if not path.is_file():
                continue
            extension = path.suffix.lower()
            media_type = "image/png" if extension == ".png" else "image/jpeg"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}})
        if len([item for item in content if item["type"] == "image"]) < 2:
            return fallback, "template"
        metrics = alert.get("metrics", {})
        content.append({
            "type": "text",
            "text": (
                "Write exactly 3 short sentences: what is visible, why it may be dangerous, and what operators should do. "
                "Use only the supplied frames and numeric metrics. Do not invent measurements or numbers; any digits in the answer "
                "must already appear in the metrics JSON. Do not identify people or infer identity, age, gender, or ethnicity. "
                f"Alert type: {alert.get('type')}. Metrics JSON: {metrics}"
            ),
        })
        model = str((config or {}).get("anthropic_model", "claude-3-5-haiku-latest"))
        response = client.messages.create(
            model=model,
            max_tokens=180,
            messages=[{"role": "user", "content": content}],
        )
        text = " ".join(block.text.strip() for block in response.content if getattr(block, "type", None) == "text" and block.text.strip())
        allowed_numbers = _numeric_tokens(str(metrics))
        if not text or not _numeric_tokens(text).issubset(allowed_numbers):
            return fallback, "template"
        return text, "vision_ai"
    except Exception:
        return fallback, "template"
