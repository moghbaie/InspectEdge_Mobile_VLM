"""Single source of truth for the VLM instruction and SFT target (used by baseline + Colab SFT)."""
from __future__ import annotations

from inspectedge.schemas.report import DEFECT_TYPES, QUALITY_ISSUES, SCHEMA_VERSION, canonical_json

INSTRUCTION = (
    "You are an industrial visual inspection assistant. The technician selected asset category: "
    "{asset}. Inspect ONLY what is visible in the photo. Do not guess hidden faults.\n"
    "Return ONLY one JSON object, no prose, with exactly these keys:\n"
    '{{"schema_version":"' + SCHEMA_VERSION + '","asset_category":"{asset}",'
    '"image_quality":{{"usable":true|false,"issues":[...]}},'
    '"findings":[{{"type":<defect type>,"region_xyxy_normalized":[x1,y1,x2,y2],'
    '"visible_evidence":<short literal description>,"review_required":true|false}}],'
    '"inspection_status":"no_visible_anomaly"|"visible_anomaly"|"uncertain"|"unsupported_asset"}}\n'
    "Allowed defect types: " + ", ".join(DEFECT_TYPES) + ".\n"
    "Allowed image quality issues: " + ", ".join(QUALITY_ISSUES) + ".\n"
    "Coordinates are fractions of image width/height in [0,1]. If there is no visible defect, "
    'use "findings":[] and "no_visible_anomaly". If evidence is insufficient, use "uncertain".'
)


def build_instruction(asset: str) -> str:
    return INSTRUCTION.format(asset=asset)


def build_messages(asset: str, target: dict | None = None) -> list[dict]:
    msgs = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": build_instruction(asset)}],
        }
    ]
    if target is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": canonical_json(target)}]})
    return msgs


def build_target(rec: dict, quality_issues: list[str] | None = None, max_findings: int = 3) -> dict:
    """Manifest record -> canonical report. Only annotation-backed facts are emitted."""
    findings = []
    if rec["label"] == "anomaly":
        types = rec["defect_types"] or ["unknown"]
        boxes = rec["boxes"][:max_findings] or []
        for i, b in enumerate(boxes):
            t = types[i] if i < len(types) else types[0]
            findings.append(
                {
                    "type": t,
                    "region_xyxy_normalized": b,
                    "visible_evidence": rec["defect_raw"][:120] or "visible surface anomaly",
                    "review_required": True,
                }
            )
    status = "visible_anomaly" if findings else "no_visible_anomaly"
    if rec["label"] == "anomaly" and not findings:
        status = "uncertain"  # anomaly label but no localisable region survives
    return {
        "schema_version": SCHEMA_VERSION,
        "asset_category": rec["category"],
        "image_quality": {"usable": True, "issues": list(quality_issues or [])},
        "findings": findings,
        "inspection_status": status,
    }
