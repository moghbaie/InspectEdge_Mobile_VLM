"""Inspection report schema (v1.0), strict validator, and tolerant model-output parser.

The schema mirrors section 5.3 of the engineering plan. Validation is deterministic and is
the gate for the ">= 99.5% JSON schema validity" acceptance criterion.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import jsonschema

SCHEMA_VERSION = "1.0"

# Closed taxonomy shared by the annotation mapper, SFT targets and evaluation.
DEFECT_TYPES = [
    "scratch",
    "crack",
    "chip_or_missing_material",
    "contamination_or_spot",
    "deformation_or_dent",
    "discoloration",
    "missing_or_misplaced_component",
    "foreign_object",
    "unknown",
]
INSPECTION_STATUS = ["no_visible_anomaly", "visible_anomaly", "uncertain", "unsupported_asset"]
QUALITY_ISSUES = ["blur", "noise", "low_light", "overexposure", "glare", "occlusion", "low_resolution", "compression"]

REPORT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "asset_category", "image_quality", "findings", "inspection_status"],
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "asset_category": {"type": "string", "minLength": 1},
        "image_quality": {
            "type": "object",
            "additionalProperties": False,
            "required": ["usable", "issues"],
            "properties": {
                "usable": {"type": "boolean"},
                "issues": {"type": "array", "items": {"enum": QUALITY_ISSUES}, "uniqueItems": True},
            },
        },
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "region_xyxy_normalized", "visible_evidence", "review_required"],
                "properties": {
                    "type": {"enum": DEFECT_TYPES},
                    "region_xyxy_normalized": {
                        "type": "array",
                        "items": {"type": "number", "minimum": 0, "maximum": 1},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "visible_evidence": {"type": "string", "minLength": 1, "maxLength": 200},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "review_required": {"type": "boolean"},
                },
            },
        },
        "inspection_status": {"enum": INSPECTION_STATUS},
    },
}

_VALIDATOR = jsonschema.Draft202012Validator(REPORT_SCHEMA)


@dataclass
class ParseResult:
    raw: str
    parsed: dict | None = None
    json_ok: bool = False
    schema_ok: bool = False
    errors: list[str] = field(default_factory=list)


def semantic_errors(report: dict) -> list[str]:
    """Cross-field rules that JSON Schema cannot express."""
    errs = []
    for i, f in enumerate(report.get("findings", [])):
        x1, y1, x2, y2 = f["region_xyxy_normalized"]
        if not (x1 < x2 and y1 < y2):
            errs.append(f"findings[{i}].region not x1<x2,y1<y2")
    status = report.get("inspection_status")
    if status == "no_visible_anomaly" and report.get("findings"):
        errs.append("status no_visible_anomaly but findings non-empty")
    if status == "visible_anomaly" and not report.get("findings"):
        errs.append("status visible_anomaly but findings empty")
    return errs


def validate_report(report: dict) -> list[str]:
    errs = [f"{'/'.join(map(str, e.path))}: {e.message}" for e in _VALIDATOR.iter_errors(report)]
    if not errs:
        errs += semantic_errors(report)
    return errs


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json_text(text: str) -> str | None:
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_model_output(text: str) -> ParseResult:
    res = ParseResult(raw=text)
    js = extract_json_text(text)
    if js is None:
        res.errors.append("no JSON object found")
        return res
    try:
        res.parsed = json.loads(js)
        res.json_ok = isinstance(res.parsed, dict)
    except json.JSONDecodeError as e:
        res.errors.append(f"json decode: {e}")
        return res
    if not res.json_ok:
        res.errors.append("top-level JSON is not an object")
        return res
    res.errors = validate_report(res.parsed)
    res.schema_ok = not res.errors
    return res


def canonical_json(report: dict) -> str:
    """Deterministic serialization used for SFT targets (stable key order, 3-dp coordinates)."""
    def _round(o):
        if isinstance(o, float):
            return round(o, 3)
        if isinstance(o, list):
            return [_round(v) for v in o]
        if isinstance(o, dict):
            return {k: _round(v) for k, v in o.items()}
        return o

    return json.dumps(_round(report), separators=(",", ":"), ensure_ascii=False)
