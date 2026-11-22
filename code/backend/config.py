"""运行时配置（阈值等），持久化到 data/config.json。"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get("FACE_APP_CONFIG", ROOT / "data" / "config.json"))

DEFAULTS = {
    "thresholds": {"考勤": 0.70, "门禁": 0.80},
    "liveness_threshold": 0.0,
}


def load() -> dict:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            thresholds = {**DEFAULTS["thresholds"], **data.get("thresholds", {})}
            return {
                "thresholds": thresholds,
                "liveness_threshold": float(data.get("liveness_threshold", DEFAULTS["liveness_threshold"])),
            }
        except Exception:
            pass
    return {"thresholds": dict(DEFAULTS["thresholds"]), "liveness_threshold": DEFAULTS["liveness_threshold"]}


def save(data: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
