"""pytest 公共夹具：用随机权重 checkpoint 启动一个可测试的后端实例。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "backend"))


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    import torch

    from face_recog.models import build_model

    runtime = tmp_path_factory.mktemp("runtime")
    checkpoint = runtime / "best.pt"
    model = build_model("full")
    torch.save({"model": model.state_dict(), "variant": "full", "embedding_dim": 128}, checkpoint)

    # 必须在导入 backend 模块之前设置，模块级常量会读取这些环境变量
    os.environ["FACE_APP_CHECKPOINT"] = str(checkpoint)
    os.environ["FACE_APP_DB"] = str(runtime / "test.db")
    os.environ["FACE_APP_CONFIG"] = str(runtime / "config.json")
    os.environ["FACE_APP_ENROLL"] = str(runtime / "enroll")

    import app as backend_app
    import db

    db.init_db()
    backend_app.app.config.update(TESTING=True)
    return backend_app.app.test_client()
