"""后端 API 冒烟测试：随机权重也能验证路由、校验与配置读写。"""
from __future__ import annotations

import io

import numpy as np


def _blank_jpeg() -> bytes:
    import cv2

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["gallery"] == 0
    assert "thresholds" in payload


def test_config_roundtrip(client):
    updated = {"thresholds": {"考勤": 0.62, "门禁": 0.88}, "liveness_threshold": 0.15}
    response = client.put("/api/config", json=updated)
    assert response.status_code == 200

    payload = client.get("/api/config").get_json()
    assert payload["thresholds"]["考勤"] == 0.62
    assert payload["thresholds"]["门禁"] == 0.88
    assert payload["liveness_threshold"] == 0.15


def test_records_and_stats_start_empty(client):
    assert client.get("/api/records").get_json()["records"] == []
    stats = client.get("/api/stats").get_json()
    assert stats == {"users": 0, "today": 0, "passed": 0, "records": 0}
    assert client.get("/api/users").get_json()["users"] == []


def test_recognize_requires_image(client):
    response = client.post("/api/recognize", data={})
    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_recognize_rejects_undecodable_payload(client):
    response = client.post(
        "/api/recognize",
        data={"image": (io.BytesIO(b"not-an-image"), "bad.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["message"] == "无法解析图像"


def test_recognize_without_face_returns_friendly_message(client):
    response = client.post(
        "/api/recognize",
        data={"image": (io.BytesIO(_blank_jpeg()), "blank.jpg"), "scene": "考勤"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is False
    assert "未检测到人脸" in payload["message"]


def test_enroll_validates_fields(client):
    response = client.post("/api/enroll", data={"user_id": "", "name": ""})
    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_identify_respects_scene_threshold(client):
    import numpy as np

    from inference import FaceEngine

    engine = FaceEngine()
    vector = np.zeros(128, dtype=np.float32)
    vector[0] = 1.0
    engine.gallery = [{"user_id": "20230101", "name": "测试", "category": "学生"}]
    engine.matrix = np.stack([vector])

    match, score, _ = engine.identify(vector, "考勤")
    assert match is not None
    assert abs(score - 1.0) < 1e-6

    # 门禁阈值比考勤高 0.10，0.75 的相似度在门禁场景应被拒绝
    engine.thresholds = {"考勤": 0.7, "门禁": 0.8}
    engine.gallery = [{"user_id": "20230101", "name": "测试", "category": "学生"}]
    engine.matrix = np.stack([vector * 0.75])
    assert engine.identify(vector, "考勤")[0] is not None
    assert engine.identify(vector, "门禁")[0] is None
