"""Flask 后端服务与前端静态资源。"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT / "code"))

import config
import db
from inference import ENROLL_DIR, get_engine

FRONTEND = ROOT / "code" / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/api/health")
def health():
    engine = get_engine()
    return jsonify(
        {
            "status": "ok",
            "device": str(engine.device),
            "threshold": engine.threshold,
            "thresholds": engine.thresholds,
            "liveness": engine.liveness is not None,
            "liveness_threshold": engine.liveness_threshold,
            "gallery": len(engine.gallery),
            "checkpoint": str(engine.checkpoint.name),
        }
    )


@app.post("/api/recognize")
def recognize():
    file = request.files.get("image")
    if file is None:
        return jsonify({"success": False, "message": "缺少图像"}), 400
    scene = request.form.get("scene", "考勤")
    engine = get_engine()
    start = time.perf_counter()
    try:
        image = engine.decode(file.read())
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400

    vector, face, crop = engine.embed(image)
    if vector is None:
        return jsonify({"success": False, "message": "未检测到人脸，请正对摄像头"}), 200
    ok, reason = engine.quality(image, face)
    if not ok:
        return jsonify({"success": False, "message": reason}), 200

    liveness_score = engine.liveness_score(crop)
    is_live = liveness_score is None or liveness_score >= engine.liveness_threshold
    if not is_live:
        elapsed = (time.perf_counter() - start) * 1000
        db.add_record(None, "活体不通过", 0.0, False, scene, liveness_score)
        return jsonify(
            {
                "success": True,
                "passed": False,
                "liveness": round(liveness_score, 3),
                "liveness_pass": False,
                "elapsed_ms": round(elapsed, 1),
                "message": "检测到非真实人脸，已拦截",
            }
        )

    match, score, match_ms = engine.identify(vector, scene)
    elapsed = (time.perf_counter() - start) * 1000
    passed = match is not None
    db.add_record(
        match["user_id"] if match else None,
        match["name"] if match else "未注册",
        score, passed, scene, liveness_score,
    )
    payload = {
        "success": True,
        "passed": passed,
        "score": round(score, 4),
        "threshold": engine.thresholds.get(scene, engine.threshold),
        "liveness": round(liveness_score, 3),
        "liveness_pass": True,
        "elapsed_ms": round(elapsed, 1),
        "match_ms": round(match_ms, 2),
        "box": [int(v) for v in face[:4]],
        "user": {"user_id": match["user_id"], "name": match["name"], "category": match["category"]} if match else None,
        "message": "识别通过" if passed else "未注册人员",
    }
    return jsonify(payload)


@app.post("/api/enroll")
def enroll():
    user_id = (request.form.get("user_id") or "").strip()
    name = (request.form.get("name") or "").strip()
    category = request.form.get("category", "学生")
    if not user_id or not name:
        return jsonify({"success": False, "message": "学号与姓名不能为空"}), 400
    files = request.files.getlist("images")
    if not files:
        return jsonify({"success": False, "message": "请至少上传一张人脸照片"}), 400

    engine = get_engine()
    saved, failed = 0, []
    db.upsert_user(user_id, name, category)
    folder = ENROLL_DIR / user_id
    folder.mkdir(parents=True, exist_ok=True)
    for index, file in enumerate(files):
        blob = file.read()
        try:
            image = engine.decode(blob)
        except ValueError:
            failed.append(index)
            continue
        vector, face, crop = engine.embed(image)
        if vector is None:
            failed.append(index)
            continue
        path = folder / f"{int(time.time() * 1000)}_{index}.jpg"
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        db.add_feature(user_id, vector.tobytes(), str(path.relative_to(ROOT)))
        saved += 1

    engine.reload()
    return jsonify(
        {
            "success": saved > 0,
            "saved": saved,
            "failed": len(failed),
            "message": f"已录入 {saved} 张人脸特征" if saved else "未检测到可用人脸",
        }
    )


@app.get("/api/records")
def records():
    limit = min(int(request.args.get("limit", 50)), 500)
    return jsonify({"records": db.list_records(limit)})


@app.get("/api/users")
def users():
    return jsonify({"users": db.list_users()})


@app.delete("/api/users/<user_id>")
def delete_user(user_id: str):
    db.delete_user(user_id)
    get_engine().reload()
    return jsonify({"success": True})


@app.get("/api/config")
def get_config():
    engine = get_engine()
    return jsonify({"thresholds": engine.thresholds, "liveness_threshold": engine.liveness_threshold})


@app.put("/api/config")
def put_config():
    engine = get_engine()
    data = request.get_json(silent=True) or {}
    thresholds = data.get("thresholds")
    if isinstance(thresholds, dict):
        for key, value in thresholds.items():
            if key in ("考勤", "门禁"):
                value = float(value)
                if 0.0 <= value <= 1.0:
                    engine.thresholds[key] = value
    if "liveness_threshold" in data:
        engine.liveness_threshold = float(data["liveness_threshold"])
    config.save({"thresholds": engine.thresholds, "liveness_threshold": engine.liveness_threshold})
    return jsonify({"success": True, "thresholds": engine.thresholds, "liveness_threshold": engine.liveness_threshold})


@app.get("/api/stats")
def stats():
    return jsonify(db.stats())


@app.get("/uploads/<path:path>")
def uploads(path: str):
    return send_from_directory(ROOT / "data", path)


if __name__ == "__main__":
    db.init_db()
    get_engine()
    host = os.environ.get("FACE_APP_HOST", "127.0.0.1")
    port = int(os.environ.get("FACE_APP_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
