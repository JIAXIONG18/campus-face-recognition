# HTTP API 参考

服务默认监听 `http://127.0.0.1:5000`，同时托管前端静态资源（同源，无 CORS 问题）。所有接口返回 `application/json`，上传接口使用 `multipart/form-data`。

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | [`/api/health`](#get-apihealth) | 服务状态、设备、阈值、底库规模 |
| `POST` | [`/api/recognize`](#post-apirecognize) | 识别一张人脸照片 |
| `POST` | [`/api/enroll`](#post-apienroll) | 录入人员与多张人脸 |
| `GET` | [`/api/users`](#get-apiusers) | 已录入人员列表 |
| `DELETE` | [`/api/users/<user_id>`](#delete-apiusersuser_id) | 删除人员及其特征 |
| `GET` | [`/api/records`](#get-apirecords) | 识别记录 |
| `GET` | [`/api/stats`](#get-apistats) | 统计计数 |
| `GET` / `PUT` | [`/api/config`](#getput-apiconfig) | 读取 / 更新阈值 |
| `GET` | `/uploads/<path>` | 访问注册照（`data/` 下的相对路径） |

错误约定：请求本身有问题返回 **400**；识别流程中"没检测到人脸""质量不合格"属于业务结果，返回 **200** 且 `success=false`，前端据此展示原因。

---

## GET /api/health

```bash
curl http://127.0.0.1:5000/api/health
```

```json
{
  "status": "ok",
  "device": "cpu",
  "threshold": 0.7,
  "thresholds": { "考勤": 0.7, "门禁": 0.8 },
  "liveness": true,
  "liveness_threshold": 0.0,
  "gallery": 6,
  "checkpoint": "best.pt"
}
```

| 字段 | 说明 |
|---|---|
| `device` | `cpu` 或 `cuda:0` |
| `threshold` | 来自 `experiments/metrics.json` 的基准阈值（0.70） |
| `thresholds` | 场景化阈值，可被 `GET/PUT /api/config` 覆盖 |
| `liveness` | 活体模型是否加载成功（权重缺失时为 `false`） |
| `gallery` | 当前底库人数（`face_feature` 去重后的可用特征） |
| `checkpoint` | 当前识别权重文件名，便于确认加载的是哪个模型 |

首次调用会触发模型加载，可能比后续请求慢 1–2 秒。

---

## POST /api/recognize

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `image` | file | 是 | JPEG/PNG，建议 ≤ 2 MB；上限 32 MB |
| `scene` | string | 否 | `考勤`（默认）或 `门禁`，决定使用哪个阈值 |

```bash
curl -X POST http://127.0.0.1:5000/api/recognize \
  -F "image=@face.jpg" \
  -F "scene=考勤"
```

**识别通过**

```json
{
  "success": true,
  "passed": true,
  "score": 0.8734,
  "threshold": 0.7,
  "liveness": 4.21,
  "liveness_pass": true,
  "elapsed_ms": 45.7,
  "match_ms": 0.12,
  "box": [312, 180, 168, 168],
  "user": { "user_id": "20230104", "name": "Liu Yang", "category": "学生" },
  "message": "识别通过"
}
```

**未注册人员**（`passed=false`，`user=null`）

```json
{ "success": true, "passed": false, "score": 0.4123, "threshold": 0.8,
  "liveness": 3.02, "liveness_pass": true, "elapsed_ms": 43.1, "match_ms": 0.1,
  "box": [300, 168, 170, 170], "user": null, "message": "未注册人员" }
```

**活体拦截**（不返回 `score`，直接拒绝并写入记录）

```json
{ "success": true, "passed": false, "liveness": -3.84, "liveness_pass": false,
  "elapsed_ms": 38.6, "message": "检测到非真实人脸，已拦截" }
```

**未检测到人脸 / 质量不合格**（HTTP 200，`success=false`）

```json
{ "success": false, "message": "未检测到人脸，请正对摄像头" }
{ "success": false, "message": "人脸太小，请靠近一些" }
{ "success": false, "message": "画面偏模糊，请保持稳定" }
```

**请求错误**（HTTP 400）

```json
{ "success": false, "message": "缺少图像" }
{ "success": false, "message": "无法解析图像" }
```

字段含义：`score` 为余弦相似度（0–1），`threshold` 为该场景实际使用的阈值，`box` 为检测框 `[x, y, w, h]`，`elapsed_ms` 为服务端总耗时，`match_ms` 为底库比对耗时。

---

## POST /api/enroll

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `user_id` | string | 是 | 学号/工号，主键 |
| `name` | string | 是 | 姓名 |
| `category` | string | 否 | `学生`（默认）/`教师`/`访客` |
| `images` | file[] | 是 | 3–5 张不同角度照片，可重复该字段 |

```bash
curl -X POST http://127.0.0.1:5000/api/enroll \
  -F "user_id=20230104" -F "name=Liu Yang" -F "category=学生" \
  -F "images=@a.jpg" -F "images=@b.jpg" -F "images=@c.jpg"
```

```json
{ "success": true, "saved": 3, "failed": 0, "message": "已录入 3 张人脸特征" }
```

- 每张成功照片都会对齐、裁剪并保存到 `data/enroll/<user_id>/`，特征以 BLOB 写入 `face_feature`；
- 未检测到人脸的照片计入 `failed`，不影响其余照片；
- 已存在的 `user_id` 会更新姓名与类别，特征累加（重复录入同一人多张会增强底库，建议每 3–5 张足够）；
- 录入后引擎自动 `reload()`，无需重启。

**参数缺失**（HTTP 400）

```json
{ "success": false, "message": "学号与姓名不能为空" }
{ "success": false, "message": "请至少上传一张人脸照片" }
```

---

## GET /api/users

```bash
curl http://127.0.0.1:5000/api/users
```

```json
{
  "users": [
    { "user_id": "20230104", "name": "Liu Yang", "category": "学生",
      "status": 1, "created_at": "2026-11-16T20:41:02", "photos": 4 }
  ]
}
```

`photos` 为该人员已入库的特征数。

---

## DELETE /api/users/<user_id>

```bash
curl -X DELETE http://127.0.0.1:5000/api/users/20230104
```

```json
{ "success": true }
```

删除 `user_info` 与 `face_feature` 中的记录并刷新底库；`data/enroll/<user_id>/` 下的照片文件保留，便于审计或人工清理。

---

## GET /api/records

| 参数 | 默认 | 说明 |
|---|---|---|
| `limit` | 50 | 返回条数，超过 500 会被截断 |

```bash
curl "http://127.0.0.1:5000/api/records?limit=20"
```

```json
{
  "records": [
    { "rec_id": 12, "user_id": "20230104", "name": "Liu Yang", "score": 0.8734,
      "passed": 1, "scene": "考勤", "liveness": 4.21, "rec_time": "2026-11-16T20:48:16" }
  ]
}
```

按 `rec_id` 倒序。活体拦截的记录 `user_id` 为 `null`、`name` 为「活体不通过」、`score` 为 0、`passed` 为 0。

---

## GET /api/stats

```bash
curl http://127.0.0.1:5000/api/stats
```

```json
{ "users": 6, "today": 6, "passed": 4, "records": 6 }
```

| 字段 | 含义 |
|---|---|
| `users` | 注册人数 |
| `today` | 今日识别次数（含拦截与未通过） |
| `passed` | 累计通过次数 |
| `records` | 累计记录数 |

前端的近 7 日趋势由 `/api/records` 的 `rec_time` 在浏览器端聚合，服务端不额外维护时间序列。

---

## GET/PUT /api/config

```bash
curl http://127.0.0.1:5000/api/config
```

```json
{ "thresholds": { "考勤": 0.7, "门禁": 0.8 }, "liveness_threshold": 0.0 }
```

```bash
curl -X PUT http://127.0.0.1:5000/api/config \
  -H "Content-Type: application/json" \
  -d '{"thresholds": {"考勤": 0.72, "门禁": 0.82}, "liveness_threshold": 0.5}'
```

- 仅接受 `考勤` / `门禁` 两个键，值须在 `[0, 1]`，否则忽略；
- 更新后立即写入 `data/config.json` 并作用于后续请求，无需重启；
- `liveness_threshold` 直接与活体 logit 比较，0 表示"比 0 更像真人即通过"。

---

## 状态码与错误语义

| 状态码 | 场景 |
|---|---|
| 200 | 正常返回；识别类接口即使"未通过"也是 200（业务结果，非错误） |
| 400 | 缺少 `image`、图像无法解析、录入参数缺失 |
| 404 | 访问不存在的静态资源 |
| 413 | 上传超过 32 MB（Flask 默认返回 HTML 错误页） |
| 500 | 模型加载失败、数据库异常（检查终端日志） |

## 集成建议

- 门禁硬件：调用 `/api/recognize` 时传 `scene=门禁`，以 `passed` 与 `user.user_id` 作为开闸条件；
- 批量导入：循环调用 `/api/enroll`，每人 3–5 张，注意串行以避免底库频繁重建；
- 客户端限流：服务端无内置限流，建议在网关层按 IP 限制 QPS；
- 生产部署：`FACE_APP_HOST=0.0.0.0` 配合 gunicorn（`gunicorn -w 1 --threads 4`，单 worker 以复用模型单例）。
