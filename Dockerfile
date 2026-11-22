# 校园人脸识别系统 —— 单容器镜像（CPU 推理）
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FACE_APP_HOST=0.0.0.0 \
    FACE_APP_PORT=5000

# OpenCV 运行时依赖（精简）
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir \
        torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# 运行时数据库与上传目录（建议挂载卷持久化）
RUN mkdir -p /app/data

EXPOSE 5000

CMD ["python", "code/backend/app.py"]
