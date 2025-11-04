# Dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/workspace/.hf \
    HF_HOME=/workspace/.hf \
    HF_ENDPOINT=https://hf-mirror.com

RUN apt-get update && apt-get install -y python3 python3-pip git ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6 && \
    ln -s /usr/bin/python3 /usr/bin/python

# 你可以锁定版本以保证可复现
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
RUN pip install -U --no-cache-dir transformers FlagEmbedding uvicorn fastapi

WORKDIR /app
COPY manager/ /app/manager/
COPY worker/ /app/worker/

# 默认启动 Manager
ENV HOST=0.0.0.0 PORT=18000
EXPOSE 8000
CMD ["uvicorn", "manager.app:app", "--host", "0.0.0.0", "--port", "18000"]