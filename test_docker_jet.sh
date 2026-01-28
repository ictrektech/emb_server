#!/bin/bash

GPU_FLAG=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
  GPU_FLAG="--runtime=nvidia"
fi

docker run -dit \
  --name emb \
  $GPU_FLAG \
  -p 18000:18000 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 \
  -e MAX_WORKERS=3 \
  -e PER_MODEL_MAX=2 \
  -e IDLE_TIMEOUT_S=900 \
  -e MODEL_PORT_BASE=18008 \
  -e PYTHON_BIN=python3 \
  -v /home/jhu/dev/models/embs:/root/models \
  -v /home/jhu/dev/media:/root/media \
  -v /etc/localtime:/etc/localtime:ro \
  emb_server