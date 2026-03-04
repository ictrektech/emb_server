#!/bin/bash
set -e

# 模型目录
export MODEL_ROOT="${MODEL_ROOT:-/root/models}"

# ===== 默认常驻 immich（canonical 名称）=====
# 如果外部没传，就默认 vit-b-16-siglip2__webli
export PRELOAD_MODELS="${PRELOAD_MODELS:-vit-b-16-siglip2__webli}"
export PIN_MODELS="${PIN_MODELS:-vit-b-16-siglip2__webli}"

# 其它参数也允许外部覆盖
export MODEL_PORT_BASE="${MODEL_PORT_BASE:-18008}"
export MAX_WORKERS="${MAX_WORKERS:-3}"
export PER_MODEL_MAX="${PER_MODEL_MAX:-2}"
export IDLE_TIMEOUT_S="${IDLE_TIMEOUT_S:-900}"
export EVICTOR_INTERVAL_S="${EVICTOR_INTERVAL_S:-60}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

exec uvicorn manager.app:app --host 0.0.0.0 --port "${PORT:-18000}"