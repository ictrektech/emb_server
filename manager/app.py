# manager/app.py
import os
import time
import subprocess
import requests
from typing import Dict, List
import threading

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

# -------- 配置 --------
MODEL_PORT_BASE = int(os.getenv("MODEL_PORT_BASE", 18008))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 3))
IDLE_TIMEOUT_S = int(os.getenv("IDLE_TIMEOUT_S", 900))  # 15 min
PER_MODEL_MAX = int(os.getenv("PER_MODEL_MAX", 2))
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python")  # 用 python -m uvicorn 更稳
EVICTOR_INTERVAL_S = int(os.getenv("EVICTOR_INTERVAL_S", 60))  # 每 60s 扫一遍

# 声明可用模型（用于白名单 & 类型标注）
MODEL_SPECS: Dict[str, Dict] = {
    "bge-m3": {"type": "text"},
    "bge-vl": {"type": "vl"},
    # ---- Qwen3 Embedding 兼容别名（新增）----
    "qwen": {"type": "text"},
    "qwen-embedding": {"type": "text"},
    "qwen3-embedding": {"type": "text"},
    "qwen3-embedding-0.6b": {"type": "text"},
    "qwen/qwen3-embedding-0.6b": {"type": "text"},
}

# -------- 状态 --------
class Worker:
    def __init__(self, model: str, port: int, proc: subprocess.Popen):
        self.model = model
        self.port = port
        self.proc = proc
        self.last_used = time.time()
        self.inflight = 0

workers: Dict[str, List[Worker]] = {}   # model -> [Worker,...]
port_alloc = set()

app = FastAPI()

# -------- 工具 --------
def find_free_port() -> int:
    p = MODEL_PORT_BASE
    while p in port_alloc:
        p += 1
    port_alloc.add(p)
    return p

def wait_healthy(w: Worker, timeout: int = 60):
    start = time.time()
    url = f"http://127.0.0.1:{w.port}/health"
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise HTTPException(500, f"Worker failed to become healthy on {url}")

def spawn_worker(model: str) -> Worker:
    # 全局数量限制
    total = sum(len(v) for v in workers.values())
    if total >= MAX_WORKERS:
        evict_idle()
    if total >= MAX_WORKERS:
        raise HTTPException(503, "Too many workers loaded")

    # 每模型副本限制
    if len(workers.get(model, [])) >= PER_MODEL_MAX:
        raise HTTPException(429, f"Too many replicas for {model}")

    port = find_free_port()
    env = os.environ.copy()
    env["MODEL_NAME"] = model
    env["PORT"] = str(port)

    # 用 uvicorn 起 Worker：worker.app:app
    proc = subprocess.Popen(
        [PYTHON_BIN, "-m", "uvicorn", "worker.app:app", "--host", "127.0.0.1", "--port", str(port)],
        env=env
    )

    w = Worker(model, port, proc)
    workers.setdefault(model, []).append(w)
    wait_healthy(w)
    return w

def pick_worker(model: str) -> Worker:
    arr = workers.get(model, [])
    if arr:
        # 简单负载均衡：优先 inflight 小的、最近使用的
        arr = sorted(arr, key=lambda x: (x.inflight, -x.last_used))
        return arr[0]
    return spawn_worker(model)

def evict_idle():
    # 先找空闲超时的
    idle: List[Worker] = []
    now = time.time()
    for arr in workers.values():
        for w in arr:
            if w.inflight == 0 and (now - w.last_used) > IDLE_TIMEOUT_S:
                idle.append(w)

    # 如果没有，LRU 驱逐一个（不驱逐正在处理的）
    if not idle:
        allw = [w for arr in workers.values() for w in arr if w.inflight == 0]
        if not allw:
            return
        idle = [sorted(allw, key=lambda x: x.last_used)[0]]

    for w in idle:
        try:
            requests.post(f"http://127.0.0.1:{w.port}/shutdown", timeout=1)
        except Exception:
            pass
        try:
            w.proc.terminate()
        except Exception:
            pass
        workers[w.model].remove(w)
        port_alloc.discard(w.port)

# -------- API --------
class TextInput(BaseModel):
    input: list
    normalize: bool | None = True
    pooling: str | None = None
    fp16: bool | None = True
    batch: int | None = 32

@app.post("/embed")
def embed(model: str, body: dict = Body(...)):
    # 模型白名单校验
    if model not in MODEL_SPECS:
        raise HTTPException(404, f"Unknown model {model}")

    w = pick_worker(model)
    w.inflight += 1
    try:
        r = requests.post(f"http://127.0.0.1:{w.port}/embed", json=body, timeout=120)
        w.last_used = time.time()
        # r.raise_for_status()  # 将 worker 的错误显式抛出
        if r.status_code >= 400:
            # 把 worker 返回的具体错误正文带出来，便于定位
            raise HTTPException(r.status_code, f"worker@{w.port} -> {r.text}")
        return r.json()
    finally:
        w.inflight -= 1

@app.get("/metrics")
def metrics():
    return {
        "total": sum(len(v) for v in workers.values()),
        "workers": [
            {"model": w.model, "port": w.port, "inflight": w.inflight, "last_used": w.last_used}
            for arr in workers.values() for w in arr
        ]
    }

def _evictor_loop():
    while True:
        try:
            evict_idle()
        except Exception:
            pass
        time.sleep(EVICTOR_INTERVAL_S)

@app.on_event("startup")
def _start_evictor():
    t = threading.Thread(target=_evictor_loop, daemon=True)
    t.start()

@app.post("/evict")
def evict_now():
    evict_idle()
    return {"ok": True}