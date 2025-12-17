# manager/app.py
import os
import time
import subprocess
import requests
from typing import Dict, List
import threading
from collections import defaultdict

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
    "siglip2-base-patch16-224": {"type": "vl"},
    "siglip2-base-patch16-256": {"type": "vl"},
    "siglip2-base-patch16-384": {"type": "vl"},
    "siglip2-base-patch16-512": {"type": "vl"},
    "siglip2-large-patch16-256": {"type": "vl"},
    "siglip2-large-patch16-384": {"type": "vl"},
    "siglip2-large-patch16-512": {"type": "vl"},
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
model_locks = defaultdict(threading.Lock)  # 每个模型一个锁
start_events: Dict[str, threading.Event] = {}  # 冷启动进行中的事件

app = FastAPI()

# -------- 工具 --------
def find_free_port() -> int:
    p = MODEL_PORT_BASE
    while p in port_alloc:
        p += 1
    port_alloc.add(p)
    return p

def wait_healthy(w: Worker, timeout: int = 90):
    import time, requests
    url = f"http://127.0.0.1:{w.port}/health"
    start = time.time()
    ok_in_a_row = 0
    last_err = None
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=1.0)
            if r.status_code == 200:
                ok_in_a_row += 1
                if ok_in_a_row >= 2:
                    # 给 uvicorn 再 200ms 缓冲，避免刚绑定端口立刻被并发打爆
                    time.sleep(0.2)
                    return
                time.sleep(0.1)
                continue
            last_err = f"status={r.status_code}"
        except Exception as e:
            last_err = str(e)
        ok_in_a_row = 0
        time.sleep(0.3)
    raise HTTPException(500, f"Worker failed to become healthy on {url}; last_err={last_err}")

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

    proc = None
    try:
        # 启动 worker
        proc = subprocess.Popen(
            [PYTHON_BIN, "-m", "uvicorn", "worker.app:app", "--host", "127.0.0.1", "--port", str(port)],
            env=env
        )

        # 注册 worker
        w = Worker(model, port, proc)
        workers.setdefault(model, []).append(w)

        # 等健康
        wait_healthy(w)
        return w

    except Exception as e:
        # === 关键：失败要清理 ===
        try:
            if proc is not None:
                proc.terminate()
        except Exception:
            pass

        # 从 worker 表删除
        if model in workers:
            workers[model] = [x for x in workers[model] if x.port != port]
            if not workers[model]:
                workers.pop(model, None)

        # 端口释放
        port_alloc.discard(port)

        # 抛出错误
        raise HTTPException(503, f"Failed to spawn worker for {model}: {e}")

def pick_worker(model: str) -> Worker:
    # 先看有没有可用实例
    arr = workers.get(model, [])
    if arr:
        arr = sorted(arr, key=lambda x: (x.inflight, -x.last_used))
        return arr[0]

    # 没有 → 串行化冷启动
    lock = model_locks[model]
    with lock:
        # 二次检查（可能在等锁时已被别人拉起）
        arr = workers.get(model, [])
        if arr:
            arr = sorted(arr, key=lambda x: (x.inflight, -x.last_used))
            return arr[0]

        # 若没人启动，当前线程成为“发起者”
        ev = start_events.get(model)
        if ev is None:
            ev = threading.Event()
            start_events[model] = ev
            try:
                w = spawn_worker(model)
                return w
            finally:
                # 无论成功/失败，都要释放等待者
                ev.set()
                start_events.pop(model, None)
        else:
            # 已有发起者在启动，当前线程等待结果
            # 超时时间与 wait_healthy 对齐或略大
            ev.wait(timeout=90)

            arr = workers.get(model, [])
            if not arr:
                # 冷启动失败，给出“正在预热失败”的 503，比 500 友好
                raise HTTPException(503, f"{model} is warming up but failed to be ready; please retry")
            arr = sorted(arr, key=lambda x: (x.inflight, -x.last_used))
            return arr[0]

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
    # 白名单
    if model not in MODEL_SPECS:
        raise HTTPException(404, f"Unknown model {model}")

    def _request_with_worker(w: Worker):
        r = requests.post(f"http://127.0.0.1:{w.port}/embed", json=body, timeout=120)
        w.last_used = time.time()
        return r

    # 取/起 worker
    w = pick_worker(model)
    w.inflight += 1
    try:
        try:
            r = _request_with_worker(w)
        except requests.exceptions.ConnectionError:
            # === 关键：把疑似半启动/已崩的实例摘掉并重试一次 ===
            try:
                try:
                    requests.post(f"http://127.0.0.1:{w.port}/shutdown", timeout=0.5)
                except Exception:
                    pass
                try:
                    w.proc.terminate()
                except Exception:
                    pass
            finally:
                # 清理注册与端口占用
                workers[w.model].remove(w)
                port_alloc.discard(w.port)

            # 重新拉起一个再打一次
            w2 = spawn_worker(model)
            w2.inflight += 1
            try:
                r = _request_with_worker(w2)
            finally:
                w2.inflight -= 1

        # 透传下游错误文本，方便定位
        if r.status_code >= 400:
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


# ===== 强制回收：不管 inflight 是否在工作，全部干掉 =====
def evict_all(force_kill: bool = False):
    """
    强制回收所有 worker（无视 inflight）：
    - 尝试调用 /shutdown（优雅退出）
    - terminate
    - 可选：kill（force_kill=True 时）
    - 清理 workers 注册与 port_alloc
    """
    # 先扁平化拷贝，避免遍历时修改 dict/list
    all_workers: List[Worker] = [w for arr in workers.values() for w in arr]

    # 先尽力优雅 shutdown（不等待太久）
    for w in all_workers:
        try:
            requests.post(f"http://127.0.0.1:{w.port}/shutdown", timeout=0.5)
        except Exception:
            pass

    # terminate + 可选 kill
    for w in all_workers:
        try:
            if w.proc and w.proc.poll() is None:
                w.proc.terminate()
        except Exception:
            pass

    # 稍微等一下让 terminate 生效
    for w in all_workers:
        try:
            if w.proc and w.proc.poll() is None:
                w.proc.wait(timeout=1.5)
        except Exception:
            pass

    if force_kill:
        for w in all_workers:
            try:
                if w.proc and w.proc.poll() is None:
                    w.proc.kill()
            except Exception:
                pass

    # 清理注册表与端口占用
    for w in all_workers:
        try:
            if w.model in workers and w in workers[w.model]:
                workers[w.model].remove(w)
                if not workers[w.model]:
                    workers.pop(w.model, None)
        except Exception:
            pass
        try:
            port_alloc.discard(w.port)
        except Exception:
            pass


@app.post("/evict_all")
def evict_all_now(force_kill: bool = False):
    """
    强制清理全部 worker（无视 inflight）。
    force_kill=false：terminate 为主（默认）
    force_kill=true：terminate 后仍不退出则 kill
    """
    evict_all(force_kill=force_kill)
    return {"ok": True, "force_kill": force_kill}