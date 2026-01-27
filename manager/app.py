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
    # ---- SigLip2 family (transformers) ----
    "siglip2-base-patch16-224": {"type": "vl"},
    "siglip2-base-patch16-256": {"type": "vl"},
    "siglip2-base-patch16-384": {"type": "vl"},
    "siglip2-base-patch16-512": {"type": "vl"},
    "siglip2-large-patch16-256": {"type": "vl"},
    "siglip2-large-patch16-384": {"type": "vl"},
    "siglip2-large-patch16-512": {"type": "vl"},
    # ---- open_clip pretrained local-dir ----
    "openclip-siglip2-vit-b-16": {"type": "vl"},
    "openclip-vit-b-16-siglip2": {"type": "vl"},
    "vit-b-16-siglip2": {"type": "vl"},
    "baai/bge-vl-large": {"type": "vl"},

    # ---- Immich ONNX SigLIP2 (ViT-B-16-SigLIP2__webli) ----
    # 你可以用以下任意名字请求，worker 会映射到 HF repo immich-app/ViT-B-16-SigLIP2__webli
    "immich-vit-b-16-siglip2__webli": {"type": "vl"},
    "vit-b-16-siglip2__webli": {"type": "vl"},
    "immich-app/vit-b-16-siglip2__webli": {"type": "vl"},
    "immich-app/vit-b-16-siglip2__webli@onnx": {"type": "vl"},
}

# -------- 状态 --------
class Worker:
    def __init__(self, model: str, port: int, proc: subprocess.Popen):
        self.model = model  # canonical model name
        self.port = port
        self.proc = proc
        self.last_used = time.time()
        self.inflight = 0

workers: Dict[str, List[Worker]] = {}   # canonical_model -> [Worker,...]
port_alloc = set()
model_locks = defaultdict(threading.Lock)  # 每个模型一个锁
start_events: Dict[str, threading.Event] = {}  # 冷启动进行中的事件

app = FastAPI()

# 关键：保护 workers / port_alloc / start_events 的并发读写
workers_lock = threading.Lock()


# -------- Canonicalize model name --------
def canonical_model_name(model: str) -> str:
    """
    将多个别名统一到一个 canonical key，避免同一个模型因为不同字符串被当成多个模型启动多份 worker。

    例：
      - immich-vit-b-16-siglip2__webli
      - immich-app/vit-b-16-siglip2__webli
      - immich-app/vit-b-16-siglip2__webli@onnx
    都归一为：
      - vit-b-16-siglip2__webli

    同理：openclip 的一些别名也可归一。
    """
    m = (model or "").strip().lower()
    alias = {
        # immich onnx
        "immich-vit-b-16-siglip2__webli": "vit-b-16-siglip2__webli",
        "immich-app/vit-b-16-siglip2__webli": "vit-b-16-siglip2__webli",
        "immich-app/vit-b-16-siglip2__webli@onnx": "vit-b-16-siglip2__webli",

        # openclip aliases
        "openclip-siglip2-vit-b-16": "openclip-vit-b-16-siglip2",
        "vit-b-16-siglip2": "openclip-vit-b-16-siglip2",
    }
    return alias.get(m, model)


# -------- 工具 --------
def find_free_port() -> int:
    with workers_lock:
        p = MODEL_PORT_BASE
        while p in port_alloc:
            p += 1
        port_alloc.add(p)
        return p


def wait_healthy(w: Worker, timeout: int = 90):
    import time as _time
    url = f"http://127.0.0.1:{w.port}/health"
    start = _time.time()
    ok_in_a_row = 0
    last_err = None
    while _time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=1.0)
            if r.status_code == 200:
                ok_in_a_row += 1
                if ok_in_a_row >= 2:
                    _time.sleep(0.2)
                    return
                _time.sleep(0.1)
                continue
            last_err = f"status={r.status_code}"
        except Exception as e:
            last_err = str(e)
        ok_in_a_row = 0
        _time.sleep(0.3)
    raise HTTPException(500, f"Worker failed to become healthy on {url}; last_err={last_err}")


def spawn_worker(model: str) -> Worker:
    """
    model：必须是 canonical name（外部调用前先 canonical_model_name）
    """
    # 全局数量限制
    with workers_lock:
        total = sum(len(v) for v in workers.values())
    if total >= MAX_WORKERS:
        evict_idle()
    with workers_lock:
        total = sum(len(v) for v in workers.values())
    if total >= MAX_WORKERS:
        raise HTTPException(503, "Too many workers loaded")

    # 每模型副本限制
    with workers_lock:
        if len(workers.get(model, [])) >= PER_MODEL_MAX:
            raise HTTPException(429, f"Too many replicas for {model}")

    port = find_free_port()
    env = os.environ.copy()
    env["MODEL_NAME"] = model
    env["PORT"] = str(port)

    proc = None
    try:
        proc = subprocess.Popen(
            [PYTHON_BIN, "-m", "uvicorn", "worker.app:app", "--host", "127.0.0.1", "--port", str(port)],
            env=env
        )

        w = Worker(model, port, proc)
        with workers_lock:
            workers.setdefault(model, []).append(w)

        wait_healthy(w)
        return w

    except Exception as e:
        try:
            if proc is not None:
                proc.terminate()
        except Exception:
            pass

        with workers_lock:
            if model in workers:
                workers[model] = [x for x in workers[model] if x.port != port]
                if not workers[model]:
                    workers.pop(model, None)
            port_alloc.discard(port)

        raise HTTPException(503, f"Failed to spawn worker for {model}: {e}")


def pick_worker(model: str) -> Worker:
    """
    model：必须是 canonical name（外部调用前先 canonical_model_name）
    """
    with workers_lock:
        arr = list(workers.get(model, []))
    if arr:
        arr = sorted(arr, key=lambda x: (x.inflight, -x.last_used))
        return arr[0]

    lock = model_locks[model]
    with lock:
        with workers_lock:
            arr = list(workers.get(model, []))
        if arr:
            arr = sorted(arr, key=lambda x: (x.inflight, -x.last_used))
            return arr[0]

        with workers_lock:
            ev = start_events.get(model)
            if ev is None:
                ev = threading.Event()
                start_events[model] = ev
                is_initiator = True
            else:
                is_initiator = False

        if is_initiator:
            try:
                w = spawn_worker(model)
                return w
            finally:
                with workers_lock:
                    ev2 = start_events.pop(model, None)
                try:
                    if ev2 is not None:
                        ev2.set()
                except Exception:
                    pass
        else:
            ev.wait(timeout=90)
            with workers_lock:
                arr = list(workers.get(model, []))
            if not arr:
                raise HTTPException(503, f"{model} is warming up but failed to be ready; please retry")
            arr = sorted(arr, key=lambda x: (x.inflight, -x.last_used))
            return arr[0]


def evict_idle():
    idle: List[Worker] = []
    now = time.time()

    with workers_lock:
        snapshot = [(m, list(arr)) for m, arr in workers.items()]

    for _, arr in snapshot:
        for w in arr:
            if w.inflight == 0 and (now - w.last_used) > IDLE_TIMEOUT_S:
                idle.append(w)

    if not idle:
        with workers_lock:
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
        with workers_lock:
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


def evict_all(force_kill: bool = False):
    with workers_lock:
        all_workers: List[Worker] = [w for arr in workers.values() for w in arr]

        for ev in list(start_events.values()):
            try:
                ev.set()
            except Exception:
                pass
        start_events.clear()

    for w in all_workers:
        try:
            requests.post(f"http://127.0.0.1:{w.port}/shutdown", timeout=0.5)
        except Exception:
            pass

    for w in all_workers:
        try:
            if w.proc and w.proc.poll() is None:
                w.proc.terminate()
        except Exception:
            pass

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
        for w in all_workers:
            try:
                if w.proc and w.proc.poll() is None:
                    w.proc.wait(timeout=0.5)
            except Exception:
                pass

    with workers_lock:
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


# -------- API --------
class TextInput(BaseModel):
    input: list
    normalize: bool | None = True
    pooling: str | None = None
    fp16: bool | None = True
    batch: int | None = 32


@app.post("/embed")
def embed(model: str, body: dict = Body(...)):
    raw_model = model
    model = canonical_model_name(model)  # (1) canonicalize

    if model not in MODEL_SPECS:
        raise HTTPException(404, f"Unknown model {model}")

    def _request_with_worker(w: Worker):
        r = requests.post(f"http://127.0.0.1:{w.port}/embed", json=body, timeout=120)
        w.last_used = time.time()
        return r

    # 取/起 worker（使用 canonical name）
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
                with workers_lock:
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

            # 重新拉起一个再打一次
            w2 = spawn_worker(model)
            w2.inflight += 1
            try:
                r = _request_with_worker(w2)
            finally:
                w2.inflight -= 1

        if r.status_code >= 400:
            # (2) 区分 raw_model / canonical model，方便定位
            raise HTTPException(r.status_code, f"worker@{w.port} ({raw_model} -> {model}) -> {r.text}")
        return r.json()
    finally:
        w.inflight -= 1


@app.get("/metrics")
def metrics():
    with workers_lock:
        total = sum(len(v) for v in workers.values())
        flat = [
            # (3) metrics 输出 canonical model，调试稳定；请求侧如需 raw_model，可从日志/HTTPException里看
            {"model": w.model, "port": w.port, "inflight": w.inflight, "last_used": w.last_used}
            for arr in workers.values() for w in arr
        ]
    return {"total": total, "workers": flat}


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


@app.post("/evict_all")
def evict_all_now(force_kill: bool = False):
    evict_all(force_kill=force_kill)
    return {"ok": True, "force_kill": force_kill}
