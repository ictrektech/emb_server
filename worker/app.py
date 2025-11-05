# worker/app.py
import os
import threading
from typing import List, Optional, Dict, Any

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

app = FastAPI()

# -------- 环境 --------
MODEL_NAME = os.getenv("MODEL_NAME", "bge-m3").strip()
MODEL_ROOT = os.getenv("MODEL_ROOT", "/root/models").strip()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 进程内模型缓存（懒加载）+ 启动锁（避免并发冷启动竞态）
_model = None
_start_lock = threading.Lock()


# -------- Schemas --------
class TextBatch(BaseModel):
    input: List[str]
    batch: Optional[int] = 32
    max_length: Optional[int] = 8192
    normalize: Optional[bool] = True
    prompt_name: Optional[str] = None  # for qwen3


class VLItem(BaseModel):
    image_url: Optional[str] = None
    image_path: Optional[str] = None
    text: Optional[str] = None


class VLMixedBatch(BaseModel):
    input: List[VLItem]


# -------- Utils --------
def _to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# -------- Lazy load --------
def load_model():
    """
    懒加载指定模型：
    - bge-m3: FlagEmbedding BGEM3FlagModel（禁用内置 fp16，避免 .half 报错）
    - bge-vl: transformers AutoModel (trust_remote_code)
    - qwen3-embedding: sentence-transformers SentenceTransformer（构造期不 to）
    """
    global _model
    if _model is not None:
        return

    with _start_lock:  # 同一进程内串行初始化，避免并发导入/构造竞态
        if _model is not None:
            return

        name = MODEL_NAME.lower()

        # ---- BGE-M3 (text) ----
        if name in ["bge-m3", "baai/bge-m3"]:
            from FlagEmbedding import BGEM3FlagModel
            local_path = os.path.join(MODEL_ROOT, "bge-m3")
            resolved = local_path if os.path.isdir(local_path) else MODEL_NAME
            # 关键：禁用 use_fp16，规避库里 .half() 路径的 dtype 报错
            _model = BGEM3FlagModel(resolved, use_fp16=False)
            # 尝试把内部 encoder 放到 CUDA（若库不自动放置）
            try:
                if torch.cuda.is_available() and hasattr(_model, "model"):
                    _model.model.to("cuda")
            except Exception:
                pass
            return

        # ---- BGE-VL (vision-language) ----
        if name in ["bge-vl", "baai/bge-vl", "baai/bge-vl-base", "baai/bge-vl-large"]:
            # 并发 import 也放到锁里
            from transformers import AutoModel

            # 禁用 fused/mem-efficient SDPA，避免 dtype 不一致触发严格校验
            if torch.cuda.is_available():
                try:
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
                except Exception:
                    pass

            # 解析本地路径；base/large 自行映射
            base_path = os.path.join(MODEL_ROOT, "BGE-VL-base")
            large_path = os.path.join(MODEL_ROOT, "BGE-VL-large")
            mapping = {
                "bge-vl": base_path,
                "baai/bge-vl": base_path,
                "baai/bge-vl-base": base_path,
                "baai/bge-vl-large": large_path,
            }
            resolved = mapping.get(name, MODEL_NAME)

            # 统一 dtype（CUDA=fp16，CPU=fp32）
            target_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # 新版 transformers 推荐 dtype=...
            _model = AutoModel.from_pretrained(
                resolved,
                trust_remote_code=True,
                dtype=target_dtype,
            )
            _model.set_processor(resolved)
            _model.to(device=DEVICE, dtype=target_dtype)
            _model.eval()
            return

        # ---- Qwen3-Embedding-0.6B (sentence-transformers) ----
        if name in [
            "qwen", "qwen-embedding", "qwen3-embedding",
            "qwen/qwen3-embedding-0.6b", "qwen3-embedding-0.6b"
        ]:
            from sentence_transformers import SentenceTransformer
            local_path = os.path.join(MODEL_ROOT, "Qwen3-Embedding-0.6B")
            resolved = local_path if os.path.isdir(local_path) else MODEL_NAME

            # 构造期不传 device（避免 meta tensor 迁移报错）
            _model = SentenceTransformer(
                resolved,
                # 可选： model_kwargs={"attn_implementation": "flash_attention_2"},
                # 可选： tokenizer_kwargs={"padding_side": "left"},
            )
            return

        raise RuntimeError(f"Unknown MODEL_NAME={MODEL_NAME}")


# -------- Routes --------
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "device": DEVICE}


@app.post("/shutdown")
def shutdown():
    """
    优雅退出：
    - 将模型移到 CPU（尽量）
    - 清空 CUDA 缓存（需要进程退出才彻底释放）
    - 进程退出
    """
    import time as _time
    import os as _os

    try:
        global _model
        if _model is not None:
            try:
                _model.to("cpu")
            except Exception:
                pass
            _model = None
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
    finally:
        _time.sleep(0.2)
        _os._exit(0)


@app.post("/embed")
def embed(body: Dict[str, Any] = Body(...)):
    """
    - bge-m3: TextBatch -> BGEM3FlagModel.encode(...), 取 ['dense_vecs']，按需归一化
    - bge-vl: VLMixedBatch -> _model.encode(images=..., text=...)；传入字符串（URL/路径）
    - qwen3: TextBatch -> SentenceTransformer.encode(..., device=...)
    """
    load_model()
    name = MODEL_NAME.lower()

    # ----- bge-m3 -----
    if name in ["bge-m3", "baai/bge-m3"]:
        try:
            args = TextBatch(**body)
        except Exception as e:
            raise HTTPException(400, f"invalid body for bge-m3: {e}")

        try:
            with torch.inference_mode():
                res = _model.encode(
                    args.input,
                    batch_size=args.batch,
                    max_length=args.max_length,
                )
                out = res["dense_vecs"]
                # 归一化（可选）
                if args.normalize:
                    import torch.nn.functional as F
                    if isinstance(out, torch.Tensor):
                        out = F.normalize(out, p=2, dim=-1)
                    else:
                        out = torch.tensor(out)
                        out = F.normalize(out, p=2, dim=-1)
        except Exception as e:
            import traceback
            raise HTTPException(500, f"bge-m3 encode failed: {e}\n{traceback.format_exc()}")

        out = _to_list(out)
        dim = len(out[0]) if out else 0
        return {"embeddings": out, "dim": dim, "model": MODEL_NAME}

    # ----- qwen3-embedding -----
    if name in [
        "qwen", "qwen-embedding", "qwen3-embedding",
        "qwen/qwen3-embedding-0.6b", "qwen3-embedding-0.6b"
    ]:
        try:
            items = body.get("input", None)
            if not isinstance(items, list) or not items:
                raise ValueError("input must be a non-empty list[str]")
            batch_size = int(body.get("batch", 32))
            normalize = bool(body.get("normalize", True))
            prompt_name = body.get("prompt_name", None)
        except Exception as e:
            raise HTTPException(400, f"invalid body for qwen3-embedding: {e}")

        device_arg = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            kwargs = {
                "batch_size": batch_size,
                "convert_to_numpy": True,
                "normalize_embeddings": normalize,
                "show_progress_bar": False,
                "device": device_arg,  # 运行期指定设备，规避 meta tensor 迁移问题
            }
            if prompt_name:
                kwargs["prompt_name"] = prompt_name

            vecs = _model.encode(items, **kwargs)
        except Exception as e:
            import traceback
            raise HTTPException(500, f"qwen3 encode failed: {e}\n{traceback.format_exc()}")

        vecs = _to_list(vecs)
        dim = len(vecs[0]) if len(vecs) > 0 else 0
        return {"embeddings": vecs, "dim": dim, "model": MODEL_NAME}

    # ----- bge-vl -----
    try:
        batch = VLMixedBatch(**body)
    except Exception as e:
        raise HTTPException(400, f"invalid body for bge-vl: {e}")

    if not batch.input:
        raise HTTPException(400, "input must be non-empty list")

    results: List[List[float]] = []
    with torch.inference_mode():
        for i, item in enumerate(batch.input):
            img_ref = item.image_url or item.image_path
            txt = item.text
            if not img_ref and txt is None:
                raise HTTPException(400, f"item[{i}] must provide image_url/image_path and/or text")

            try:
                # 保持 images 与 text 的“标量类型 str”一致（单样本）
                if img_ref is not None and txt is not None:
                    vec = _model.encode(images=img_ref, text=txt)
                elif img_ref is not None:
                    vec = _model.encode(images=img_ref)
                else:
                    vec = _model.encode(text=txt)
            except HTTPException:
                raise
            except Exception as e:
                import traceback
                raise HTTPException(500, f"bge-vl encode failed on item[{i}]: {e}\n{traceback.format_exc()}")

            vec_list = _to_list(vec)
            # 统一处理 [1, D] -> [D]
            if isinstance(vec_list, list) and len(vec_list) == 1 and isinstance(vec_list[0], list):
                vec_list = vec_list[0]
            results.append(vec_list)

    dim = len(results[0]) if results else 0
    return {"embeddings": results, "dim": dim, "model": MODEL_NAME}