# worker/app.py
import os
from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

app = FastAPI()
MODEL_NAME = os.getenv("MODEL_NAME", "bge-m3").strip()
MODEL_ROOT = os.getenv("MODEL_ROOT", "/root/models").strip()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None  # lazy

# ---------- Schemas ----------
class TextBatch(BaseModel):
    input: List[str]
    batch: Optional[int] = 32
    max_length: Optional[int] = 8192
    normalize: Optional[bool] = True  # -> BGEM3FlagModel.encode(normalize_embeddings=...)

class VLItem(BaseModel):
    image_url: Optional[str] = None
    image_path: Optional[str] = None
    text: Optional[str] = None

class VLMixedBatch(BaseModel):
    input: List[VLItem]

# ---------- Helpers ----------
def _is_http_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def _load_image_from_ref(ref: str):
    from PIL import Image
    if _is_http_url(ref):
        import requests as rq
        try:
            resp = rq.get(ref, stream=True, timeout=20)
            resp.raise_for_status()
            return Image.open(resp.raw).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"fetch/open image url failed: {ref}; {e}")
    else:
        if not os.path.exists(ref):
            raise HTTPException(400, f"image_path not found: {ref}")
        try:
            return Image.open(ref).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"open image path failed: {ref}; {e}")

# ---------- Lazy load ----------
def load_model():
    global _model
    if _model is not None:
        return

    name = MODEL_NAME.lower()

    # ---- BGE-M3 (text) ----
    if name in ["bge-m3", "baai/bge-m3"]:
        from FlagEmbedding import BGEM3FlagModel
        repo = os.path.join(MODEL_ROOT, "bge-m3") if name == "bge-m3" else MODEL_NAME
        _model = BGEM3FlagModel(repo, use_fp16=torch.cuda.is_available())
        return

    # ---- BGE-VL (vision-language) ----
    if name in ["bge-vl", "baai/bge-vl", "baai/bge-vl-base", "baai/bge-vl-large"]:
        from transformers import AutoModel
        model_path = os.path.join(MODEL_ROOT, "BGE-VL-base")
        mapping = {
            "bge-vl": model_path,
            "baai/bge-vl": model_path,
            "baai/bge-vl-base": model_path,
            "baai/bge-vl-large": model_path,
        }
        resolved = mapping.get(name, MODEL_NAME)
        dtype = torch.float16 if torch.cuda.is_available() else None
        _model = AutoModel.from_pretrained(resolved, trust_remote_code=True, torch_dtype=dtype)
        _model.set_processor(resolved)  # 官方要求
        _model.to(DEVICE)
        _model.eval()
        return

    # ---- Qwen3-Embedding-0.6B (sentence-transformers) ----
    if name in [
        "qwen", "qwen-embedding", "qwen3-embedding", "qwen/qwen3-embedding-0.6b", "qwen3-embedding-0.6b"
    ]:
        from sentence_transformers import SentenceTransformer
        # 默认走本地目录，离线更稳；需要联网加载就把 MODEL_NAME 设成 HF Repo 名
        local_path = os.path.join(MODEL_ROOT, "Qwen3-Embedding-0.6B")
        resolved = local_path if os.path.isdir(local_path) else MODEL_NAME
        # 直接放到当前设备；需要 flash-attn 等进阶可自行在构造里加 kwargs
        _model = SentenceTransformer(resolved, device=DEVICE)
        return

    raise RuntimeError(f"Unknown MODEL_NAME={MODEL_NAME}")

# ---------- Routes ----------
@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "device": DEVICE}

app.post("/shutdown")
def shutdown():
    import threading, time, os as _os
    def _exit():
        try:
            import torch
            global _model
            if _model is not None:
                try:
                    # 将权重移出 GPU（可选）
                    _model.to("cpu")
                except Exception:
                    pass
                # 删除引用
                _model = None
            # 清掉 PyTorch 缓存（还给 driver 需要进程退出）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        finally:
            time.sleep(0.2)
            _os._exit(0)   # 进程退出 → 驱动显存彻底释放
    threading.Thread(target=_exit, daemon=True).start()
    return {"ok": True}

@app.post("/embed")
def embed(body: Dict[str, Any] = Body(...)):
    load_model()
    name = MODEL_NAME.lower()

    # ----- bge-m3 -----
    if name in ["bge-m3", "baai/bge-m3"]:
        try:
            args = TextBatch(**body)
        except Exception as e:
            raise HTTPException(400, f"invalid body for bge-m3: {e}")

        with torch.inference_mode():
            # 不要传 normalize_embeddings
            res = _model.encode(
                args.input,
                batch_size=args.batch,
                max_length=args.max_length,
            )
            out = res["dense_vecs"]  # 官方返回 dict，要取 dense_vecs

            # 如需单位化，这里自己做
            if args.normalize:
                import torch.nn.functional as F
                if isinstance(out, torch.Tensor):
                    out = F.normalize(out, p=2, dim=-1)
                else:
                    out = torch.tensor(out)
                    out = F.normalize(out, p=2, dim=-1)

        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().tolist()
        dim = len(out[0]) if out else 0
        return {"embeddings": out, "dim": dim, "model": MODEL_NAME}
    # ----- Qwen3-Embedding-0.6B (text embedding) -----
    if name in [
        "qwen", "qwen-embedding", "qwen3-embedding", "qwen/qwen3-embedding-0.6b", "qwen3-embedding-0.6b"
    ]:
        # 不用 TextBatch 做校验，直接从 body 取，方便传 prompt_name
        items = body.get("input")
        if not isinstance(items, list) or not items:
            raise HTTPException(400, "invalid body for qwen: `input` must be a non-empty list[str]")

        # 可选：传查询提示名，官方示例是 "query"
        prompt_name = body.get("prompt_name", None)
        batch_size = int(body.get("batch", 32))
        normalize = bool(body.get("normalize", True))

        # SentenceTransformer.encode 支持 normalize_embeddings / batch_size
        with torch.inference_mode():
            vecs = _model.encode(
                items,
                prompt_name=prompt_name,           # None 时不生效；传 "query" 走查询提示
                batch_size=batch_size,
                convert_to_numpy=True,             # 直接拿到 numpy，省一次拷贝
                normalize_embeddings=normalize,    # 返回前归一化（余弦相似度友好）
                show_progress_bar=False,
            )

        out = vecs.tolist()
        dim = len(out[0]) if out else 0
        return {"embeddings": out, "dim": dim, "model": MODEL_NAME}
    # ----- bge-vl -----
    try:
        batch = VLMixedBatch(**body)
    except Exception as e:
        raise HTTPException(400, f"invalid body for bge-vl: {e}")

    if not batch.input:
        raise HTTPException(400, "input must be non-empty list")

    results: List[List[float]] = []
    with torch.no_grad():
        for i, item in enumerate(batch.input):
            img_ref = item.image_url or item.image_path
            txt = item.text
            if not img_ref and txt is None:
                raise HTTPException(400, f"item[{i}] must provide image_url/image_path and/or text")

            try:
                if img_ref and (txt is not None):
                    img = _load_image_from_ref(img_ref)
                    vec = _model.encode(images=img, text=txt)
                elif img_ref:
                    img = _load_image_from_ref(img_ref)
                    vec = _model.encode(images=img)
                else:
                    vec = _model.encode(text=txt)
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(500, f"encode failed on item[{i}]: {e}")

            if isinstance(vec, torch.Tensor):
                vec = vec.detach().to("cpu")
                if vec.ndim == 2 and vec.shape[0] == 1:
                    vec = vec.squeeze(0)
                vec_list = vec.tolist()
            elif isinstance(vec, (list, tuple)):
                vec_list = list(vec)
            else:
                raise HTTPException(500, f"unexpected embedding type on item[{i}]: {type(vec)}")

            results.append(vec_list)

    dim = len(results[0]) if results else 0
    return {"embeddings": results, "dim": dim, "model": MODEL_NAME}