# worker/app.py
import os
import threading
from typing import List, Optional, Dict, Any

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel

import re
import base64
import tempfile
import io

import requests
from PIL import Image

app = FastAPI()

# -------- 环境 --------
MODEL_NAME = os.getenv("MODEL_NAME", "bge-m3").strip()
MODEL_ROOT = os.getenv("MODEL_ROOT", "/root/models").strip()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 可选：显式指定 open-clip 本地目录（最稳）
# 例如：OPENCLIP_MODEL_DIR=/home/jhu/dev/models/embs/ViT-B-16-SigLIP2
OPENCLIP_MODEL_DIR = os.getenv("OPENCLIP_MODEL_DIR", "").strip()

# 进程内模型缓存（懒加载）+ 启动锁（避免并发冷启动竞态）
_model = None
_processor = None  # 给 transformers SigLip2 / 其他需要 processor 的模型用

# open-clip 相关
_openclip_preprocess = None
_openclip_tokenizer = None

_start_lock = threading.Lock()


# -------- Schemas --------
class TextBatch(BaseModel):
    input: List[str]
    batch: Optional[int] = 32
    max_length: Optional[int] = 8192
    normalize: Optional[bool] = True
    prompt_name: Optional[str] = None  # for qwen3


class VLItem(BaseModel):
    image_url: Optional[str] = None        # http(s):// 或 data:<mime>;base64,...
    image_path: Optional[str] = None       # 容器内可见的本地路径
    image_base64: Optional[str] = None     # data:<mime>;base64,...
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


# ---- base64 data URL 处理 ----
_DATA_URL_RE = re.compile(r"^data:(?P<mime>[-\w.+/]+);base64,(?P<b64>.+)$", re.IGNORECASE)

_MIME_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}


def _is_data_url(s: Optional[str]) -> bool:
    if not isinstance(s, str):
        return False
    return s.startswith("data:") and ";base64," in s


def _save_data_url_to_tmp(data_url: str) -> str:
    """
    将 data:<mime>;base64,<...> 写入临时文件，返回文件路径。
    """
    m = _DATA_URL_RE.match(data_url.strip())
    if not m:
        raise HTTPException(400, "invalid data URL format for image (expect data:<mime>;base64,...)")

    mime = m.group("mime").lower()
    b64 = m.group("b64")

    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise HTTPException(400, f"invalid base64 image data: {e}")

    # 简单的大小保护（可按需调整）
    if len(raw) > 64 * 1024 * 1024:
        raise HTTPException(413, "image too large (>64MB)")

    suffix = _MIME_SUFFIX.get(mime, ".img")
    fd, path = tempfile.mkstemp(prefix="bgevl_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
    except Exception as e:
        try:
            os.unlink(path)
        except Exception:
            pass
        raise HTTPException(500, f"write temp image failed: {e}")

    return path


def _load_pil_image_from_item(item: VLItem, tmp_files: List[str]) -> Image.Image:
    """
    给 SigLip2 / BGE-VL / OpenCLIP 用，把 VLItem 转成 PIL.Image
    - 支持 image_base64(data URL)、image_url(http/https 或本地路径)、image_path
    """
    # data URL 优先
    if item.image_base64 and _is_data_url(item.image_base64):
        p = _save_data_url_to_tmp(item.image_base64)
        tmp_files.append(p)
        img = Image.open(p).convert("RGB")
        return img

    if item.image_url:
        url = item.image_url
        if url.startswith("http://") or url.startswith("https://"):
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
            except Exception as e:
                raise HTTPException(400, f"failed to fetch image_url[{url}]: {e}")
            try:
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                raise HTTPException(400, f"failed to decode image from url[{url}]: {e}")
            return img
        else:
            # 当作本地路径
            try:
                img = Image.open(url).convert("RGB")
            except Exception as e:
                raise HTTPException(400, f"failed to open local image_url path[{url}]: {e}")
            return img

    if item.image_path:
        try:
            img = Image.open(item.image_path).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"failed to open image_path[{item.image_path}]: {e}")
        return img

    raise HTTPException(400, "VLItem must provide image_url/image_path/image_base64")


# -------- Lazy load --------
def load_model():
    """
    懒加载指定模型：
    - bge-m3: FlagEmbedding BGEM3FlagModel（禁用内置 fp16，避免 .half 报错）
    - bge-vl: transformers AutoModel (trust_remote_code)
    - qwen3-embedding: sentence-transformers SentenceTransformer（构造期不 to）
    - siglip2-*: google/siglip2-... AutoModel + AutoProcessor
    - openclip-vit-b-16-siglip2: open-clip-torch create_model_from_pretrained + get_tokenizer
    """
    global _model, _processor, _openclip_preprocess, _openclip_tokenizer
    if _model is not None:
        return

    with _start_lock:
        if _model is not None:
            return

        name = MODEL_NAME.lower()

        # ---- BGE-M3 (text) ----
        if name in ["bge-m3", "baai/bge-m3"]:
            from FlagEmbedding import BGEM3FlagModel
            local_path = os.path.join(MODEL_ROOT, "bge-m3")
            resolved = local_path if os.path.isdir(local_path) else MODEL_NAME
            _model = BGEM3FlagModel(resolved, use_fp16=False)
            try:
                if torch.cuda.is_available() and hasattr(_model, "model"):
                    _model.model.to("cuda")
            except Exception:
                pass
            return

        # ---- BGE-VL (vision-language) ----
        if name in ["bge-vl", "baai/bge-vl", "baai/bge-vl-base", "baai/bge-vl-large"]:
            from transformers import AutoModel

            if torch.cuda.is_available():
                try:
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
                except Exception:
                    pass

            base_path = os.path.join(MODEL_ROOT, "BGE-VL-base")
            large_path = os.path.join(MODEL_ROOT, "BGE-VL-large")
            mapping = {
                "bge-vl": base_path,
                "baai/bge-vl": base_path,
                "baai/bge-vl-base": base_path,
                "baai/bge-vl-large": large_path,
            }
            resolved = mapping.get(name, MODEL_NAME)

            target_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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
            _model = SentenceTransformer(resolved)
            return

        # ---- OpenCLIP SigLIP2 (ViT-B-16-SigLIP2) ----
        # 推荐 MODEL_NAME：openclip-vit-b-16-siglip2
        if name in ["openclip-vit-b-16-siglip2", "openclip-siglip2-vit-b-16", "vit-b-16-siglip2"]:
            try:
                from open_clip import create_model_from_pretrained, get_tokenizer
            except Exception as e:
                raise RuntimeError(f"open_clip import failed (need open-clip-torch>=2.31.0): {e}")

            # 目录优先级：
            # 1) OPENCLIP_MODEL_DIR
            # 2) MODEL_ROOT/ViT-B-16-SigLIP2
            # 3) MODEL_ROOT/<MODEL_NAME>
            candidates = []
            if OPENCLIP_MODEL_DIR:
                candidates.append(OPENCLIP_MODEL_DIR)
            candidates.append(os.path.join(MODEL_ROOT, "ViT-B-16-SigLIP2"))
            candidates.append(os.path.join(MODEL_ROOT, MODEL_NAME))

            model_dir = None
            for c in candidates:
                if c and os.path.isdir(c):
                    model_dir = c
                    break
            if model_dir is None:
                raise RuntimeError(f"OpenCLIP model dir not found. Tried: {candidates}")

            model_path = f"local-dir:{model_dir}"

            _model, _openclip_preprocess = create_model_from_pretrained(model_path)
            _openclip_tokenizer = get_tokenizer(model_path)

            _model.to(DEVICE)
            _model.eval()
            return

        # ---- SigLip2 (google/siglip2-*) via transformers ----
        if name.startswith("siglip2-"):
            from transformers import AutoModel, AutoProcessor

            local_dir = os.path.join(MODEL_ROOT, MODEL_NAME)
            if os.path.isdir(local_dir):
                model_id = local_dir
                processor_id = local_dir
            else:
                model_id = f"google/{MODEL_NAME}"
                processor_id = model_id

            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    target_dtype = torch.bfloat16
                else:
                    target_dtype = torch.float16
            else:
                target_dtype = torch.float32

            _model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=target_dtype,
                device_map=None,
            )
            _model.to(DEVICE, dtype=target_dtype)
            _model.eval()

            _processor = AutoProcessor.from_pretrained(processor_id)
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
        global _model, _processor, _openclip_preprocess, _openclip_tokenizer
        if _model is not None:
            try:
                _model.to("cpu")
            except Exception:
                pass
            _model = None
        _processor = None
        _openclip_preprocess = None
        _openclip_tokenizer = None
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
    - bge-vl: VLMixedBatch -> _model.encode(images=..., text=...)
    - qwen3: TextBatch -> SentenceTransformer.encode(...)
    - openclip-vit-b-16-siglip2:
        * input 为 list[str] -> encode_text
        * input 为 list[VLItem] -> encode_image（PIL->preprocess->tensor）
    - siglip2-* (transformers):
        * input 为 list[str] -> get_text_features
        * input 为 list[VLItem] -> get_image_features
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
                "device": device_arg,
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

    # ----- OpenCLIP SigLIP2 (ViT-B-16) -----
    if name in ["openclip-vit-b-16-siglip2", "openclip-siglip2-vit-b-16", "vit-b-16-siglip2"]:
        if _openclip_preprocess is None or _openclip_tokenizer is None:
            raise HTTPException(500, "OpenCLIP preprocess/tokenizer not initialized")

        inputs = body.get("input", None)
        if not isinstance(inputs, list) or not inputs:
            raise HTTPException(400, "input must be a non-empty list")

        normalize = bool(body.get("normalize", True))

        # 文本模式：list[str]
        if isinstance(inputs[0], str):
            texts: List[str] = inputs
            try:
                text_tokens = _openclip_tokenizer(
                    texts,
                    context_length=getattr(_model, "context_length", None) or 77
                ).to(DEVICE)
            except Exception as e:
                raise HTTPException(400, f"OpenCLIP tokenize failed: {e}")

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                vecs = _model.encode_text(text_tokens, normalize=normalize)

            vecs = _to_list(vecs)
            dim = len(vecs[0]) if vecs else 0
            return {"embeddings": vecs, "dim": dim, "model": MODEL_NAME, "modality": "text"}

        # 图片模式：list[VLItem]
        tmp_files: List[str] = []
        try:
            batch = VLMixedBatch(**body)
        except Exception as e:
            raise HTTPException(400, f"invalid body for openclip image mode: {e}")

        if not batch.input:
            raise HTTPException(400, "input must be non-empty list")

        imgs: List[torch.Tensor] = []
        try:
            for item in batch.input:
                pil = _load_pil_image_from_item(item, tmp_files)
                t = _openclip_preprocess(pil).unsqueeze(0)
                imgs.append(t)

            image_tensor = torch.cat(imgs, dim=0).to(DEVICE)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                vecs = _model.encode_image(image_tensor, normalize=normalize)

            vecs = _to_list(vecs)
            dim = len(vecs[0]) if vecs else 0
            return {"embeddings": vecs, "dim": dim, "model": MODEL_NAME, "modality": "image"}

        finally:
            for p in tmp_files:
                try:
                    if p and os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

    # ----- SigLip2 (transformers) -----
    if name.startswith("siglip2-"):
        if _processor is None:
            raise HTTPException(500, "SigLip2 processor not initialized")

        inputs = body.get("input", None)
        if not isinstance(inputs, list) or len(inputs) == 0:
            raise HTTPException(400, "input must be a non-empty list")

        normalize = bool(body.get("normalize", True))

        # 文本模式：list[str]
        if isinstance(inputs[0], str):
            texts: List[str] = inputs
            try:
                enc = _processor(
                    text=texts,
                    padding="max_length",
                    max_length=64,
                    return_tensors="pt",
                )
            except Exception as e:
                raise HTTPException(400, f"SigLip2 text processing failed: {e}")

            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            with torch.no_grad():
                vecs = _model.get_text_features(**enc)  # (B, D)
                if normalize:
                    import torch.nn.functional as F
                    vecs = F.normalize(vecs, p=2, dim=-1)

            vecs = _to_list(vecs)
            dim = len(vecs[0]) if vecs else 0
            return {"embeddings": vecs, "dim": dim, "model": MODEL_NAME, "modality": "text"}

        # 图片模式：list[VLItem-like]
        tmp_files: List[str] = []
        try:
            batch = VLMixedBatch(**body)
        except Exception as e:
            raise HTTPException(400, f"invalid body for siglip2 image mode: {e}")

        if not batch.input:
            raise HTTPException(400, "input must be non-empty list")

        images: List[Image.Image] = []
        try:
            for item in batch.input:
                img = _load_pil_image_from_item(item, tmp_files)
                images.append(img)

            try:
                enc = _processor(images=images, return_tensors="pt")
            except Exception as e:
                raise HTTPException(400, f"SigLip2 image processing failed: {e}")

            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            with torch.no_grad():
                vecs = _model.get_image_features(**enc)  # (B, D)
                if normalize:
                    import torch.nn.functional as F
                    vecs = F.normalize(vecs, p=2, dim=-1)

            vecs = _to_list(vecs)
            dim = len(vecs[0]) if vecs else 0
            return {"embeddings": vecs, "dim": dim, "model": MODEL_NAME, "modality": "image"}

        finally:
            for p in tmp_files:
                try:
                    if p and os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

    # ----- bge-vl (默认走这里) -----
    try:
        batch = VLMixedBatch(**body)
    except Exception as e:
        raise HTTPException(400, f"invalid body for bge-vl: {e}")

    if not batch.input:
        raise HTTPException(400, "input must be non-empty list")

    results: List[List[float]] = []
    tmp_files: List[str] = []

    with torch.inference_mode():
        try:
            for i, item in enumerate(batch.input):
                img_ref: Optional[str] = None
                if item.image_base64 and _is_data_url(item.image_base64):
                    img_ref = _save_data_url_to_tmp(item.image_base64)
                    tmp_files.append(img_ref)
                elif item.image_url and _is_data_url(item.image_url):
                    img_ref = _save_data_url_to_tmp(item.image_url)
                    tmp_files.append(img_ref)
                else:
                    img_ref = item.image_url or item.image_path

                txt = item.text
                if not img_ref and txt is None:
                    raise HTTPException(400, f"item[{i}] must provide image_url/image_path/image_base64 and/or text")

                images_arg = [img_ref] if img_ref is not None else None
                text_arg = [txt] if txt is not None else None

                def _encode_once():
                    if images_arg is not None and text_arg is not None:
                        return _model.encode(images=images_arg, text=text_arg)
                    elif images_arg is not None:
                        return _model.encode(images=images_arg)
                    else:
                        return _model.encode(text=text_arg)

                try:
                    vec = _encode_once()
                except Exception as e:
                    msg = str(e)
                    if (
                        "Expected all tensors to be on the same device" in msg
                        and "cuda" in msg and "cpu" in msg
                    ):
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            vec = _encode_once()
                        except Exception as e2:
                            import traceback
                            raise HTTPException(500, f"bge-vl encode failed on item[{i}] after same-device retry: {e2}\n{traceback.format_exc()}")
                    else:
                        import traceback
                        raise HTTPException(500, f"bge-vl encode failed on item[{i}]: {e}\n{traceback.format_exc()}")

                vec_list = _to_list(vec)
                if isinstance(vec_list, list) and len(vec_list) == 1 and isinstance(vec_list[0], list):
                    vec_list = vec_list[0]
                results.append(vec_list)
        finally:
            for p in tmp_files:
                try:
                    if p and os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

    dim = len(results[0]) if results else 0
    return {"embeddings": results, "dim": dim, "model": MODEL_NAME}