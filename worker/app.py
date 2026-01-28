# worker/app.py
# -*- coding: utf-8 -*-
"""
Embedding worker (one model per process).

Newly added:
- Immich ONNX SigLIP2 dual-encoder: immich-app/ViT-B-16-SigLIP2__webli
  * text:  textual/model.onnx  input name: "text"  int32 [B, context_length]
  * image: visual/model.onnx   input name: "image" float32 [B,3,H,W]
  * tokenizer: textual/tokenizer.json (HuggingFace tokenizers format)
  * preprocess: visual/preprocess_cfg.json

Key fixes for your errors:
- DO NOT enable TensorRT EP by default (it throws if TensorRT libs are not installed).
- Prefer CUDAExecutionProvider when available; otherwise CPU.

Env knobs:
- MODEL_NAME: which model to load (set by manager)
- MODEL_ROOT: root folder for local models (default /root/models)
- IMMICH_MODEL_DIR: if set and points to a directory that contains config.json/textual/visual, it will be used
- IMMICH_HF_REPO: repo id for snapshot_download if local dir not present
- HF_TOKEN: optional HF token for private repos
"""

import os
import io
import re
import json
import base64
import tempfile
import threading
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from PIL import Image

import requests

app = FastAPI()

# -------- Env --------
MODEL_NAME = os.getenv("MODEL_NAME", "bge-m3").strip()
MODEL_ROOT = os.getenv("MODEL_ROOT", "/root/models").strip()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# OpenCLIP local dir (optional)
OPENCLIP_MODEL_DIR = os.getenv("OPENCLIP_MODEL_DIR", "").strip()

# Immich ONNX (optional overrides)
IMMICH_HF_REPO = os.getenv("IMMICH_HF_REPO", "immich-app/ViT-B-16-SigLIP2__webli").strip()
IMMICH_MODEL_DIR = os.getenv("IMMICH_MODEL_DIR", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# -------- In-process caches (lazy) --------
_model = None
_processor = None  # transformers processor (siglip2)
_openclip_preprocess = None
_openclip_tokenizer = None

# Immich ONNX caches
_immich_text_sess = None
_immich_visual_sess = None
_immich_tokenizer = None
_immich_preprocess = None  # dict: size/mean/std/resampling
_immich_context_len = None

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
    image_base64: Optional[str] = None  # data:<mime>;base64,...
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


def _l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a numpy array on the last dimension."""
    x = x.astype(np.float32, copy=False)
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom

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
    return isinstance(s, str) and s.startswith("data:") and ";base64," in s


def _save_data_url_to_tmp(data_url: str) -> str:
    m = _DATA_URL_RE.match(data_url.strip())
    if not m:
        raise HTTPException(400, "invalid data URL (expect data:<mime>;base64,...)")
    mime = m.group("mime").lower()
    b64 = m.group("b64")

    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise HTTPException(400, f"invalid base64 image data: {e}")

    if len(raw) > 64 * 1024 * 1024:
        raise HTTPException(413, "image too large (>64MB)")

    suffix = _MIME_SUFFIX.get(mime, ".img")
    fd, path = tempfile.mkstemp(prefix="img_", suffix=suffix)
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
    # data URL first
    if item.image_base64 and _is_data_url(item.image_base64):
        p = _save_data_url_to_tmp(item.image_base64)
        tmp_files.append(p)
        return Image.open(p).convert("RGB")

    # image_url
    if item.image_url:
        url = item.image_url
        if _is_data_url(url):
            p = _save_data_url_to_tmp(url)
            tmp_files.append(p)
            return Image.open(p).convert("RGB")

        if url.startswith("http://") or url.startswith("https://"):
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
            except Exception as e:
                raise HTTPException(400, f"failed to fetch image_url[{url}]: {e}")
            try:
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                raise HTTPException(400, f"failed to decode image from url[{url}]: {e}")

        # local path
        try:
            return Image.open(url).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"failed to open local image_url path[{url}]: {e}")

    # image_path
    if item.image_path:
        try:
            return Image.open(item.image_path).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"failed to open image_path[{item.image_path}]: {e}")

    raise HTTPException(400, "VLItem must provide image_url/image_path/image_base64")


# -------- Immich ONNX helpers --------
def _is_immich_model(model_name: str) -> bool:
    n = (model_name or "").lower()
    return (
        "siglip2__webli" in n
        or n in ("immich-vit-b-16-siglip2__webli", "vit-b-16-siglip2__webli")
        or n.startswith("immich-app/")
    )


def _hf_or_local_immich_dir() -> str:
    """
    Your tree matches what we expect:
      config.json
      textual/{model.onnx, tokenizer.json, tokenizer_config.json, ...}
      visual/{model.onnx, preprocess_cfg.json, ...}
    """
    if IMMICH_MODEL_DIR and os.path.isdir(IMMICH_MODEL_DIR):
        return IMMICH_MODEL_DIR

    # Your suggested local name under MODEL_ROOT
    cand = os.path.join(MODEL_ROOT, "ViT-B-16-SigLIP2__webli")
    if os.path.isdir(cand):
        return cand

    # Or last segment of repo id
    cand2 = os.path.join(MODEL_ROOT, IMMICH_HF_REPO.split("/")[-1])
    if os.path.isdir(cand2):
        return cand2

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(f"huggingface_hub not available: {e}. Install: pip install -U huggingface_hub")

    local_cache = os.path.join(MODEL_ROOT, "hf_cache")
    os.makedirs(local_cache, exist_ok=True)

    return snapshot_download(
        repo_id=IMMICH_HF_REPO,
        cache_dir=local_cache,
        token=HF_TOKEN or None,
        allow_patterns=["*.json", "*.onnx", "*tokenizer*", "*preprocess*", "*"],
    )


def _pil_resampling_from_str(name: str):
    name = (name or "").lower()
    if name in ("nearest", "nearestneighbor", "nearest-neighbor"):
        return Image.Resampling.NEAREST
    if name in ("bilinear",):
        return Image.Resampling.BILINEAR
    if name in ("bicubic",):
        return Image.Resampling.BICUBIC
    if name in ("lanczos",):
        return Image.Resampling.LANCZOS
    return Image.Resampling.BICUBIC


def _immich_resize_center_crop(img: Image.Image, size: int, resample) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        raise HTTPException(400, "invalid image with zero dimension")

    scale = size / min(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), resample=resample)

    left = int(round((new_w - size) / 2.0))
    top = int(round((new_h - size) / 2.0))
    return img.crop((left, top, left + size, top + size))


def _immich_image_to_tensor(img: Image.Image, cfg: Dict[str, Any]) -> np.ndarray:
    img = _immich_resize_center_crop(img, size=int(cfg["size"]), resample=cfg["resampling"])
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC RGB
    arr = (arr - cfg["mean"]) / cfg["std"]
    arr = arr.transpose(2, 0, 1)  # CHW
    return arr.astype(np.float32)
def _immich_select_ort_providers() -> List[str]:
    import onnxruntime as ort
    avail = ort.get_available_providers()

    # Critical: DON'T add TensorrtExecutionProvider unless you *know* TRT libs are installed.
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _immich_load() -> None:
    global _immich_text_sess, _immich_visual_sess, _immich_tokenizer, _immich_preprocess, _immich_context_len
    if _immich_text_sess is not None and _immich_visual_sess is not None:
        return

    repo_dir = _hf_or_local_immich_dir()

    config_path = os.path.join(repo_dir, "config.json")
    visual_cfg_path = os.path.join(repo_dir, "visual", "preprocess_cfg.json")
    text_model_path = os.path.join(repo_dir, "textual", "model.onnx")
    visual_model_path = os.path.join(repo_dir, "visual", "model.onnx")
    tokenizer_path = os.path.join(repo_dir, "textual", "tokenizer.json")
    tokenizer_cfg_path = os.path.join(repo_dir, "textual", "tokenizer_config.json")

    for p in (config_path, visual_cfg_path, text_model_path, visual_model_path, tokenizer_path, tokenizer_cfg_path):
        if not os.path.exists(p):
            raise RuntimeError(f"Immich model file missing: {p}")

    with open(config_path, "r", encoding="utf-8") as f:
        model_cfg = json.load(f)
    text_cfg = model_cfg.get("text_cfg", {})
    _immich_context_len = int(text_cfg.get("context_length", 77))

    with open(visual_cfg_path, "r", encoding="utf-8") as f:
        preprocess_cfg = json.load(f)

    size = preprocess_cfg["size"]
    size = int(size[0] if isinstance(size, list) else size)

    mean = np.array(preprocess_cfg["mean"], dtype=np.float32)
    std = np.array(preprocess_cfg["std"], dtype=np.float32)
    interp = preprocess_cfg.get("interpolation", "bicubic")
    resampling = _pil_resampling_from_str(interp)

    _immich_preprocess = {
        "size": size,
        "mean": mean,
        "std": std,
        "resampling": resampling,
        "interpolation": interp,
        "repo_dir": repo_dir,
    }

    # tokenizer
    try:
        from tokenizers import Tokenizer
    except Exception as e:
        raise RuntimeError(f"tokenizers not available: {e}. Install: pip install -U tokenizers")

    tok = Tokenizer.from_file(tokenizer_path)

    with open(tokenizer_cfg_path, "r", encoding="utf-8") as f:
        tok_cfg = json.load(f)

    pad_token = tok_cfg.get("pad_token", "<pad>")
    pad_id = tok.token_to_id(pad_token)

    if pad_id is None:
        # fallback candidates
        for t in (pad_token, "<|endoftext|>", "<pad>", "</s>", "<unk>"):
            pid = tok.token_to_id(t)
            if pid is not None:
                pad_token = t
                pad_id = pid
                break
    if pad_id is None:
        raise RuntimeError("cannot determine pad_token/pad_id from tokenizer_config.json")

    tok.enable_padding(length=_immich_context_len, pad_token=pad_token, pad_id=int(pad_id))
    tok.enable_truncation(max_length=_immich_context_len)
    _immich_tokenizer = tok

    import onnxruntime as ort
    providers = _immich_select_ort_providers()

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.getenv("ORT_INTRA_OP_THREADS", "0") or 0)
    so.inter_op_num_threads = int(os.getenv("ORT_INTER_OP_THREADS", "0") or 0)

    _immich_text_sess = ort.InferenceSession(text_model_path, sess_options=so, providers=providers)
    _immich_visual_sess = ort.InferenceSession(visual_model_path, sess_options=so, providers=providers)


def _immich_encode_text(texts: List[str], normalize: bool) -> np.ndarray:
    _immich_load()
    assert _immich_text_sess is not None and _immich_tokenizer is not None

    enc = _immich_tokenizer.encode_batch([(t or "").strip() for t in texts])
    ids = np.array([e.ids for e in enc], dtype=np.int32)

    # Some exports are fixed-batch (=1). We try batched first; if ORT rejects dim0, fallback to per-item.
    def _run(ids_batch: np.ndarray) -> np.ndarray:
        out = _immich_text_sess.run(None, {"text": ids_batch})[0]
        return out.astype(np.float32)

    vecs: np.ndarray
    try:
        vecs = _run(ids)
    except Exception as e:
        msg = str(e)
        # Typical error: "Got invalid dimensions ... Got: 4 Expected: 1"
        if ("invalid dimensions" in msg.lower() and "expected: 1" in msg.lower()) or ("expected: 1" in msg):
            outs: List[np.ndarray] = []
            for i in range(ids.shape[0]):
                out_i = _run(ids[i:i+1])[0]
                outs.append(out_i)
            vecs = np.stack(outs, axis=0).astype(np.float32)
        else:
            raise

    if normalize:
        vecs = _l2_normalize_np(vecs)
    return vecs


def _immich_encode_image(images: List[Image.Image], normalize: bool) -> np.ndarray:
    _immich_load()
    assert _immich_visual_sess is not None and _immich_preprocess is not None

    xs = [_immich_image_to_tensor(img, _immich_preprocess) for img in images]
    x = np.stack(xs, axis=0).astype(np.float32)

    # Handle fixed batch dim (=1) exports
    try:
        inp0 = _immich_visual_sess.get_inputs()[0]
        b0 = inp0.shape[0] if inp0.shape and len(inp0.shape) > 0 else None
    except Exception:
        b0 = None

    if b0 == 1 and x.shape[0] != 1:
        outs: List[np.ndarray] = []
        for i in range(x.shape[0]):
            out = _immich_visual_sess.run(None, {"image": x[i:i+1]})[0]
            outs.append(out[0])
        vecs = np.stack(outs, axis=0).astype(np.float32)
    else:
        vecs = _immich_visual_sess.run(None, {"image": x})[0].astype(np.float32)

    if normalize:
        vecs = _l2_normalize_np(vecs)
    return vecs



def load_model():
    global _model, _processor, _openclip_preprocess, _openclip_tokenizer

    if _is_immich_model(MODEL_NAME):
        _immich_load()
        return

    if _model is not None:
        return

    with _start_lock:
        if _is_immich_model(MODEL_NAME):
            _immich_load()
            return
        if _model is not None:
            return

        name = MODEL_NAME.lower()

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
            _model = AutoModel.from_pretrained(resolved, trust_remote_code=True, dtype=target_dtype)
            _model.set_processor(resolved)
            _model.to(device=DEVICE, dtype=target_dtype)
            _model.eval()
            return

        if name in [
            "qwen", "qwen-embedding", "qwen3-embedding",
            "qwen/qwen3-embedding-0.6b", "qwen3-embedding-0.6b"
        ]:
            from sentence_transformers import SentenceTransformer
            local_path = os.path.join(MODEL_ROOT, "Qwen3-Embedding-0.6B")
            resolved = local_path if os.path.isdir(local_path) else MODEL_NAME
            _model = SentenceTransformer(resolved)
            return

        if name in ["openclip-vit-b-16-siglip2", "openclip-siglip2-vit-b-16", "vit-b-16-siglip2"]:
            try:
                from open_clip import create_model_from_pretrained, get_tokenizer
            except Exception as e:
                raise RuntimeError(f"open_clip import failed (need open-clip-torch>=2.31.0): {e}")

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
                target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                target_dtype = torch.float32

            _model = AutoModel.from_pretrained(model_id, torch_dtype=target_dtype, device_map=None)
            _model.to(DEVICE, dtype=target_dtype)
            _model.eval()
            _processor = AutoProcessor.from_pretrained(processor_id)
            return

        raise RuntimeError(f"Unknown MODEL_NAME={MODEL_NAME}")


# -------- Routes --------
@app.get("/health")
def health():
    if _is_immich_model(MODEL_NAME):
        try:
            import onnxruntime as ort
            return {
                "ok": True,
                "model": MODEL_NAME,
                "device": DEVICE,
                "onnxruntime_available_providers": ort.get_available_providers(),
            }
        except Exception:
            return {"ok": True, "model": MODEL_NAME, "device": DEVICE}
    return {"ok": True, "model": MODEL_NAME, "device": DEVICE}


@app.post("/shutdown")
def shutdown():
    import time as _time
    import os as _os

    global _model, _processor, _openclip_preprocess, _openclip_tokenizer
    global _immich_text_sess, _immich_visual_sess, _immich_tokenizer, _immich_preprocess, _immich_context_len

    try:
        if _model is not None:
            try:
                _model.to("cpu")
            except Exception:
                pass
            _model = None
        _processor = None
        _openclip_preprocess = None
        _openclip_tokenizer = None

        _immich_text_sess = None
        _immich_visual_sess = None
        _immich_tokenizer = None
        _immich_preprocess = None
        _immich_context_len = None

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
    load_model()

    # --- Immich ONNX ---
    if _is_immich_model(MODEL_NAME):
        inputs = body.get("input", None)
        if not isinstance(inputs, list) or not inputs:
            raise HTTPException(400, "input must be a non-empty list")
        normalize = bool(body.get("normalize", True))

        if isinstance(inputs[0], str):
            vecs = _immich_encode_text([str(x) for x in inputs], normalize=normalize)
            out = _to_list(vecs)
            dim = len(out[0]) if out else 0
            return {"embeddings": out, "dim": dim, "model": MODEL_NAME, "modality": "text", "backend": "onnxruntime"}

        tmp_files: List[str] = []
        try:
            batch = VLMixedBatch(**body)
        except Exception as e:
            raise HTTPException(400, f"invalid body for immich image mode: {e}")

        images: List[Image.Image] = []
        try:
            for item in batch.input:
                images.append(_load_pil_image_from_item(item, tmp_files))
            vecs = _immich_encode_image(images, normalize=normalize)
            out = _to_list(vecs)
            dim = len(out[0]) if out else 0
            return {"embeddings": out, "dim": dim, "model": MODEL_NAME, "modality": "image", "backend": "onnxruntime"}
        finally:
            for p in tmp_files:
                try:
                    if p and os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

    # --- Everything else ---
    name = MODEL_NAME.lower()

    if name in ["bge-m3", "baai/bge-m3"]:
        try:
            args = TextBatch(**body)
        except Exception as e:
            raise HTTPException(400, f"invalid body for bge-m3: {e}")
        try:
            with torch.inference_mode():
                res = _model.encode(args.input, batch_size=args.batch, max_length=args.max_length)
                out = res["dense_vecs"]
                if args.normalize:
                    import torch.nn.functional as F
                    out = out if isinstance(out, torch.Tensor) else torch.tensor(out)
                    out = F.normalize(out, p=2, dim=-1)
        except Exception as e:
            import traceback
            raise HTTPException(500, f"bge-m3 encode failed: {e}\n{traceback.format_exc()}")
        out = _to_list(out)
        return {"embeddings": out, "dim": (len(out[0]) if out else 0), "model": MODEL_NAME, "modality": "text"}

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
        kwargs = {
            "batch_size": batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": normalize,
            "show_progress_bar": False,
            "device": device_arg,
        }
        if prompt_name:
            kwargs["prompt_name"] = prompt_name

        try:
            vecs = _model.encode(items, **kwargs)
        except Exception as e:
            import traceback
            raise HTTPException(500, f"qwen3 encode failed: {e}\n{traceback.format_exc()}")

        vecs = _to_list(vecs)
        return {"embeddings": vecs, "dim": (len(vecs[0]) if vecs else 0), "model": MODEL_NAME, "modality": "text"}

    if name in ["openclip-vit-b-16-siglip2", "openclip-siglip2-vit-b-16", "vit-b-16-siglip2"]:
        if _openclip_preprocess is None or _openclip_tokenizer is None:
            raise HTTPException(500, "OpenCLIP preprocess/tokenizer not initialized")

        inputs = body.get("input", None)
        if not isinstance(inputs, list) or not inputs:
            raise HTTPException(400, "input must be a non-empty list")

        normalize = bool(body.get("normalize", True))

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
            return {"embeddings": vecs, "dim": (len(vecs[0]) if vecs else 0), "model": MODEL_NAME, "modality": "text"}

        tmp_files: List[str] = []
        try:
            batch = VLMixedBatch(**body)
        except Exception as e:
            raise HTTPException(400, f"invalid body for openclip image mode: {e}")

        imgs: List[torch.Tensor] = []
        try:
            for item in batch.input:
                pil = _load_pil_image_from_item(item, tmp_files)
                imgs.append(_openclip_preprocess(pil).unsqueeze(0))
            image_tensor = torch.cat(imgs, dim=0).to(DEVICE)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                vecs = _model.encode_image(image_tensor, normalize=normalize)

            vecs = _to_list(vecs)
            return {"embeddings": vecs, "dim": (len(vecs[0]) if vecs else 0), "model": MODEL_NAME, "modality": "image"}
        finally:
            for p in tmp_files:
                try:
                    if p and os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

    if name.startswith("siglip2-"):
        if _processor is None:
            raise HTTPException(500, "SigLip2 processor not initialized")

        inputs = body.get("input", None)
        if not isinstance(inputs, list) or not inputs:
            raise HTTPException(400, "input must be a non-empty list")

        normalize = bool(body.get("normalize", True))

        if isinstance(inputs[0], str):
            enc = _processor(text=inputs, padding="max_length", max_length=64, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                vecs = _model.get_text_features(**enc)
                if normalize:
                    import torch.nn.functional as F
                    vecs = F.normalize(vecs, p=2, dim=-1)
            vecs = _to_list(vecs)
            return {"embeddings": vecs, "dim": (len(vecs[0]) if vecs else 0), "model": MODEL_NAME, "modality": "text"}

        tmp_files: List[str] = []
        try:
            batch = VLMixedBatch(**body)
        except Exception as e:
            raise HTTPException(400, f"invalid body for siglip2 image mode: {e}")

        images: List[Image.Image] = []
        try:
            for item in batch.input:
                images.append(_load_pil_image_from_item(item, tmp_files))
            enc = _processor(images=images, return_tensors="pt")
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                vecs = _model.get_image_features(**enc)
                if normalize:
                    import torch.nn.functional as F
                    vecs = F.normalize(vecs, p=2, dim=-1)
            vecs = _to_list(vecs)
            return {"embeddings": vecs, "dim": (len(vecs[0]) if vecs else 0), "model": MODEL_NAME, "modality": "image"}
        finally:
            for p in tmp_files:
                try:
                    if p and os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

    # fallback: bge-vl mixed encode
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
            for item in batch.input:
                img_ref = None
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
                    raise HTTPException(400, "each item must provide image_* and/or text")

                images_arg = [img_ref] if img_ref is not None else None
                text_arg = [txt] if txt is not None else None

                if images_arg is not None and text_arg is not None:
                    vec = _model.encode(images=images_arg, text=text_arg)
                elif images_arg is not None:
                    vec = _model.encode(images=images_arg)
                else:
                    vec = _model.encode(text=text_arg)

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

    return {"embeddings": results, "dim": (len(results[0]) if results else 0), "model": MODEL_NAME}
