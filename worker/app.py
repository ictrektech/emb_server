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

# === NEW: ONNXRuntime + HF download for immich-app/ViT-B-16-SigLIP2__webli ===
import onnxruntime as ort
from huggingface_hub import snapshot_download

app = FastAPI()

# -------- 环境 --------
MODEL_NAME = os.getenv("MODEL_NAME", "bge-m3").strip()
MODEL_ROOT = os.getenv("MODEL_ROOT", "/root/models").strip()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 可选：显式指定 open-clip 本地目录（最稳）
# 例如：OPENCLIP_MODEL_DIR=/home/jhu/dev/models/embs/ViT-B-16-SigLIP2
OPENCLIP_MODEL_DIR = os.getenv("OPENCLIP_MODEL_DIR", "").strip()

# === NEW: immich SigLIP2 webli (ONNX) 目录/缓存 ===
# 你可以：
# - 本地：MODEL_ROOT/ViT-B-16-SigLIP2__webli
# - 或显式指定：IMMICH_SIGLIP2_WEBLI_DIR=/path/to/ViT-B-16-SigLIP2__webli
# - 或自动从 HF 下载到上述目录
IMMICH_SIGLIP2_WEBLI_DIR = os.getenv("IMMICH_SIGLIP2_WEBLI_DIR", "").strip()  # optional override
HF_CACHE_DIR = os.getenv("HF_CACHE_DIR", "").strip()  # optional (huggingface_hub cache_dir)

# 进程内模型缓存（懒加载）+ 启动锁（避免并发冷启动竞态）
_model = None
_processor = None  # 给 transformers SigLip2 / 其他需要 processor 的模型用

# open-clip 相关
_openclip_preprocess = None
_openclip_tokenizer = None

# === NEW: immich onnx session + preprocess cfg ===
_immich_onnx_sess = None
_immich_preprocess_cfg = None
_immich_visual_input_name = "image"

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
    给 SigLip2 / BGE-VL / OpenCLIP / immich-onnx 用，把 VLItem 转成 PIL.Image
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


# === NEW: immich preprocess helpers (resize + center crop + normalize) ===
def _pil_resize_shorter_side(img: Image.Image, size: int, resample=Image.BICUBIC) -> Image.Image:
    """immich_ml 的 resize_pil 是“先 resize 再 crop”。这里按最常见实现：把短边缩放到 size，并保持比例。"""
    w, h = img.size
    if w == 0 or h == 0:
        raise HTTPException(400, "invalid image with zero width/height")
    if w < h:
        new_w = size
        new_h = int(round(h * (size / w)))
    else:
        new_h = size
        new_w = int(round(w * (size / h)))
    return img.resize((new_w, new_h), resample=resample)

def _pil_center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w < size or h < size:
        # 兜底：如果 resize 逻辑因为奇怪输入没到 size，再强制 resize 成正方形
        return img.resize((size, size), Image.BICUBIC)
    left = int(round((w - size) / 2.0))
    top = int(round((h - size) / 2.0))
    return img.crop((left, top, left + size, top + size))

def _immich_onnx_transform(img: Image.Image) -> np.ndarray:
    """
    对齐 immich_ml.models.visual.OpenClipVisualEncoder.transform：
    - resize -> crop -> to_numpy -> normalize(mean/std) -> CHW -> add batch
    返回 float32, shape=(1,3,H,W)
    """
    global _immich_preprocess_cfg
    if not isinstance(_immich_preprocess_cfg, dict):
        raise RuntimeError("immich preprocess_cfg not loaded")

    cfg = _immich_preprocess_cfg
    size = cfg.get("size", 224)
    if isinstance(size, list):
        size = int(size[0])
    else:
        size = int(size)

    interp = str(cfg.get("interpolation", "bicubic")).lower()
    if "bilinear" in interp:
        resample = Image.BILINEAR
    elif "lanczos" in interp:
        resample = Image.LANCZOS
    else:
        resample = Image.BICUBIC

    mean = np.array(cfg.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32)
    std = np.array(cfg.get("std", [0.5, 0.5, 0.5]), dtype=np.float32)

    img = img.convert("RGB")
    img = _pil_resize_shorter_side(img, size, resample=resample)
    img = _pil_center_crop(img, size)

    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC [0,1]
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # CHW
    arr = np.expand_dims(arr, axis=0)  # 1,3,H,W
    return arr.astype(np.float32)


def _immich_onnx_resolve_dir() -> str:
    """
    返回 ViT-B-16-SigLIP2__webli 的本地目录（包含 visual/ 子目录）。
    优先级：
    1) IMMICH_SIGLIP2_WEBLI_DIR
    2) MODEL_ROOT/ViT-B-16-SigLIP2__webli
    3) snapshot_download -> 写到 MODEL_ROOT/ViT-B-16-SigLIP2__webli
    """
    if IMMICH_SIGLIP2_WEBLI_DIR and os.path.isdir(IMMICH_SIGLIP2_WEBLI_DIR):
        return IMMICH_SIGLIP2_WEBLI_DIR

    local_dir = os.path.join(MODEL_ROOT, "ViT-B-16-SigLIP2__webli")
    if os.path.isdir(local_dir):
        return local_dir

    # 下载到本地目录（可复用/可缓存）
    kwargs = {
        "repo_id": "immich-app/ViT-B-16-SigLIP2__webli",
        "local_dir": local_dir,
        "local_dir_use_symlinks": False,
    }
    if HF_CACHE_DIR:
        kwargs["cache_dir"] = HF_CACHE_DIR

    try:
        snapshot_download(**kwargs)
    except Exception as e:
        raise RuntimeError(f"snapshot_download immich-app/ViT-B-16-SigLIP2__webli failed: {e}")
    return local_dir


def _immich_onnx_load():
    """
    加载 immich-app/ViT-B-16-SigLIP2__webli 视觉 encoder（ONNX）：
    目录结构（HF 上）：
      ViT-B-16-SigLIP2__webli/
        visual/
          model.onnx
          preprocess_cfg.json
        rknpu/...
    """
    global _immich_onnx_sess, _immich_preprocess_cfg, _immich_visual_input_name
    if _immich_onnx_sess is not None:
        return

    model_dir = _immich_onnx_resolve_dir()
    visual_dir = os.path.join(model_dir, "visual")
    onnx_path = os.path.join(visual_dir, "model.onnx")
    cfg_path = os.path.join(visual_dir, "preprocess_cfg.json")

    if not os.path.isfile(onnx_path):
        raise RuntimeError(f"immich onnx not found: {onnx_path}")
    if not os.path.isfile(cfg_path):
        raise RuntimeError(f"immich preprocess cfg not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        _immich_preprocess_cfg = json.load(f)

    # providers：尊重 ORT 当前环境可用 provider 顺序
    providers = ort.get_available_providers()
    _immich_onnx_sess = ort.InferenceSession(onnx_path, providers=providers)

    # 输入名（大概率是 "image"；做一次探测更稳）
    try:
        ins = _immich_onnx_sess.get_inputs()
        if ins and getattr(ins[0], "name", None):
            _immich_visual_input_name = ins[0].name
    except Exception:
        _immich_visual_input_name = "image"


# -------- Lazy load --------
def load_model():
    """
    懒加载指定模型：
    - bge-m3: FlagEmbedding BGEM3FlagModel（禁用内置 fp16，避免 .half 报错）
    - bge-vl: transformers AutoModel (trust_remote_code)
    - qwen3-embedding: sentence-transformers SentenceTransformer（构造期不 to）
    - siglip2-*: google/siglip2-... AutoModel + AutoProcessor
    - openclip-vit-b-16-siglip2: open-clip-torch create_model_from_pretrained + get_tokenizer
    - === NEW === vit-b-16-siglip2__webli: immich 的 ONNX 视觉 encoder（onnxruntime）
    """
    global _model, _processor, _openclip_preprocess, _openclip_tokenizer, _immich_onnx_sess
    # immich onnx 走独立 session，不占用 _model
    if _model is not None or _immich_onnx_sess is not None:
        return

    with _start_lock:
        if _model is not None or _immich_onnx_sess is not None:
            return

        name = MODEL_NAME.lower()

        # === NEW: immich-app/ViT-B-16-SigLIP2__webli (ONNX visual) ===
        if name in ["vit-b-16-siglip2__webli", "immich-app/vit-b-16-siglip2__webli", "immich/vit-b-16-siglip2__webli"]:
            _immich_onnx_load()
            return

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
        global _model, _processor, _openclip_preprocess, _openclip_tokenizer, _immich_onnx_sess, _immich_preprocess_cfg
        if _model is not None:
            try:
                _model.to("cpu")
            except Exception:
                pass
            _model = None
        _processor = None
        _openclip_preprocess = None
        _openclip_tokenizer = None

        # immich onnx
        _immich_onnx_sess = None
        _immich_preprocess_cfg = None

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
    - === NEW === vit-b-16-siglip2__webli (immich onnx):
        * input 为 list[VLItem] -> ORT session.run({"image": ...}) -> embeddings
    """
    load_model()
    name = MODEL_NAME.lower()

    # === NEW: immich-app/ViT-B-16-SigLIP2__webli (ONNX visual) ===
    if name in ["vit-b-16-siglip2__webli", "immich-app/vit-b-16-siglip2__webli", "immich/vit-b-16-siglip2__webli"]:
        if _immich_onnx_sess is None:
            raise HTTPException(500, "immich onnx session not initialized")

        try:
            batch = VLMixedBatch(**body)
        except Exception as e:
            raise HTTPException(400, f"invalid body for immich onnx: {e}")

        if not batch.input:
            raise HTTPException(400, "input must be non-empty list")

        normalize = bool(body.get("normalize", True))
        tmp_files: List[str] = []
        outs: List[List[float]] = []

        try:
            for item in batch.input:
                img = _load_pil_image_from_item(item, tmp_files)
                x = _immich_onnx_transform(img)

                try:
                    y = _immich_onnx_sess.run(None, {_immich_visual_input_name: x})
                except Exception as e:
                    raise HTTPException(500, f"immich onnx run failed: {e}")

                # 约定：第一个输出，shape (1, D) 或 (B, D)
                vec = y[0][0]
                if normalize:
                    n = float(np.linalg.norm(vec) + 1e-12)
                    vec = vec / n
                outs.append(vec.astype(np.float32).tolist())

        finally:
            for p in tmp_files:
                try:
                    if p and os.path.exists(p):
                        os.unlink(p)
                except Exception:
                    pass

        dim = len(outs[0]) if outs else 0
        return {"embeddings": outs, "dim": dim, "model": MODEL_NAME, "modality": "image"}

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
