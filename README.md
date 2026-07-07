# emb_server

Embedding 模型按需推理服务，支持多模型动态加载 / 卸载，基于 FastAPI + Uvicorn。

> 详细文档：[飞书链接](https://twp99wyutbs.feishu.cn/docx/Y6svdilvxoTk0txPDAwctiT6ndb)

## 项目结构

```
emb_server/
├── manager/          # 网关服务 - 调度 Worker 生命周期
├── worker/           # 推理 Worker - 加载模型并响应请求
├── Dockerfile        # CPU / 无 CUDA 基础镜像
├── Dockerfile_l4t    # Jetson (L4T) 镜像
├── Dockerfile_cu124  # CUDA 12.4 镜像
├── Dockerfile_cu128  # CUDA 12.8 镜像
├── Dockerfile_thor   # Thor (ARM + CUDA) 镜像
├── build_image.sh    # 构建并推送镜像（自动写飞书版本表）
├── start.sh          # 容器入口脚本
└── dc.yml            # docker-compose 配置
```

## 镜像构建

使用 `build_image.sh` 构建 Docker 镜像并推送到华为云 SWR，构建成功后自动将版本号写入飞书多维表格。

### 用法

```bash
# 默认构建（无 CUDA）
./build_image.sh

# 指定 profile
./build_image.sh --profile Dockerfile
./build_image.sh --profile Dockerfile_cu124
./build_image.sh --profile Dockerfile_cu128
./build_image.sh --profile Dockerfile_l4t
./build_image.sh --profile Dockerfile_thor
```

### 可选环境变量

| 变量 | 说明 |
|------|------|
| `PROXY` | 构建时代理地址，如 `http://proxy:7890` |
| `USE_OLD_TRANSFORMERS` | 设为 `1` 使用旧版 transformers |

### Profile 与镜像标签

脚本根据当前机器架构 + profile 自动生成标签，格式为：

```
swr.cn-southwest-2.myhuaweicloud.com/ictrek/emb_server:<PROFILE_TAG>_<VERSION>_<DATE>
```

| Profile | x86_64 标签后缀 | aarch64 (Jetson) 标签后缀 | aarch64 (其他) 标签后缀 |
|---------|----------------|--------------------------|----------------------|
| `Dockerfile` | `amd` | `arm` | `arm` |
| `Dockerfile_l4t` | - | `l4t` | `arm_l4t` |
| `Dockerfile_cu124` | `amd_cu124` | - | `arm_cu124` |
| `Dockerfile_cu128` | `amd_cu128` | - | `arm_cu128` |
| `Dockerfile_thor` | - | - | `thor` |

若项目根目录存在 `VERSION` 文件，标签中会包含版本号；否则仅用日期。

### 飞书版本表映射

构建成功后版本号写入飞书表格，profile 与 Sheet 的对应关系：

| Profile | x86_64 Sheet | aarch64 Sheet |
|---------|-------------|---------------|
| `Dockerfile` | AMD_without_cuda | ARM_without_cuda |
| `Dockerfile_cu124` | AMD_with_cuda | ARM_with_cuda |
| `Dockerfile_cu128` | AMD_with_cuda | ARM_with_cuda |
| `Dockerfile_l4t` | l4t | l4t |
| `Dockerfile_thor` | thor_spark | thor_spark |

### 前置依赖

- Docker（含 NVIDIA runtime，GPU 镜像需要）
- `curl`、`python3`
- 飞书配置文件 `~/.feishu.json`，内容格式：

```json
{
  "feishu_app_id": "<your_app_id>",
  "feishu_app_secret": "<your_app_secret>"
}
```

## 运行

### docker-compose

```bash
docker compose -f dc.yml up
```

### 构建镜像

```bash
./build_image.sh --profile Dockerfile_thor
```

`build_image.sh` 默认使用 `DOCKER_BUILDKIT=0` 构建，避免当前 SWR registry 拒绝 BuildKit manifest。

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_ROOT` | `/root/models` | 模型文件根目录 |
| `PINNED_MODELS` | `vit-b-16-siglip2__webli` | 常驻并启动预热的模型，逗号分隔 |
| `PIN_MODELS` | `vit-b-16-siglip2__webli` | 旧变量名，未设置 `PINNED_MODELS` 时兼容读取 |
| `PRELOAD_MODELS` | `vit-b-16-siglip2__webli` | 旧变量名，未设置 `PINNED_MODELS` / `PIN_MODELS` 时兼容读取 |
| `MODEL_PORT_BASE` | `18008` | Worker 端口起始值 |
| `MAX_WORKERS` | `3` | 全局最大并发 Worker 数 |
| `PER_MODEL_MAX` | `2` | 单模型最大副本数 |
| `IDLE_TIMEOUT_S` | `900` | 空闲超时秒数（超时自动卸载） |
| `EVICTOR_INTERVAL_S` | `60` | 驱逐检查间隔 |
| `PORT` | `18000` | 网关监听端口 |
| `IMMICH_MODEL_DIR` | 空 | Immich ONNX 模型目录，包含 `config.json`、`textual/`、`visual/` |
| `IMMICH_HF_REPO` | `immich-app/ViT-B-16-SigLIP2__webli` | 本地目录不存在时下载的 Hugging Face repo |
| `HF_TOKEN` | 空 | Hugging Face 私有模型 token |
| `OPENCLIP_MODEL_DIR` | 空 | OpenCLIP `local-dir:` 模型目录 |
| `ORT_INTRA_OP_THREADS` | `0` | ONNX Runtime intra-op 线程数，`0` 为默认 |
| `ORT_INTER_OP_THREADS` | `0` | ONNX Runtime inter-op 线程数，`0` 为默认 |

## 支持的模型

| 模型名 / 别名 | 类型 | 后端 | 说明 |
|---------------|------|------|------|
| `vit-b-16-siglip2__webli` | 图文 | ONNX Runtime | Immich `ViT-B-16-SigLIP2__webli` 双塔模型，推荐用于 Immich 接入 |
| `immich-vit-b-16-siglip2__webli` | 图文 | ONNX Runtime | `vit-b-16-siglip2__webli` 别名 |
| `immich-app/vit-b-16-siglip2__webli` | 图文 | ONNX Runtime | `vit-b-16-siglip2__webli` 别名 |
| `immich-app/vit-b-16-siglip2__webli@onnx` | 图文 | ONNX Runtime | `vit-b-16-siglip2__webli` 别名 |
| `bge-m3` / `baai/bge-m3` | 文本 | FlagEmbedding | dense embedding |
| `qwen` / `qwen-embedding` / `qwen3-embedding` / `qwen3-embedding-0.6b` / `qwen/qwen3-embedding-0.6b` | 文本 | SentenceTransformer | Qwen3 embedding |
| `bge-vl` / `baai/bge-vl` / `baai/bge-vl-base` / `baai/bge-vl-large` | 图文 | Transformers | BGE-VL |
| `openclip-vit-b-16-siglip2` / `openclip-siglip2-vit-b-16` / `vit-b-16-siglip2` | 图文 | open_clip | OpenCLIP local-dir 模型 |
| `siglip2-base-patch16-224` / `256` / `384` / `512` | 图文 | Transformers | Google SigLIP2 base |
| `siglip2-large-patch16-256` / `384` / `512` | 图文 | Transformers | Google SigLIP2 large |

## API 用法

### 健康检查

```bash
curl http://127.0.0.1:18000/ping
curl http://127.0.0.1:18000/metrics
```

`/ping` 返回 `pong`，用于 Immich Machine Learning URL 健康检查；`/metrics` 返回已加载 worker、常驻模型和并发状态。

常驻模型只预热 1 个 worker。bge-m3 在单个模型实例内处理请求；并发请求会排队复用同一份 GPU 模型，避免多副本加载占满显存。需要吞吐时优先把多条文本放在同一个 `input` 数组里批量编码。

### 通用文本 embedding

```bash
curl -X POST 'http://127.0.0.1:18000/embed?model=bge-m3' \
  -H 'Content-Type: application/json' \
  -d '{"input":["hello world","你好世界"],"normalize":true,"batch":32}'
```

返回：

```json
{
  "embeddings": [[0.1, 0.2]],
  "dim": 1024,
  "model": "bge-m3",
  "modality": "text"
}
```

### OpenAI embedding 兼容接口

`/v1/embeddings` 和 `/embeddings` 可用于 OpenAI embedding API 调用方：

```bash
curl -X POST 'http://127.0.0.1:18000/v1/embeddings' \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3","input":["hello world"],"encoding_format":"float","truncate_prompt_tokens":8192}'
```

返回字段为 OpenAI 兼容的 `data[].embedding`。

### Qwen3 embedding

```bash
curl -X POST 'http://127.0.0.1:18000/embed?model=qwen3-embedding-0.6b' \
  -H 'Content-Type: application/json' \
  -d '{"input":["query text"],"normalize":true,"prompt_name":"query"}'
```

### 图文模型的文本 embedding

```bash
curl -X POST 'http://127.0.0.1:18000/embed?model=vit-b-16-siglip2__webli' \
  -H 'Content-Type: application/json' \
  -d '{"input":["a dog on the beach"],"normalize":true}'
```

### 图文模型的图片 embedding

```bash
curl -X POST 'http://127.0.0.1:18000/embed?model=vit-b-16-siglip2__webli' \
  -H 'Content-Type: application/json' \
  -d '{"input":[{"image_path":"/root/images/demo.jpg"}],"normalize":true}'
```

图片输入支持：

| 字段 | 说明 |
|------|------|
| `image_path` | 容器内本地图片路径 |
| `image_url` | HTTP/HTTPS URL、本地路径或 `data:<mime>;base64,...` |
| `image_base64` | `data:<mime>;base64,...` |
| `text` | BGE-VL mixed encode 可同时传文本 |

### OpenCLIP / Transformers SigLIP2

```bash
curl -X POST 'http://127.0.0.1:18000/embed?model=openclip-vit-b-16-siglip2' \
  -H 'Content-Type: application/json' \
  -d '{"input":["a photo of a train"],"normalize":true}'

curl -X POST 'http://127.0.0.1:18000/embed?model=siglip2-base-patch16-256' \
  -H 'Content-Type: application/json' \
  -d '{"input":[{"image_url":"https://example.com/image.jpg"}],"normalize":true}'
```

### Immich 兼容接入

将 Immich 的 Machine Learning URL 指向 emb_server 网关：

```bash
IMMICH_MACHINE_LEARNING_URL=http://emb_server:18000
```

emb_server 支持 Immich 对 CLIP 图片和文字 embedding 的接口要求：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/ping` | 返回 `pong` |
| `POST` | `/predict` | `multipart/form-data`，字段为 `entries` 加 `image` 或 `text` |

文字 embedding 请求：

```bash
curl -X POST http://127.0.0.1:18000/predict \
  -F 'entries={"clip":{"textual":{"modelName":"ViT-B-16-SigLIP2__webli","options":{"language":"zh-Hans"}}}}' \
  -F 'text=一张海边列车的照片'
```

返回格式：

```json
{
  "clip": "[0.1,0.2,...]"
}
```

图片 embedding 请求：

```bash
curl -X POST http://127.0.0.1:18000/predict \
  -F 'entries={"clip":{"visual":{"modelName":"ViT-B-16-SigLIP2__webli"}}}' \
  -F 'image=@/path/to/demo.jpg'
```

返回格式：

```json
{
  "clip": "[0.1,0.2,...]",
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

说明：

- Immich 兼容输出中的 `clip` 是 JSON 数组字符串，和 Immich 原生 ML 服务一致。
- `/predict` 默认不额外 L2 normalize，以贴近 Immich 原生 ONNX 输出；通用 `/embed` 默认 `normalize=true`。
- 当前 `/predict` 只实现 CLIP `visual` / `textual`，不处理 face embedding 和 OCR。

## tc232 测试示例

tc232 上已有模型目录 `/home/jhu/dev/models/embs`，可用 CUDA 12.8 profile 构建/运行：

```bash
ssh tc232
cd /path/to/emb_server
docker build -f Dockerfile_cu128 -t emb_server:cu128 .
docker run --rm --gpus all \
  -p 18000:18000 \
  -v /home/jhu/dev/models/embs:/root/models:ro \
  -e MODEL_ROOT=/root/models \
  -e PINNED_MODELS=vit-b-16-siglip2__webli \
  emb_server:cu128
```

启动后可用 `/ping`、`/embed?model=vit-b-16-siglip2__webli` 和 `/predict` 验证。
