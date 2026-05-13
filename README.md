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
| `Dockerfile_l4t` | `amd_l4t` | `l4t` | `arm_l4t` |
| `Dockerfile_cu124` | `amd_cu124` | - | `arm_cu124` |
| `Dockerfile_cu128` | `amd_cu128` | - | `arm_cu128` |
| `Dockerfile_thor` | `thor` | `thor` | `thor` |

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

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_ROOT` | `/root/models` | 模型文件根目录 |
| `PRELOAD_MODELS` | `vit-b-16-siglip2__webli` | 启动时预加载的模型 |
| `PIN_MODELS` | `vit-b-16-siglip2__webli` | 常驻不卸载的模型 |
| `MODEL_PORT_BASE` | `18008` | Worker 端口起始值 |
| `MAX_WORKERS` | `3` | 全局最大并发 Worker 数 |
| `PER_MODEL_MAX` | `2` | 单模型最大副本数 |
| `IDLE_TIMEOUT_S` | `900` | 空闲超时秒数（超时自动卸载） |
| `EVICTOR_INTERVAL_S` | `60` | 驱逐检查间隔 |
| `PORT` | `18000` | 网关监听端口 |
