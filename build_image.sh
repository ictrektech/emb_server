#!/bin/bash

# build_image.sh
IMG_NAME="emb_server" #


# 获取架构
ARCH=$(uname -m)

case "$ARCH" in
  aarch64)
    if [[ -f "/etc/nv_tegra_release" ]] || grep -qi "nvidia" /proc/device-tree/model 2>/dev/null; then
      ARCH_TAG="jet"
    else
      ARCH_TAG="arm"
    fi
    ;;
  x86_64)
    ARCH_TAG="amd"
    ;;
  *)
    ARCH_TAG="unknown"
    ;;
esac


while [[ $# -gt 0 ]]; do
  case $1 in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 根据 PROFILE 生成 PROFILE_TAG
case "$PROFILE" in
  Dockerfile)
    PROFILE_TAG="${ARCH_TAG}"
    ;;
  Dockerfile_cu124)
    PROFILE_TAG="${ARCH_TAG}_cu124"
    ;;
  Dockerfile_l4t)
    PROFILE_TAG="${ARCH_TAG}_l4t"
    ;;
  *)
    echo "Unsupported profile: $PROFILE"
    exit 1
    ;;
esac


# 获取日期
DATE=$(date +%Y%m%d)

# 检查 version.txt
if [[ -f "VERSION" ]]; then
    VERSION=$(cat VERSION | tr -d ' \t\n\r')
    TAG="${PROFILE_TAG}_${VERSION}_${DATE}"
    echo "Using version from VERSION: ${VERSION}"
else
    TAG="${PROFILE_TAG}_${DATE}"
    echo "No VERSION file found, using default tag format."
fi


# 构建并推送镜像
docker build \
    --build-arg PROXY=$PROXY \
    -t emb_server \
    -f $PROFILE .
docker tag emb_server swr.cn-southwest-2.myhuaweicloud.com/ictrek/${IMG_NAME}:${TAG}

docker push swr.cn-southwest-2.myhuaweicloud.com/ictrek/${IMG_NAME}:${TAG}