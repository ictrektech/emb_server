#!/usr/bin/env bash
set -euo pipefail

# =========================
# build_image.sh
# =========================

IMG_NAME="emb_server"

# 飞书配置
FEISHU_CONFIG_FILE="${HOME}/.feishu.json"
FEISHU_SPREADSHEET_TOKEN="Htotsn3oahO1zxt73YMcaB1zn8e"

# profile -> sheet 名称映射
# 后续如需调整，直接改这里
# 根据当前架构决定 CUDA sheet 前缀
case "$(uname -m)" in
  x86_64)
    CUDA_PREFIX="AMD"
    ;;
  aarch64)
    CUDA_PREFIX="ARM"
    ;;
  *)
    CUDA_PREFIX="UNKNOWN"
    ;;
esac

declare -A PROFILE_TO_SHEET_TITLE=(
  ["Dockerfile"]="ARM_without_cuda"
  ["Dockerfile_l4t"]="l4t"
  ["Dockerfile_cu124"]="${CUDA_PREFIX}_with_cuda"
  ["Dockerfile_cu128"]="${CUDA_PREFIX}_with_cuda"
)

# -------------------------
# 基础函数
# -------------------------

log() {
  echo "[INFO] $*"
}

err() {
  echo "[ERROR] $*" >&2
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    err "missing command: $1"
    exit 1
  }
}

read_feishu_field() {
  local field="$1"
  python3 - "$FEISHU_CONFIG_FILE" "$field" <<'PY'
import json, sys
path, field = sys.argv[1], sys.argv[2]
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
val = data.get(field, "")
if not isinstance(val, str):
    val = str(val)
print(val)
PY
}

json_extract_or_fail() {
  local resp="$1"
  local py="$2"
  python3 - "$resp" "$py" <<'PY'
import json, sys
resp = sys.argv[1]
code = sys.argv[2]
if not resp:
    raise SystemExit("empty response")
try:
    data = json.loads(resp)
except Exception as e:
    raise SystemExit(f"invalid json response: {resp[:500]!r}, error={e}")
ns = {"data": data}
exec(code, ns, ns)
PY
}

get_feishu_token() {
  local app_id="$1"
  local app_secret="$2"
  local resp

  resp=$(
    curl --fail -sS -X POST 'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal' \
      -H 'Content-Type: application/json' \
      -d "{
        \"app_id\": \"${app_id}\",
        \"app_secret\": \"${app_secret}\"
      }"
  ) || {
    err "get_feishu_token: curl failed"
    return 1
  }

  python3 - "$resp" <<'PY'
import json, sys
resp = sys.argv[1]
if not resp:
    raise SystemExit("get_feishu_token: empty response")
try:
    data = json.loads(resp)
except Exception as e:
    raise SystemExit(f"get_feishu_token: invalid json: {resp[:500]!r}, error={e}")
if data.get("code") != 0:
    raise SystemExit(f"get_feishu_token failed: {data}")
print(data["tenant_access_token"])
PY
}

feishu_api_json() {
  local method="$1"
  local url="$2"
  local token="$3"
  local body="${4:-}"

  if [[ -n "$body" ]]; then
    curl --fail -sS -X "$method" "$url" \
      -H "Authorization: Bearer ${token}" \
      -H "Content-Type: application/json" \
      --data "$body"
  else
    curl --fail -sS -X "$method" "$url" \
      -H "Authorization: Bearer ${token}"
  fi
}

get_sheet_id_by_title() {
  local token="$1"
  local spreadsheet_token="$2"
  local target_title="$3"
  local resp

  resp=$(
    feishu_api_json "GET" \
      "https://open.feishu.cn/open-apis/sheets/v3/spreadsheets/${spreadsheet_token}/sheets/query" \
      "$token"
  ) || {
    err "get_sheet_id_by_title: curl failed"
    return 1
  }

  python3 - "$target_title" "$resp" <<'PY'
import sys, json
target = sys.argv[1]
resp = sys.argv[2]
if not resp:
    raise SystemExit("get_sheet_id_by_title: empty response")
try:
    data = json.loads(resp)
except Exception as e:
    raise SystemExit(f"get_sheet_id_by_title invalid json: {resp[:500]!r}, error={e}")
if data.get("code") != 0:
    raise SystemExit(f"query sheets failed: {data}")
for s in data["data"]["sheets"]:
    if s.get("title") == target:
        print(s["sheet_id"])
        raise SystemExit(0)
raise SystemExit(f"sheet title not found: {target}")
PY
}

get_range_values() {
  local token="$1"
  local spreadsheet_token="$2"
  local range="$3"

  feishu_api_json "GET" \
    "https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/${spreadsheet_token}/values/${range}" \
    "$token"
}

find_component_column_letter() {
  local token="$1"
  local spreadsheet_token="$2"
  local sheet_id="$3"
  local component_name="$4"
  local resp

  resp=$(get_range_values "$token" "$spreadsheet_token" "${sheet_id}!A1:AZ1") || {
    err "find_component_column_letter: read range failed"
    return 1
  }

  python3 - "$component_name" "$resp" <<'PY'
import sys, json
target = sys.argv[1]
resp = sys.argv[2]
if not resp:
    raise SystemExit("find_component_column_letter: empty response")
try:
    data = json.loads(resp)
except Exception as e:
    raise SystemExit(f"find_component_column_letter invalid json: {resp[:500]!r}, error={e}")
if data.get("code") != 0:
    raise SystemExit(f"read header failed: {data}")
values = data.get("data", {}).get("valueRange", {}).get("values", [])
row = values[0] if values else []
for i, v in enumerate(row, start=1):
    if str(v).strip() == target:
        n = i
        s = ""
        while n > 0:
            n, r = divmod(n - 1, 26)
            s = chr(ord("A") + r) + s
        print(s)
        raise SystemExit(0)
raise SystemExit(f"component column not found in row1: {target}")
PY
}

find_date_row() {
  local token="$1"
  local spreadsheet_token="$2"
  local sheet_id="$3"
  local target_date="$4"
  local resp

  resp=$(get_range_values "$token" "$spreadsheet_token" "${sheet_id}!A4:A2000") || {
    err "find_date_row: read range failed"
    return 1
  }

  python3 - "$target_date" "$resp" <<'PY'
import sys, json
target = sys.argv[1]
resp = sys.argv[2]
if not resp:
    raise SystemExit("find_date_row: empty response")
try:
    data = json.loads(resp)
except Exception as e:
    raise SystemExit(f"find_date_row invalid json: {resp[:500]!r}, error={e}")
if data.get("code") != 0:
    raise SystemExit(f"read date column failed: {data}")
values = data.get("data", {}).get("valueRange", {}).get("values", [])
for idx, row in enumerate(values, start=4):
    if row and str(row[0]).strip() == target:
        print(idx)
        raise SystemExit(0)
print("")
PY
}

prepend_date_row() {
  local token="$1"
  local spreadsheet_token="$2"
  local sheet_id="$3"
  local today="$4"
  local resp

  resp=$(
    feishu_api_json "POST" \
      "https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/${spreadsheet_token}/values_prepend" \
      "$token" \
      "{\"valueRange\":{\"range\":\"${sheet_id}!A4:A4\",\"values\":[[\"${today}\"]]}}"
  ) || {
    err "prepend_date_row: curl failed"
    return 1
  }

  python3 - "$resp" <<'PY'
import json, sys
resp = sys.argv[1]
if not resp:
    raise SystemExit("prepend_date_row: empty response")
try:
    data = json.loads(resp)
except Exception as e:
    raise SystemExit(f"prepend_date_row invalid json: {resp[:500]!r}, error={e}")
if data.get("code") != 0:
    raise SystemExit(f"prepend_date_row failed: {data}")
print("ok")
PY
}

write_cell() {
  local token="$1"
  local spreadsheet_token="$2"
  local sheet_id="$3"
  local cell="$4"
  local value="$5"
  local resp

  resp=$(
    feishu_api_json "PUT" \
      "https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/${spreadsheet_token}/values" \
      "$token" \
      "{\"valueRange\":{\"range\":\"${sheet_id}!${cell}:${cell}\",\"values\":[[\"${value}\"]]}}"
  ) || {
    err "write_cell: curl failed"
    return 1
  }

  python3 - "$resp" <<'PY'
import json, sys
resp = sys.argv[1]
if not resp:
    raise SystemExit("write_cell: empty response")
try:
    data = json.loads(resp)
except Exception as e:
    raise SystemExit(f"write_cell invalid json: {resp[:500]!r}, error={e}")
if data.get("code") != 0:
    raise SystemExit(f"write_cell failed: {data}")
print("ok")
PY
}

# -------------------------
# 参数与架构
# -------------------------

PROFILE="Dockerfile"

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
  case "$1" in
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

case "$PROFILE" in
  Dockerfile)
    PROFILE_TAG="${ARCH_TAG}"
    ;;
  Dockerfile_cu124)
    PROFILE_TAG="${ARCH_TAG}_cu124"
    ;;
  Dockerfile_cu128)
    PROFILE_TAG="${ARCH_TAG}_cu128"
    ;;
  Dockerfile_l4t)
    PROFILE_TAG="${ARCH_TAG}_l4t"
    ;;
  *)
    echo "Unsupported profile: $PROFILE"
    exit 1
    ;;
esac

TARGET_SHEET_TITLE="${PROFILE_TO_SHEET_TITLE[$PROFILE]:-}"
if [[ -z "$TARGET_SHEET_TITLE" ]]; then
  err "No sheet mapping configured for profile: $PROFILE"
  exit 1
fi

# -------------------------
# build args
# -------------------------

BUILD_ARGS=()
if [[ -n "${PROXY:-}" ]]; then
  echo "Using PROXY=${PROXY}"
  BUILD_ARGS+=(--build-arg "PROXY=${PROXY}")
fi

if [[ -n "${USE_OLD_TRANSFORMERS:-}" ]]; then
  echo "Using USE_OLD_TRANSFORMERS=${USE_OLD_TRANSFORMERS}"
  BUILD_ARGS+=(--build-arg "USE_OLD_TRANSFORMERS=${USE_OLD_TRANSFORMERS}")
fi

# -------------------------
# 版本与 tag
# -------------------------

DATE=$(date +%Y%m%d)

if [[ -f "VERSION" ]]; then
  VERSION=$(tr -d ' \t\n\r' < VERSION)
  TAG="${PROFILE_TAG}_${VERSION}_${DATE}"
  echo "Using version from VERSION: ${VERSION}"
else
  TAG="${PROFILE_TAG}_${DATE}"
  echo "No VERSION file found, using default tag format."
fi

IMAGE_URI="swr.cn-southwest-2.myhuaweicloud.com/ictrek/${IMG_NAME}:${TAG}"

# -------------------------
# 前置检查
# -------------------------

require_cmd curl
require_cmd python3
require_cmd docker

if [[ ! -f "$FEISHU_CONFIG_FILE" ]]; then
  err "Feishu config not found: $FEISHU_CONFIG_FILE"
  exit 1
fi

FEISHU_APP_ID="$(read_feishu_field "feishu_app_id")"
FEISHU_APP_SECRET="$(read_feishu_field "feishu_app_secret")"

if [[ -z "$FEISHU_APP_ID" || -z "$FEISHU_APP_SECRET" ]]; then
  err "feishu_app_id or feishu_app_secret missing in $FEISHU_CONFIG_FILE"
  exit 1
fi

# -------------------------
# 构建并推送
# -------------------------

log "PROFILE=${PROFILE}"
log "PROFILE_TAG=${PROFILE_TAG}"
log "TARGET_SHEET_TITLE=${TARGET_SHEET_TITLE}"
log "IMG_NAME=${IMG_NAME}"
log "TAG=${TAG}"

docker build \
  "${BUILD_ARGS[@]}" \
  -t "${IMG_NAME}" \
  -f "$PROFILE" .

docker tag "${IMG_NAME}" "${IMAGE_URI}"
docker push "${IMAGE_URI}"

log "Docker push succeeded: ${IMAGE_URI}"

# -------------------------
# push 成功后写飞书
# -------------------------

FEISHU_TOKEN="$(get_feishu_token "$FEISHU_APP_ID" "$FEISHU_APP_SECRET")"
SHEET_ID="$(get_sheet_id_by_title "$FEISHU_TOKEN" "$FEISHU_SPREADSHEET_TOKEN" "$TARGET_SHEET_TITLE")"
log "Resolved sheet: ${TARGET_SHEET_TITLE} -> ${SHEET_ID}"

FEISHU_TOKEN="$(get_feishu_token "$FEISHU_APP_ID" "$FEISHU_APP_SECRET")"
COMPONENT_COL="$(find_component_column_letter "$FEISHU_TOKEN" "$FEISHU_SPREADSHEET_TOKEN" "$SHEET_ID" "$IMG_NAME")"
log "Resolved component column: ${IMG_NAME} -> ${COMPONENT_COL}"

FEISHU_TOKEN="$(get_feishu_token "$FEISHU_APP_ID" "$FEISHU_APP_SECRET")"
DATE_ROW="$(find_date_row "$FEISHU_TOKEN" "$FEISHU_SPREADSHEET_TOKEN" "$SHEET_ID" "$DATE")"

if [[ -z "$DATE_ROW" ]]; then
  log "Date ${DATE} not found, creating a new row at top of data area"
  FEISHU_TOKEN="$(get_feishu_token "$FEISHU_APP_ID" "$FEISHU_APP_SECRET")"
  prepend_date_row "$FEISHU_TOKEN" "$FEISHU_SPREADSHEET_TOKEN" "$SHEET_ID" "$DATE" >/dev/null
  DATE_ROW=4
else
  log "Date ${DATE} already exists at row ${DATE_ROW}"
fi

FEISHU_TOKEN="$(get_feishu_token "$FEISHU_APP_ID" "$FEISHU_APP_SECRET")"
write_cell "$FEISHU_TOKEN" "$FEISHU_SPREADSHEET_TOKEN" "$SHEET_ID" "${COMPONENT_COL}${DATE_ROW}" "$TAG" >/dev/null

log "Feishu updated: ${TARGET_SHEET_TITLE}!${COMPONENT_COL}${DATE_ROW} = ${TAG}"