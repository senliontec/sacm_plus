#!/usr/bin/env bash
# Upload the full experiment bundle to the server (code + checkpoint + data).
# Default uploads EVERYTHING needed to run: code, checkpoints/, datasets/, data/.
#   ./up2server.sh --code-only   # 只传代码(快速迭代;数据/权重已在服务器上)
# NOTE: results/ 等运行产物永不传(服务器自己生成);
#       本脚本与 .gitignore 无关——git 提交仍排除 pth/数据集。
set -euo pipefail

SERVER="zkyd@172.16.1.7"
# The server checkout keeps its historical name so existing data, checkpoints
# and workdirs remain addressable; SACM-X is the framework/method name.
TARGET="/home/zkyd/code/sacm-plus"
SRC="$(cd "$(dirname "$0")" && pwd)"

# Reuse one SSH connection for all rsync calls: ask for the password once.
CONTROL_DIR="${TMPDIR:-/tmp}/sacm-x-ssh-$(id -u)"
mkdir -p "$CONTROL_DIR"
chmod 700 "$CONTROL_DIR"
CONTROL_PATH="$CONTROL_DIR/%C"
SSH_OPTS=(-o ControlMaster=auto -o ControlPath="$CONTROL_PATH"
          -o ControlPersist=600)
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

CODE_ONLY=0
for a in "$@"; do case "$a" in
  --code-only) CODE_ONLY=1 ;;
  -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
  *) echo "未知参数: $a"; exit 1 ;;
esac; done

EXCLUDES=(--exclude '__pycache__' --exclude '*.pyc'
          --exclude '.git/' --exclude '.codegraph/' --exclude '.pytest_cache/'
          --exclude '.idea/' --exclude '.ruff_cache/'
          --exclude 'results/' --exclude '3rd/'
          # Runtime artifacts: never delete server-trained checkpoints.
          --exclude 'checkpoints/')
[ "$CODE_ONLY" -eq 1 ] && EXCLUDES+=(--exclude 'checkpoints/' --exclude '*.pth'
                                     --exclude 'datasets/' --exclude 'data/')

if [ "$CODE_ONLY" -eq 1 ]; then
  echo "Uploading ${SRC} → ${SERVER}:${TARGET} (code only) ..."
else
  echo "Uploading ${SRC} → ${SERVER}:${TARGET} (code + checkpoint + data) ..."
fi

# Establish the shared connection once (and create the target directory).
echo "Authenticating ${SERVER} (one password prompt) ..."
ssh "${SSH_OPTS[@]}" "$SERVER" "mkdir -p '${TARGET}'"

# 代码目录:--checksum 按内容比较，只传真正改动的文件；--update 不覆盖服务器上更新的文件。
# 绝不使用 --delete: 服务器独有的 checkpoints/results/临时脚本
# 必须在上传后继续存在。需要清理服务器上的陈旧代码时，请手动处理。
rsync -e "$RSYNC_SSH" -avz --partial --update --checksum \
    --info=progress2 \
    "${EXCLUDES[@]}" --exclude 'datasets/' --exclude 'data/' \
    "${SRC}/" "${SERVER}:${TARGET}/"

# 数据目录:同样按内容增量上传、不带 --delete —— 本地数据不完整时绝不删除服务器文件
# (2026-10-09 教训:--delete 曾在上传时删掉服务器掩码,训练中途才崩)
if [ "$CODE_ONLY" -eq 0 ]; then
  # Checkpoints are uploaded incrementally WITHOUT --delete: a server-only
  # best_model.pth (trained on the server) must never be removed by a sync.
  for d in checkpoints datasets data; do
    [ -d "${SRC}/$d" ] || continue
    echo "Uploading $d: (incremental, no --delete) ..."
    rsync -e "$RSYNC_SSH" -avz --partial --update --checksum \
        --info=progress2 \
        "${SRC}/$d/" "${SERVER}:${TARGET}/$d/"
  done
fi

# Close the shared connection; ControlPersist would also do it later.
ssh "${SSH_OPTS[@]}" -O exit "$SERVER" >/dev/null 2>&1 || true

echo ""
echo "Done. On server:"
echo "  cd ${TARGET}"
echo "  bash scripts/run_paper.sh --model all --seeds all --gpus 0 --resume  # 一键: 训练 → 评测 → 表格 → 图"
echo "  bash scripts/run_paper.sh --model all --seeds all --tables-only       # 只重生成论文表格/图"
