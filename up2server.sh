#!/usr/bin/env bash
# Upload the full experiment bundle to the server (code + checkpoint + data).
# Default uploads EVERYTHING needed to run: code, checkpoints/, datasets/, data/.
#   ./up2server.sh --code-only   # 只传代码(快速迭代;数据/权重已在服务器上)
# NOTE: results/outputs/logs 等运行产物永不传(服务器自己生成);
#       本脚本与 .gitignore 无关——git 提交仍排除 pth/数据集。
set -euo pipefail

SERVER="zkyd@172.16.1.7"
TARGET="/home/zkyd/code/sacm-plus"
SRC="$(cd "$(dirname "$0")" && pwd)"

CODE_ONLY=0
for a in "$@"; do case "$a" in
  --code-only) CODE_ONLY=1 ;;
  -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
  *) echo "未知参数: $a"; exit 1 ;;
esac; done

EXCLUDES=(--exclude '__pycache__' --exclude '*.pyc'
          --exclude '.git/' --exclude '.codegraph/' --exclude '.pytest_cache/'
          --exclude '.idea/' --exclude '.ruff_cache/'
          --exclude 'results/' --exclude 'outputs/'
          --exclude 'logs/' --exclude 'wandb/' --exclude '3rd/')
[ "$CODE_ONLY" -eq 1 ] && EXCLUDES+=(--exclude 'checkpoints/' --exclude '*.pth'
                                     --exclude 'datasets/' --exclude 'data/')

if [ "$CODE_ONLY" -eq 1 ]; then
  echo "Uploading ${SRC} → ${SERVER}:${TARGET} (code only) ..."
else
  echo "Uploading ${SRC} → ${SERVER}:${TARGET} (code + checkpoint + data) ..."
fi
ssh "${SERVER}" "mkdir -p ${TARGET}"

rsync -avz --progress --delete \
    "${EXCLUDES[@]}" \
    "${SRC}/" "${SERVER}:${TARGET}/"

echo ""
echo "Done. On server:"
echo "  cd ${TARGET}"
echo "  bash scripts/run_all.sh                     # 一键: 环境 → 数据 → 训练 → 评测 → 报告"
echo "  PRESET=stage2 EPOCHS=30 bash scripts/run_all.sh   # 环境变量覆盖配置"
