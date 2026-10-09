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

# 不再单独 ssh mkdir:rsync 会自动创建目标目录,这样全程只需输入一次密码
# 代码目录:--delete 保持本地 = 服务器一致
rsync -avz --partial --info=progress2 --delete \
    "${EXCLUDES[@]}" --exclude 'datasets/' --exclude 'data/' \
    "${SRC}/" "${SERVER}:${TARGET}/"

# 数据目录:增量上传、不带 --delete —— 本地数据不完整时绝不删除服务器文件
# (2026-10-09 教训:--delete 曾在上传时删掉服务器掩码,训练中途才崩)
if [ "$CODE_ONLY" -eq 0 ]; then
  for d in datasets data; do
    [ -d "${SRC}/$d" ] || continue
    echo "Uploading data: $d (no --delete) ..."
    rsync -avz --partial --info=progress2 \
        "${SRC}/$d/" "${SERVER}:${TARGET}/$d/"
  done
fi

echo ""
echo "Done. On server:"
echo "  cd ${TARGET}"
echo "  bash scripts/run_all.sh                     # 一键: 环境 → 数据 → 训练 → 评测 → 报告"
echo "  PRESET=stage2 EPOCHS=30 bash scripts/run_all.sh   # 环境变量覆盖配置"
