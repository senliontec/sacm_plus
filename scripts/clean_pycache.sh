#!/bin/bash
# 一键清除项目内所有 __pycache__ 目录与散落的 .pyc/.pyo 文件
#
# 用法:
#   ./clean_pycache.sh              # 清理仓库根(脚本所在目录的上一级)
#   ./clean_pycache.sh <dir>        # 清理指定目录
#   ./clean_pycache.sh -n           # 干跑:只列出将删除的内容,不删除
#   ./clean_pycache.sh -n <dir>

set -u

DRY_RUN=0
TARGET=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    -n|--dry-run) DRY_RUN=1; shift ;;
    -h|--help) echo "用法: $0 [-n|--dry-run] [目录]"; exit 0 ;;
    *) TARGET="$1"; shift ;;
  esac
done

# 默认目标:本脚本所在目录(scripts/)的上一级 = 仓库根
if [ -z "$TARGET" ]; then
  TARGET="$(cd "$(dirname "$0")" && pwd)/.."
fi

if [ ! -d "$TARGET" ]; then
  echo "错误: 目录不存在: $TARGET" >&2
  exit 1
fi

# 先收集、后删除:删除过程中目录树会变化,边找边删不可靠
EXCLUDE=(-not -path "*/.git/*" -not -path "*/node_modules/*")
mapfile -t DIRS < <(find "$TARGET" -type d -name "__pycache__" "${EXCLUDE[@]}" 2>/dev/null)
mapfile -t FILES < <(find "$TARGET" -type f \( -name "*.pyc" -o -name "*.pyo" \) "${EXCLUDE[@]}" 2>/dev/null)

n_dirs=${#DIRS[@]}
n_files=${#FILES[@]}

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] 目标: $TARGET"
  for d in "${DIRS[@]}"; do echo "  目录 $d"; done
  for f in "${FILES[@]}"; do echo "  文件 $f"; done
  echo "[dry-run] 将删除 $n_dirs 个 __pycache__ 目录, $n_files 个散落 .pyc/.pyo 文件"
  exit 0
fi

for d in "${DIRS[@]}"; do rm -rf "$d"; done
for f in "${FILES[@]}"; do rm -f "$f"; done

echo "已清理 $TARGET: 删除 $n_dirs 个 __pycache__ 目录, $n_files 个散落 .pyc/.pyo 文件"
