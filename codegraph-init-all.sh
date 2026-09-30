#!/bin/bash
# 对当前目录下每个子文件夹分别执行 codegraph init
#
# 用法:
#   ./codegraph-init-all.sh          # 跳过已初始化(.codegraph 存在)的子目录
#   ./codegraph-init-all.sh -f       # 强制全部重新索引(已初始化 → codegraph index;
#                                    # 未初始化 → codegraph init;注意 init 对已初始化目录是无效操作)
#   ./codegraph-init-all.sh -l LOG   # 指定日志文件(默认 ./codegraph-init-all.log)
#
# 也可以指定目录列表:
#   ./codegraph-init-all.sh SACM nnUNet

set -u

FORCE=0
LOG="./codegraph-init-all.log"

while getopts "fl:" opt; do
  case "$opt" in
    f) FORCE=1 ;;
    l) LOG="$OPTARG" ;;
    *) echo "用法: $0 [-f] [-l LOG] [dirs...]"; exit 1 ;;
  esac
done
shift $((OPTIND - 1))

if [ "$#" -gt 0 ]; then
  # 使用命令行指定的目录
  dirs=("$@")
else
  # 默认: 当前目录下所有子文件夹(排除隐藏目录)
  dirs=()
  for d in */; do
    [ -d "$d" ] && dirs+=("${d%/}")
  done
fi

if ! command -v codegraph >/dev/null 2>&1; then
  echo "错误: 未找到 codegraph 命令, 请先安装" >&2
  exit 1
fi

: > "$LOG"
ok=0; fail=0; skip=0

for name in "${dirs[@]}"; do
  if [ ! -d "$name" ]; then
    echo "跳过不存在的目录: $name" | tee -a "$LOG"
    skip=$((skip + 1))
    continue
  fi
  if [ -d "$name/.codegraph" ]; then
    # 已初始化:默认跳过;-f 时重新索引
    # (codegraph init 对已初始化目录打印 Already initialized 后直接返回,
    #  必须用 codegraph index 才是真正的重建)
    if [ "$FORCE" -eq 0 ]; then
      echo "SKIP $name (已初始化, 用 -f 强制重跑)" | tee -a "$LOG"
      skip=$((skip + 1))
      continue
    fi
    echo "=== START $name (index) $(date +%H:%M:%S) ===" | tee -a "$LOG"
    if (cd "$name" && codegraph index -q >>"$LOG" 2>&1); then
      echo "=== DONE  $name $(date +%H:%M:%S) ===" | tee -a "$LOG"
      ok=$((ok + 1))
    else
      echo "=== FAIL  $name (exit $?) $(date +%H:%M:%S) ===" | tee -a "$LOG"
      fail=$((fail + 1))
    fi
    continue
  fi
  echo "=== START $name $(date +%H:%M:%S) ===" | tee -a "$LOG"
  if (cd "$name" && codegraph init -y >>"$LOG" 2>&1); then
    echo "=== DONE  $name $(date +%H:%M:%S) ===" | tee -a "$LOG"
    ok=$((ok + 1))
  else
    echo "=== FAIL  $name (exit $?) $(date +%H:%M:%S) ===" | tee -a "$LOG"
    fail=$((fail + 1))
  fi
done

echo
echo "完成: 成功 $ok, 失败 $fail, 跳过 $skip | 详细日志: $LOG"
[ "$fail" -eq 0 ]
