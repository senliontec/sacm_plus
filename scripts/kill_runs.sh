#!/usr/bin/env bash
# 一键停止服务器上所有 sacm 实验进程并释放 GPU
#
# 用法:
#   bash scripts/kill_runs.sh           # 优雅停止(先 SIGTERM,5 秒后清残留)
#   bash scripts/kill_runs.sh --force   # 立即 SIGKILL
#
# 目标: src/train/trainer.py、src/eval/evaluate.py、scripts/run_experiments.py
# (参考 fencing-algs/kill_runs.sh 的模式)
set -u

PATTERN='src/train/trainer\.py|src/eval/evaluate\.py|scripts/run_experiments\.py'
FORCE="${1:-}"

echo "==> 查找实验进程 (pattern: $PATTERN)"
ps -eo pid,etime,cmd | grep -E "$PATTERN" | grep -v grep || true

PIDS=$(pgrep -f "$PATTERN" || true)
if [ -z "$PIDS" ]; then
    echo "==> 没有找到运行中的实验进程"
else
    if [ "$FORCE" = "--force" ]; then
        echo "==> 强制停止 (SIGKILL): $PIDS"
        kill -9 $PIDS
    else
        echo "==> 优雅停止 (SIGTERM): $PIDS"
        kill -TERM $PIDS
        echo "==> 等待 5 秒让进程保存/清理..."
        sleep 5
        LEFTOVER=$(pgrep -f "$PATTERN" || true)
        if [ -n "$LEFTOVER" ]; then
            echo "==> 清理残留进程 (SIGKILL): $LEFTOVER"
            kill -9 $LEFTOVER
        fi
    fi
    sleep 1
    echo "==> 复查进程"
    ps -eo pid,etime,cmd | grep -E "$PATTERN" | grep -v grep || echo "    (全部已退出)"
fi

echo "==> GPU 状态"
if command -v nvidia-smi >/dev/null 2>&1; then
    APPS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)
    if [ -n "$APPS" ]; then
        echo "$APPS"
        echo "    ⚠ 仍有 GPU 进程占用(可能是其他用户的作业,勿强杀)"
    else
        echo "    GPU 已全部释放 ✓"
    fi
else
    echo "    (nvidia-smi 不可用,跳过)"
fi
