#!/bin/zsh
# F2 + F4: quota 重置后立刻推送 S6 会话 2（train，fold 3–4）和 S6 教师全量（teacher-s6，fold0_train）。
# 两个会话并行：train 挂缓存分片，teacher-s6 不挂（final-week-plan 的并行授权）。
# 推送后并行 watch_start 25 分钟；退出码 2 = 卡启动，需要人在网页 Active Events 停掉后重推。
set -uo pipefail
cd "$(dirname "$0")/.."

echo "== quota check =="
used=$(uv run python scripts/kx.py quota | python3 -c 'import json,sys; print(json.load(sys.stdin)["gpu_used_hours"])')
echo "gpu_used_hours=$used"
python3 -c "exit(0 if float('$used') < 1.0 else 1)" || { echo "Quota not reset yet (used=$used h); aborting."; exit 1; }

uv run python scripts/kx.py quota > experiments/S6/session-2-quota_before.json

echo "== push F2: train session 2 (folds 3-4) =="
uv run python scripts/kx.py push kaggle/train --config experiments/S6/session-2-config.json
cp runs/kx/kernels/rsna-knee-train.json experiments/S6/session-2-kernel-publication.json

echo "== push F4: teacher-s6 full (fold0_train, 3,537 studies) =="
mkdir -p experiments/S5-T4S6
uv run python scripts/kx.py push kaggle/teacher-s6 --config kaggle/teacher-s6/config.json
cp runs/kx/kernels/rsna-knee-teacher-s6.json experiments/S5-T4S6/kernel-publication.json

echo "== watch starts (25 min each, parallel) =="
uv run python scripts/watch_start.py wangshu/rsna-knee-train --minutes 25 &
w1=$!
uv run python scripts/watch_start.py wangshu/rsna-knee-teacher-s6 --minutes 25 &
w2=$!
wait $w1; r1=$?
wait $w2; r2=$?
echo "watch train      -> exit $r1 (0=started 2=hang:stop+repush 3=already ended)"
echo "watch teacher-s6 -> exit $r2 (0=started 2=hang:stop+repush 3=already ended)"
[ "$r1" -eq 0 ] && [ "$r2" -eq 0 ]
