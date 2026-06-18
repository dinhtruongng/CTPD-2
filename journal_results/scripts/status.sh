#!/bin/bash
set -euo pipefail
POS=$(grep -c loss /home/ubuntu/CTPD-2/src/prefkd/pairB_tea_positive/rank_0_stdout.log 2>/dev/null || echo 0)
NEG=$(grep -c loss /home/ubuntu/CTPD-2/src/prefkd/pairB_tea_negative/rank_0_stdout.log 2>/dev/null || echo 0)
echo "=== Phase 1a Status ==="
echo "tea_pos: ${POS}/3541 updates ($(python3 -c "print(f'{${POS}/3541*100:.1f}%')"))"
echo "tea_neg: ${NEG}/14164 updates ($(python3 -c "print(f'{${NEG}/14164*100:.1f}%')"))"
echo ""
echo "--- tea_pos latest ---"
grep loss /home/ubuntu/CTPD-2/src/prefkd/pairB_tea_positive/rank_0_stdout.log 2>/dev/null | tail -1 | cut -c1-300
echo ""
echo "--- tea_neg latest ---"
grep loss /home/ubuntu/CTPD-2/src/prefkd/pairB_tea_negative/rank_0_stdout.log 2>/dev/null | tail -1 | cut -c1-300
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | sed -n 5,8p
echo ""
echo "=== Checkpoints ==="
find /home/ubuntu/CTPD-2/src/prefkd/output -name "checkpoint-*" -type d -newer /tmp -mmin -120 2>/dev/null | sort
echo ""
echo "=== ETA ==="
python3 -c "
pos=$POS; neg=$NEG
if pos>0:
    rate=7.5 # ~updates/min
    min_p=(3541-pos)/rate
    print(f'tea_pos: ~{int(min_p)} min')
if neg>0:
    rate=(neg/((neg*8/6.5)/60))  # examples/sec ~6.5
    min_n=(14164-neg)/7.5
    print(f'tea_neg: ~{int(min_n)} min')
"
