#!/bin/bash
# 3-seed robustness batch: retrain multi-task + single-task on the deduplicated split at
# seeds 1 and 2 (seed 42 already done), evaluate each, and collect OVERALL grade metrics.
cd /Users/macbook/UCP/FYP/farm2fork-ai || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1
RES=outputs/seed_batch_results.txt
echo "seed batch started $(date '+%Y-%m-%d %H:%M:%S %Z')" > "$RES"

run() {  # $1=config  $2=tag  $3=seed
  local cfg="$1" tag="$2" seed="$3"
  local ckpt="models/checkpoints/${tag}_s${seed}"
  echo ">>> train ${tag} seed ${seed} @ $(date '+%H:%M:%S')" >> "$RES"
  .venv/bin/python -u scripts/train.py --config "$cfg" --seed "$seed" \
      --checkpoint-dir "$ckpt" > "outputs/seedbatch_${tag}_s${seed}.log" 2>&1
  local line
  line=$(.venv/bin/python scripts/eval_per_crop.py --config "$cfg" \
      --checkpoint "${ckpt}/best_model.pth" 2>/dev/null | grep OVERALL)
  echo "RESULT ${tag} seed=${seed}  ${line}" >> "$RES"
  echo "RESULT ${tag} seed=${seed}  ${line}"
}

run configs/four_crops_dedup_15ep.yaml            mt_dedup 1
run configs/four_crops_dedup_15ep.yaml            mt_dedup 2
run configs/grade_only_four_crops_dedup_15ep.yaml st_dedup 1
run configs/grade_only_four_crops_dedup_15ep.yaml st_dedup 2
echo "seed batch done $(date '+%H:%M:%S %Z')" >> "$RES"
