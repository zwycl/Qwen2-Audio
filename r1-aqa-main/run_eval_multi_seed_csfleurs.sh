#!/bin/bash

# ============================================================================
# Multi-Seed Evaluation on CS-FLEURS
# ============================================================================
#
# Evaluates multiple seed runs on CS-FLEURS, reports per-run metrics,
# and computes average CER/SHR with variance across seeds.
#
# Usage:
#   ./run_eval_multi_seed_csfleurs.sh <base_name> <seeds> [num_examples] [num_gpus] [subset] [skip_examples] [two_step] [use_vad]
#
# Examples:
#   ./run_eval_multi_seed_csfleurs.sh csfleurs_xtts_train_cer_n2310_e8_novad 42,67,68
#   ./run_eval_multi_seed_csfleurs.sh csfleurs_xtts_train_cer_n2310_e8_novad 42,67,68 999999 8 read_test 0
#
# Arguments:
#   $1 - Base run name (without _s<seed> suffix)
#   $2 - Comma-separated list of seeds (e.g., 42,67,68)
#   $3 - Number of examples to evaluate (default: 999999 = all)
#   $4 - Number of GPUs (default: 8)
#   $5 - Dataset subset: read_test, xtts_train, xtts_test1, xtts_test2, mms_test (default: read_test)
#   $6 - Number of training examples to skip (default: 0)
#   $7 - Two-step evaluation: true/false (default: false)
#   $8 - Use VAD chunking: true/false (default: true)
#
# Output:
#   - Per-run JSON results in ./outputs/eval_results/
#   - Aggregated summary JSON with mean/variance across seeds
# ============================================================================

set -e
cd /home/ubuntu/Qwen2-Audio/r1-aqa-main

# Parse arguments
BASE_NAME="${1:?Usage: $0 <base_name> <seeds> [num_examples] [num_gpus] [subset] [skip_examples] [two_step] [use_vad]}"
SEEDS="${2:?Usage: $0 <base_name> <seeds> [num_examples] [num_gpus] [subset] [skip_examples] [two_step] [use_vad]}"
NUM_EXAMPLES="${3:-999999}"
NUM_GPUS="${4:-8}"
SUBSET="${5:-read_test}"
SKIP_EXAMPLES="${6:-0}"
TWO_STEP="${7:-false}"
USE_VAD="${8:-true}"

DATA_DIR="/home/ubuntu/Qwen2-Audio/csfleurs_data"
OUTPUT_DIR="./outputs/eval_results"
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Convert seeds to array
IFS=',' read -ra SEED_ARRAY <<< "${SEEDS}"
NUM_SEEDS=${#SEED_ARRAY[@]}

echo "============================================================"
echo "Multi-Seed CS-FLEURS Evaluation"
echo "============================================================"
echo "Base name:   ${BASE_NAME}"
echo "Seeds:       ${SEEDS} (${NUM_SEEDS} runs)"
echo "Subset:      ${SUBSET}"
echo "Examples:    ${NUM_EXAMPLES}"
echo "Skip:        ${SKIP_EXAMPLES}"
echo "GPUs:        ${NUM_GPUS}"
echo "VAD chunk:   ${USE_VAD}"
echo "Two-step:    ${TWO_STEP}"
echo "============================================================"

# Build optional arguments
TWO_STEP_ARGS=""
if [ "${TWO_STEP}" = "true" ]; then
    TWO_STEP_ARGS="--two_step"
fi

# Collect output files for aggregation
OUTPUT_FILES=()

# Evaluate each seed
for SEED in "${SEED_ARRAY[@]}"; do
    RUN_DIR="./outputs/${BASE_NAME}_s${SEED}"

    if [ ! -d "${RUN_DIR}" ]; then
        echo "WARNING: Run directory not found: ${RUN_DIR}, skipping seed ${SEED}"
        continue
    fi

    OUTPUT_FILE="${OUTPUT_DIR}/eval_csfleurs_${BASE_NAME}_s${SEED}_${SUBSET}_n${NUM_EXAMPLES}_${TIMESTAMP}.json"
    OUTPUT_FILES+=("${OUTPUT_FILE}")

    echo ""
    echo "============================================================"
    echo "Evaluating seed ${SEED}: ${RUN_DIR} on ${SUBSET}"
    echo "============================================================"

    if [ "${NUM_GPUS}" -gt 1 ]; then
        torchrun --nproc_per_node=${NUM_GPUS} \
            --master_port=29501 \
            src/evaluate_csfleurs.py \
            --model_name_or_path "${RUN_DIR}" \
            --data_dir "${DATA_DIR}" \
            --subset "${SUBSET}" \
            --skip_examples "${SKIP_EXAMPLES}" \
            --num_examples "${NUM_EXAMPLES}" \
            --output_file "${OUTPUT_FILE}" \
            --use_vad_chunking "${USE_VAD}" \
            ${TWO_STEP_ARGS}
    else
        python src/evaluate_csfleurs.py \
            --model_name_or_path "${RUN_DIR}" \
            --data_dir "${DATA_DIR}" \
            --subset "${SUBSET}" \
            --skip_examples "${SKIP_EXAMPLES}" \
            --num_examples "${NUM_EXAMPLES}" \
            --output_file "${OUTPUT_FILE}" \
            --use_vad_chunking "${USE_VAD}" \
            --verbose \
            ${TWO_STEP_ARGS}
    fi

    echo "Results saved to: ${OUTPUT_FILE}"
done

# Aggregate results across seeds
echo ""
echo "============================================================"
echo "Aggregating results across ${NUM_SEEDS} seeds..."
echo "============================================================"

AGGREGATE_FILE="${OUTPUT_DIR}/eval_csfleurs_${BASE_NAME}_${SUBSET}_multi_seed_${TIMESTAMP}.json"

python3 -c "
import json
import sys
from pathlib import Path
from collections import defaultdict

output_files = '${OUTPUT_FILES[*]}'.split()
if not output_files or output_files == ['']:
    print('No output files to aggregate')
    sys.exit(1)

all_runs = []
for f in output_files:
    p = Path(f)
    if not p.exists():
        print(f'WARNING: {f} not found, skipping')
        continue
    with open(p) as fh:
        all_runs.append(json.load(fh))

n = len(all_runs)
if n == 0:
    print('No results to aggregate')
    sys.exit(1)

# Overall CER and SHR per run
run_cers = [r['avg_cer'] for r in all_runs]
run_shrs = [r['avg_script_hall_rate'] for r in all_runs]

mean_cer = sum(run_cers) / n
mean_shr = sum(run_shrs) / n
var_cer = sum((x - mean_cer) ** 2 for x in run_cers) / n if n > 1 else 0
var_shr = sum((x - mean_shr) ** 2 for x in run_shrs) / n if n > 1 else 0
std_cer = var_cer ** 0.5
std_shr = var_shr ** 0.5

# Per-language aggregation
lang_cers = defaultdict(list)
lang_shrs = defaultdict(list)
for r in all_runs:
    for lang, stats in r.get('per_language', {}).items():
        lang_cers[lang].append(stats['avg_cer'])
        lang_shrs[lang].append(stats['avg_script_hall_rate'])

# Print summary
print()
print('=' * 70)
print(f'MULTI-SEED CS-FLEURS (${SUBSET}) RESULTS ({n} seeds)')
print('=' * 70)
print(f'Overall CER:  {mean_cer:.4f} +/- {std_cer:.4f}  (var={var_cer:.6f})')
print(f'Overall SHR:  {mean_shr:.4f} +/- {std_shr:.4f}  (var={var_shr:.6f})')
print('-' * 70)
print(f'{\"Seed\":<8} {\"CER\":>8} {\"SHR\":>8}')
print('-' * 70)
for i, r in enumerate(all_runs):
    import re as _re
    _m = _re.search(r'_s(\d+)', output_files[i])
    seed = _m.group(1) if _m else f'run{i}'
    print(f's{seed:<7} {r[\"avg_cer\"]:>8.4f} {r[\"avg_script_hall_rate\"]:>8.4f}')
print('-' * 70)
print(f'{\"Mean\":<8} {mean_cer:>8.4f} {mean_shr:>8.4f}')
print(f'{\"Std\":<8} {std_cer:>8.4f} {std_shr:>8.4f}')

# Per-language breakdown
if lang_cers:
    print()
    print('-' * 70)
    print(f'{\"Language\":<12} {\"Mean CER\":>10} {\"Std CER\":>10} {\"Mean SHR\":>10} {\"Std SHR\":>10}')
    print('-' * 70)
    for lang in sorted(lang_cers.keys()):
        lc = lang_cers[lang]
        ls = lang_shrs[lang]
        lc_mean = sum(lc) / len(lc)
        ls_mean = sum(ls) / len(ls)
        lc_std = (sum((x - lc_mean)**2 for x in lc) / len(lc)) ** 0.5 if len(lc) > 1 else 0
        ls_std = (sum((x - ls_mean)**2 for x in ls) / len(ls)) ** 0.5 if len(ls) > 1 else 0
        print(f'{lang:<12} {lc_mean:>10.4f} {lc_std:>10.4f} {ls_mean:>10.4f} {ls_std:>10.4f}')
print('=' * 70)

# Save aggregate JSON
aggregate = {
    'base_name': '${BASE_NAME}',
    'subset': '${SUBSET}',
    'seeds': [int(s) for s in '${SEEDS}'.split(',')],
    'num_seeds': n,
    'num_examples': ${NUM_EXAMPLES},
    'skip_examples': ${SKIP_EXAMPLES},
    'overall': {
        'mean_cer': mean_cer,
        'std_cer': std_cer,
        'var_cer': var_cer,
        'mean_shr': mean_shr,
        'std_shr': std_shr,
        'var_shr': var_shr,
    },
    'per_seed': [
        {
            'seed': (_re.search(r'_s(\d+)', output_files[i]).group(1) if _re.search(r'_s(\d+)', output_files[i]) else f'run{i}'),
            'avg_cer': r['avg_cer'],
            'avg_shr': r['avg_script_hall_rate'],
            'output_file': output_files[i],
        }
        for i, r in enumerate(all_runs)
    ],
    'per_language': {
        lang: {
            'mean_cer': sum(lang_cers[lang]) / len(lang_cers[lang]),
            'std_cer': (sum((x - sum(lang_cers[lang])/len(lang_cers[lang]))**2 for x in lang_cers[lang]) / len(lang_cers[lang])) ** 0.5 if len(lang_cers[lang]) > 1 else 0,
            'mean_shr': sum(lang_shrs[lang]) / len(lang_shrs[lang]),
            'std_shr': (sum((x - sum(lang_shrs[lang])/len(lang_shrs[lang]))**2 for x in lang_shrs[lang]) / len(lang_shrs[lang])) ** 0.5 if len(lang_shrs[lang]) > 1 else 0,
        }
        for lang in sorted(lang_cers.keys())
    },
}

with open('${AGGREGATE_FILE}', 'w') as f:
    json.dump(aggregate, f, indent=2, ensure_ascii=False)
print(f'\nAggregate results saved to: ${AGGREGATE_FILE}')
"

echo ""
echo "Done!"
