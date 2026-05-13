#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=sid-stage1-sft
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=slurm_output/%x-%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"

source ./scripts/snellius_env.sh

CATEGORY="${CATEGORY:-Office_Products}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
TRAIN_FILE="${TRAIN_FILE:-./data/Amazon/train/Office_Products_5_2016-10-2018-11.csv}"
EVAL_FILE="${EVAL_FILE:-./data/Amazon/valid/Office_Products_5_2016-10-2018-11.csv}"
TEST_FILE="${TEST_FILE:-./data/Amazon/test/Office_Products_5_2016-10-2018-11.csv}"
INFO_FILE="${INFO_FILE:-./data/Amazon/info/Office_Products_5_2016-10-2018-11.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_dir/Office_Products_stage1_sft_Qwen3-1.7B}"
RUN_NAME="${RUN_NAME:-Office_Products_stage1_sft_Qwen3-1.7B}"
LOG_FILE="${LOG_FILE:-./logs/${RUN_NAME}.txt}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-12340}"

mkdir -p ./logs ./output_dir

{
echo "${TRAIN_FILE} ${EVAL_FILE} ${INFO_FILE} ${TEST_FILE}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" ${TORCHRUN_CMD} --nproc_per_node "${NPROC_PER_NODE}" \
    --master_port "${MASTER_PORT}" \
    sft_Qwen3.py \
    --base_model "${BASE_MODEL}" \
    --batch_size 1024 \
    --micro_batch_size 1 \
    --train_file "${TRAIN_FILE}" \
    --eval_file "${EVAL_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --wandb_project MiniOneRec \
    --wandb_run_name "${RUN_NAME}" \
    --category "${CATEGORY}" \
    --train_from_scratch False \
    --seed 42 \
    --sid_index_path "./data/Amazon/index/${CATEGORY}.index.json" \
    --item_meta_path "./data/Amazon/index/${CATEGORY}.item.json" \
    --llm_generated_data_path "./data/Amazon/index/${CATEGORY}.item_enhanced_v2.json" \
    --llm_generated_sequence_path "./data/Amazon/index/${CATEGORY}.integrated_narrative.csv" \
    --general_reasoning_path "./data/Amazon/general/sampled_data.arrow" \
    --mask_assistant True \
    "$@"
} > "${LOG_FILE}" 2>&1
