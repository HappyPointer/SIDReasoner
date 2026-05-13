#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=sid-merge-fsdp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output/%x-%j.out

# === Usage ===
# bash merge_verl_ckpt.sh /path/to/verl/checkpoint/actor
#
# Example:
# bash merge_verl_ckpt.sh ./RecRL_with_Reasoning/Qwen3-1.7B_Mix2-50K_Games/global_step_10/actor

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"

source ./scripts/snellius_env.sh

CKPT_DIR="${1:-${CKPT_DIR:-./checkpoints/RecRL_Reasoning/Office_Products_stage3_rl_Qwen3-1.7B/global_step_100/actor}}"

if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: Please provide a verl checkpoint directory."
    echo "Usage: bash merge_verl_ckpt.sh /path/to/actor"
    exit 1
fi

# Remove trailing slash if exists
CKPT_DIR="${CKPT_DIR%/}"

# Output directory
MERGED_DIR="${CKPT_DIR}_merged"

echo "Verl checkpoint directory: $CKPT_DIR"
echo "Will save merged HF model to: $MERGED_DIR"
echo ""


MERGE_PY="./scripts/merge_fsdp_checkpoint.py"

if [ ! -f "$MERGE_PY" ]; then
    echo "ERROR: Cannot find merge_fsdp_ckpt.py in Verl installation."
    echo "Expected at: $MERGE_PY"
    exit 1
fi

echo "Using merge script: $MERGE_PY"
echo ""

# Run merge
${PYTHON_CMD} "$MERGE_PY" \
    --checkpoint "$CKPT_DIR" \
    --output-dir "$MERGED_DIR"

echo ""
echo "Merge completed."
echo "Merged HuggingFace model is saved to:"
echo "   $MERGED_DIR"
echo ""
echo "You can load it with:"
echo "   from transformers import AutoModelForCausalLM"
echo "   model = AutoModelForCausalLM.from_pretrained('$MERGED_DIR')"
