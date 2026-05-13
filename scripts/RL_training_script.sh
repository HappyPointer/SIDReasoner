#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=sid-stage3-rl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=slurm_output/%x-%j.out

# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.
# export NCCL_P2P_DISABLE=1       # 禁用 NVLink
# export NCCL_IB_DISABLE=1        # 禁用 InfiniBand
# export NCCL_NET_GDR_LEVEL=0     # 禁用 GDR（GPU直连）
# export CUDA_VISIBLE_DEVICES=4,5,6,7

set -euo pipefail
set -x

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]]; then
    SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"

source ./scripts/snellius_env.sh

# ================================
# Note: please change the number of GPUs and nodes according to your setup.
# ================================
n_gpus_per_node="${N_GPUS_PER_NODE:-4}"
nnodes="${NNODES:-1}"
experiment_name="${EXPERIMENT_NAME:-Office_Products_stage3_rl_Qwen3-1.7B}"
stage2_checkpoint="${STAGE2_CHECKPOINT:-./output_dir/Office_Products_stage2_reasoning_activation_Qwen3-1.7B/final_checkpoint}"
log_file="${LOG_FILE:-./logs/${experiment_name}.log}"
# ================================

mkdir -p ./logs

{
${PYTHON_CMD} -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./data/Amazon/rec_reasoning_verl/Office_Products/train.parquet \
    data.val_files=./data/Amazon/rec_reasoning_verl/Office_Products/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${stage2_checkpoint}" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    custom_reward_function.path="./verl/utils/reward_score/direct_recommendation_StepRule_Office.py" \
    custom_reward_function.name="rule_base_reward" \
    trainer.project_name='RecRL_Reasoning' \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 "$@"
} > "${log_file}" 2>&1
