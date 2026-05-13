# SIDReasoner

This repository contains the code for **"SIDReasoner - Reasoning over Semantic IDs Enhances Generative Recommendation"**.

SIDReasoner is a generative recommendation framework that strengthens recommendation models with reasoning over semantic IDs. The repository provides the training scripts, evaluation scripts, data download workflow, and a Snellius-ready environment setup.

## Snellius Quick Start

Run these commands from a clean checkout.

```bash
git clone <repo-url>
cd SIDReasoner
```

Create the Python environment on a compute node:

```bash
sbatch scripts/setup_uv_env.sh
```

Download and extract the dataset:

```bash
make data
```

Run the three training stages in order. Submit the next job only after the previous one has finished.

```bash
sbatch scripts/sft_Qwen3_enrich.sh
sbatch scripts/sft_reasoning_activation.sh
sbatch scripts/RL_training_script.sh
```

Merge the RL checkpoint before thinking-mode evaluation. Adjust `global_step_100` if you want a different checkpoint.

```bash
sbatch scripts/merge_fsdp_ckpt.sh ./checkpoints/RecRL_Reasoning/Office_Products_stage3_rl_Qwen3-1.7B/global_step_100/actor
```

Run evaluation:

```bash
sbatch scripts/evaluate_Qwen3.sh
sbatch scripts/evaluate_Qwen3_think.sh
```

Useful monitoring commands:

```bash
squeue -u $USER
tail -f slurm_output/*.out
tail -f logs/*.txt
tail -f logs/*.log
```

## Environment

This repository uses `uv` for dependency management. The Python dependencies and linting tools are declared in `pyproject.toml`.

The base runtime stack is pinned for CUDA 12.4, PyTorch 2.6, vLLM 0.8.5, FlashAttention 2.7.4, and FlashInfer 0.2.2. The cuDNN override from the original VERL setup is installed after uv resolves the base environment because Torch pins a different cuDNN package in its dependency metadata.

The Slurm scripts load Snellius modules through `scripts/snellius_env.sh`. The setup job uses the Snellius `2023` module stack with `CUDA/12.4.0` and creates `.venv` with Python 3.10 through uv.

Optional SGLang support:

```bash
sbatch scripts/setup_uv_env.sh --sglang
```

Optional Megatron/TransformerEngine support:

```bash
sbatch scripts/setup_uv_env.sh --megatron
```

Install both optional stacks:

```bash
sbatch scripts/setup_uv_env.sh --all
```

## Dataset

Download and extract the dataset with:

```bash
make data
```

This downloads the Google Drive dataset and places it under:

```text
data/Amazon
```

To use a different Google Drive file ID or archive name:

```bash
make data DATA_FILE_ID="..." DATA_ARCHIVE="data/my_dataset.zip"
```

## Training

SIDReasoner follows a three-stage training pipeline.

| Stage | Script |
| --- | --- |
| Stage 1: Supervised Fine-Tuning | `scripts/sft_Qwen3_enrich.sh` |
| Stage 2: Reasoning Activation | `scripts/sft_reasoning_activation.sh` |
| Stage 3: RL Training | `scripts/RL_training_script.sh` |

Run on Snellius:

```bash
sbatch scripts/sft_Qwen3_enrich.sh
sbatch scripts/sft_reasoning_activation.sh
sbatch scripts/RL_training_script.sh
```

The scripts write Slurm output to `slurm_output/` and training logs to `logs/`.

Common overrides:

```bash
CATEGORY=Office_Products CUDA_DEVICES=0,1,2,3 NPROC_PER_NODE=4 sbatch scripts/sft_Qwen3_enrich.sh
N_GPUS_PER_NODE=4 NNODES=1 sbatch scripts/RL_training_script.sh
```

## Evaluation

Evaluate non-thinking and thinking modes:

```bash
sbatch scripts/evaluate_Qwen3.sh
sbatch scripts/evaluate_Qwen3_think.sh
```

Override GPU splits:

```bash
CUDA_LIST="0 1" CUDA_LIST_CSV="0,1" sbatch scripts/evaluate_Qwen3_think.sh
```

The thinking-mode evaluation expects a merged Hugging Face checkpoint named `actor_merged`. If RL training only produced raw `actor` folders, merge one first:

```bash
sbatch scripts/merge_fsdp_ckpt.sh ./checkpoints/RecRL_Reasoning/Office_Products_stage3_rl_Qwen3-1.7B/global_step_100/actor
```

## Development

Useful Make targets:

```bash
make data        # download and extract the dataset
make lint        # run ruff checks
make format      # format Python files with ruff
make precommit   # run all pre-commit hooks
```

Install development hooks:

```bash
make install-dev
```

## Checkpoints

Pretrained model checkpoints are available on Hugging Face:

https://huggingface.co/Sober-Clever/SIDReasoner-Models/tree/main

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{SIDReasoner,
  title={Reasoning over Semantic IDs Enhances Generative Recommendation},
  author={Yingzhi He and Yan Sun and Junfei Tan and Yuxin Chen and Xiaoyu Kong and Chunxu Shen and Xiang Wang and An Zhang and Tat-Seng Chua},
  journal={arXiv preprint arXiv:2603.23183},
  year={2026}
}
```

## Acknowledgement

This repo is built upon [MiniOneRec](https://github.com/AkaliKong/MiniOneRec).
