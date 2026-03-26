# SIDReasoner

This is the code implementation for **"SIDReasoner - Reasoning over Semantic IDs Enhances Generative Recommendation"**.

SIDReasoner is a generative recommendation framework that strengthens generative recommenders with reasoning ability over semantic IDs. This repository provides:

- A complete training pipeline, with each training stage integrated into an easy-to-run script.
- Full training data, including our synthesized enriched alignment corpus.
- Pretrained model checkpoints.

Our method demonstrates that, with improved SID–language alignment, effective recommendation reasoning can be achieved even under academic-scale training. SIDReasoner is able to associate SIDs with their underlying item semantics, produce coherent natural-language reasoning over interaction histories, and generate recommendations according to the reasoning process. By open-sourcing the pipeline, data, and checkpoints, we aim to facilitate further research on reasoning in generative recommendation.

<p align="center">
  <img src="assets/SIDReasoner-CaseStudy.png" width="100%" alt="Training">
</p>

## Environments

The reinforcement learning stage (Stage 3) in this project is built on top of VERL. We recommend follow the [official installation guide](https://verl.readthedocs.io/en/latest/start/install.html#requirements) to set up the environment. To execute the codes correctly, the following additional packages are required:

- `torch`
- `transformers`
- `datasets`
- `peft`
- `pandas`
- `numpy`
- `fire`
- `wandb`
- `tqdm`
- `accelerate`
- `bitsandbytes`


## Dataset

The datasets can be accessed via this [link](https://drive.google.com/file/d/1S92QvaEs7aAhCwWOUovyb4y5P-pXwUhH/view?usp=sharing). Please download the dataset and ensure the dataset folder is placed under directory ./data/Amazon .

## Training

SIDReasoner follows a three-stage training pipeline.

| Stage | Script | 
| --- | --- | 
| Stage 1: Supervised Fine-Tuning | `bash sft_Qwen3_enrich.sh` | 
| Stage 2: Reasoning Activation | `bash sft_reasoning_activation.sh` |
| Stage 3: RL Training | `bash RL_training_script.sh` |

### Run training

```bash
# Stage 1
bash sft_Qwen3_enrich.sh

# Stage 2
bash sft_reasoning_activation.sh

# Stage 3
bash RL_training_script.sh
```

The training logs are written to `./logs`.


### Checkpoints

To facilitate further research, we release our pretrained model checkpoints, which can be downloaded via this [link](https://huggingface.co/Sober-Clever/SIDReasoner-Models/tree/main).

## Evaluation

We provide the scripts to test the model performance under thinking and non-thinking mode:

```bash
# Non-thinking mode.
bash evaluate_Qwen3.sh

# Thinking mode.
bash evaluate_Qwen3_think.sh
```

### Stage 3 checkpoint merge

The reasoning evaluation script expects a merged Hugging Face checkpoint named `actor_merged`. If RL training has only produced raw `actor` folders, merge them first:

```bash
python3 ./scripts/merge_fsdp_checkpoint.py \
  --checkpoint ./checkpoints/RecRL_Reasoning/Office_Products_stage3_rl_Qwen3-1.7B/global_step_100/actor \
  --output-dir ./checkpoints/RecRL_Reasoning/Office_Products_stage3_rl_Qwen3-1.7B/global_step_100/actor_merged
```


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
