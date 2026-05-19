# GENDA

Official implementation for **GENDA: Learning Generalizable Skill Policy with Data-Efficient Unsupervised RL**.

<p align="center">
  <a href="TODO_PAPER_LINK"><strong>Paper</strong></a> ·
  <a href="TODO_PROJECT_PAGE_LINK"><strong>Project Page</strong></a> ·
  <a href="TODO_ARXIV_LINK"><strong>arXiv</strong></a>
</p>

## Overview

GENDA is an unsupervised reinforcement learning framework for learning **data-efficient and generalizable skill-conditioned policies**.  
The method addresses two practical issues in off-policy skill discovery: stale skill semantics in replay buffers and brittle skill generalization under distribution shifts.

This repository provides code for:

- Unsupervised skill pretraining
- Downstream evaluation with frozen skill policies
- State-based and pixel-based benchmark environments
- Ablation studies
- Checkpoint evaluation and visualization

## News

- `[YYYY-MM-DD]` Initial code release.
- `[YYYY-MM-DD]` Pretrained checkpoints released.
- `[YYYY-MM-DD]` Project page released.

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/TODO_ORG/TODO_REPO.git
cd TODO_REPO
```

### 2. Create a conda environment

```bash
conda create -n genda python=3.10 -y
conda activate genda
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For editable installation:

```bash
pip install -e .
```

### 4. Environment dependencies

Depending on the benchmark, install MuJoCo and DeepMind Control Suite dependencies.

```bash
pip install mujoco dm-control
```

For headless rendering, one of the following may be needed:

```bash
export MUJOCO_GL=egl
# or
export MUJOCO_GL=osmesa
```

## Repository Structure

```text
.
├── configs/                 # Experiment configuration files
│   ├── pretrain/            # Unsupervised skill pretraining configs
│   ├── downstream/          # Downstream task configs
│   └── eval/                # Evaluation configs
├── envs/                    # Environment wrappers and task definitions
├── genda/                   # Main implementation
│   ├── agents/              # SAC agents and skill-conditioned policies
│   ├── algorithms/          # GENDA training logic
│   ├── buffers/             # Replay buffers
│   ├── cib/                 # Complementary Information Bottleneck modules
│   ├── networks/            # Policy, critic, encoder, and representation networks
│   ├── relabeling/          # Skill relabeling utilities
│   └── utils/               # Logging, evaluation, seeding, checkpointing
├── scripts/                 # Reproduction scripts
├── tools/                   # Plotting and result aggregation utilities
├── pretrained/              # Pretrained checkpoints, if released
├── outputs/                 # Default output directory
├── train.py                 # Training entry point
├── eval.py                  # Evaluation entry point
├── requirements.txt
└── README.md
```

## Quick Start

### Pretrain a skill policy

```bash
python train.py \
  --config configs/pretrain/humanoid_numeric.yaml \
  --seed 0
```

### Evaluate skill coverage

```bash
python eval.py \
  --config configs/eval/humanoid_numeric.yaml \
  --checkpoint outputs/humanoid_numeric/seed0/checkpoints/latest.pt
```

### Train a downstream controller

```bash
python train.py \
  --config configs/downstream/humanoid_rs_rg.yaml \
  --skill_checkpoint outputs/humanoid_numeric/seed0/checkpoints/latest.pt \
  --seed 0
```

## Reproducing Experiments

### Skill pretraining

```bash
bash scripts/pretrain_humanoid_numeric.sh
bash scripts/pretrain_quadruped_numeric.sh
bash scripts/pretrain_dog_numeric.sh
bash scripts/pretrain_fish_numeric.sh
bash scripts/pretrain_humanoid_pixels.sh
bash scripts/pretrain_quadruped_pixels.sh
```

### Downstream evaluation

```bash
bash scripts/downstream_humanoid.sh
bash scripts/downstream_quadruped.sh
bash scripts/downstream_dog.sh
bash scripts/downstream_fish.sh
```

### Ablations

```bash
bash scripts/ablation_without_relabeling.sh
bash scripts/ablation_without_cib.sh
bash scripts/ablation_beta.sh
bash scripts/ablation_utd.sh
```

### Plot results

```bash
python tools/plot_results.py \
  --logdir outputs/ \
  --outdir figures/
```

## Configuration

Experiments are controlled by YAML config files.

Example:

```yaml
seed: 0
run_name: humanoid_numeric_genda_seed0

experiment:
  mode: pretrain
  total_steps: 10000000
  eval_interval: 50000
  save_interval: 500000

env:
  name: humanoid_numeric
  obs_type: state

agent:
  name: genda
  skill_dim: 2
  batch_size: 1024
  discount: 0.99
  utd_ratio: 0.5

relabeling:
  enabled: true
  use_episode_relabeling: true
  use_c_step_relabeling: true
  ema_tau: 0.995

cib:
  enabled: true
  latent_dim: 64

logging:
  output_dir: outputs/
  backend: wandb
```

Common overrides:

```bash
python train.py --config configs/pretrain/humanoid_numeric.yaml --seed 1
python train.py --config configs/pretrain/humanoid_numeric.yaml agent.utd_ratio=0.25
python train.py --config configs/pretrain/humanoid_numeric.yaml relabeling.enabled=false
python train.py --config configs/pretrain/humanoid_numeric.yaml cib.enabled=false
```

## Checkpoints

By default, checkpoints are saved to:

```text
outputs/<run_name>/checkpoints/
```

Example:

```text
outputs/humanoid_numeric_genda_seed0/
├── checkpoints/
│   ├── latest.pt
│   └── step_1000000.pt
├── logs/
├── videos/
└── metrics.jsonl
```

Resume training:

```bash
python train.py \
  --config configs/pretrain/humanoid_numeric.yaml \
  --resume outputs/humanoid_numeric_genda_seed0/checkpoints/latest.pt
```

Evaluate a checkpoint:

```bash
python eval.py \
  --config configs/eval/humanoid_numeric.yaml \
  --checkpoint outputs/humanoid_numeric_genda_seed0/checkpoints/latest.pt
```

## Pretrained Models

Pretrained models will be released under `pretrained/`.

| Environment | Checkpoint | Status |
|---|---|---|
| Humanoid-Numeric | TODO | TODO |
| Quadruped-Numeric | TODO | TODO |
| Dog-Numeric | TODO | TODO |
| Fish-Numeric | TODO | TODO |
| Humanoid-Pixels | TODO | TODO |
| Quadruped-Pixels | TODO | TODO |

Download all released checkpoints:

```bash
bash scripts/download_checkpoints.sh
```

## Evaluation Metrics

This repository supports the following evaluation metrics:

| Metric | Description |
|---|---|
| `state_coverage` | Number of unique coordinate bins visited by sampled skills |
| `task_coverage` | Number of unique tasks reached in task-based environments |
| `success_rate` | Downstream goal-reaching success rate |
| `episode_return` | Episodic return, when applicable |
| `video` | Rollout visualization of sampled skills |

## Logging

Supported logging backends:

- TensorBoard
- Weights & Biases
- JSONL / CSV logs

Launch TensorBoard:

```bash
tensorboard --logdir outputs/
```

## Troubleshooting

### MuJoCo rendering error

Try:

```bash
export MUJOCO_GL=egl
```

or:

```bash
export MUJOCO_GL=osmesa
```

### CUDA out of memory

Reduce the batch size:

```yaml
agent:
  batch_size: 512
```

You can also disable video logging during training.

### Results differ from the paper

RL experiments can vary depending on hardware, CUDA version, simulator version, and random seeds.  
We recommend running multiple seeds and reporting mean and standard deviation.

## Citation

If you use this codebase, please cite:

```bibtex
@inproceedings{park2026genda,
  title     = {Learning Generalizable Skill Policy with Data-Efficient Unsupervised RL},
  author    = {Park, Jongchan and Oh, Seungjun and Baek, Seungho and Kim, Yusung},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

## License

This repository is released under the `TODO` license.

Please see [`LICENSE`](LICENSE) for details.

## Contact

For questions, please open a GitHub issue or contact:

```text
Yusung Kim <yskim525@skku.edu>
```
