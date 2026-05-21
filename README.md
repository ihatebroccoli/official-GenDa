# GenDa

Official implementation for **GenDa: Learning Generalizable Skill Policy with Data-Efficient Unsupervised RL**.

<p align="center">
  <a href="TODO_PAPER_LINK"><strong>Paper</strong></a> ·
  <a href="https://ihatebroccoli.github.io/official-GenDa/"><strong>Project Page</strong></a> ·
  <a href="TODO_ARXIV_LINK"><strong>arXiv</strong></a>
</p>

## Overview

GenDa is an unsupervised reinforcement learning framework for learning **data-efficient and generalizable skill-conditioned policies**.  
The method addresses two practical issues in off-policy skill discovery: stale skill semantics in replay buffers and brittle skill generalization under distribution shifts.

This repository provides code for:

- Unsupervised skill pretraining
- Downstream evaluation with frozen skill policies
- State-based and pixel-based benchmark environments

## Installation

### 1. Clone this repository

```bash
https://github.com/ihatebroccoli/official-GenDa.git
```

### 2. Create a conda environment

```bash
conda create -n genda python=3.10 -y
conda activate genda
```

### 3. Install dependencies
❗Important: sudo apt prompts are contained in install.sh.
```bash
cd official_GenDa
bash install.sh
```

### 4. Environment dependencies

For headless rendering, one of the following may be needed:

```bash
export MUJOCO_GL=egl
```

## Repository Structure

```text
.
├── agent/                  # GenDa, high-level controller, and network modules
├── envs/                   # Environment wrappers and task definitions
├── scripts/                # Reproduction scripts
├── utils/                  # Replay, logging and utility functions
├── main.py                 # Pretraining entry point
├── run.py                  # Pretraining core
├── downstream_task.py      # Downstream task implementation
├── requirements.txt
└── README.md
```

## Quick Start
### Skill pretraining
```bash
./scripts/pretrain/humanoid_numeric.sh 0 0 debug
```

### Downstream evaluation

```bash
./scripts/downstream/humanoid_maze_numeric.sh 0 0 debug 50000 exp/your_path
```

### Results differ from the paper

RL experiments can vary depending on hardware, CUDA version, simulator version, and random seeds.  
We recommend running multiple seeds and reporting mean and standard deviation.

## Citation
TODO

## License

This repository is released under the `TODO` license.

Please see [`LICENSE`](LICENSE) for details.

## Contact

For questions, please open a GitHub issue or contact:

```text
Jongchan Park <marin0625@skku.edu>
```
