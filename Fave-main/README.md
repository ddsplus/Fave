# 🚀 FAVE: Flow-based Average Velocity Establishment for Sequential Recommendation

This repository contains the official implementation of the paper "FAVE: Flow-based Average Velocity Establishment for Sequential Recommendation" (SIGIR 2026).



[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)

**FAVE** is a **one-step generative framework for sequential recommendation** based on **flow matching**.  
It addresses the efficiency limitations of diffusion-based methods (e.g., **DreamRec**) and multi-step flow-based methods (e.g., **FMRec**) by replacing the conventional **noise-to-data iterative generation paradigm** with a **direct trajectory transport mechanism**.

At the core of FAVE is **Flow-based Average Velocity Establishment**, which learns a direct mapping from an informative semantic prior to the target user preference. This enables **order-of-magnitude faster inference** while maintaining strong recommendation accuracy.



## 🌟 Highlights

- ⚡ **One-step inference** for efficient sequential recommendation
- 🎯 **Informative semantic prior** instead of Gaussian noise
- 🧭 **Direct trajectory transport** via average velocity modeling
- 📉 **Order-of-magnitude speedup** without sacrificing performance
- 🧩 **Two-stage training strategy** for stable learning and effective consolidation



## 🏗️ Two-Stage Training Strategy

### Stage 1: Basic Manifold Construction

This stage learns a stable and expressive preference manifold by modeling instantaneous velocity fields.

- **Dual-Time Flow Modeling**  
  Introduces dual-time parameterization $(t, r)$ to better capture state positioning and refine instantaneous velocity estimation.

- **Dual-End Semantic Alignment**  
  Prevents representation collapse by enforcing constraints at both ends of the trajectory:
  - **Source-side alignment** via History Reconstruction loss $\mathcal{L}_{src}$
  - **Target-side alignment** via Item Prediction loss $\mathcal{L}_{tgt}$

- **Heavy-Tailed Sampling**  
  Uses mode sampling to emphasize high-curvature regions of the trajectory during training.

### Stage 2: Single-Step Consolidation

This stage compresses the learned trajectory into a robust one-step generation mechanism.

- **Semantic Anchor Prior**  
  Replaces Gaussian noise with a **masked history embedding**, providing a more informative starting point close to the target preference.

- **Average Velocity Modeling**  
  Predicts a global displacement vector that directly transports the prior to the target state, removing the need for multi-step Euler solving.

- **JVP Straightness Constraint**  
  Introduces a Jacobian-Vector Product (JVP) consistency loss $\mathcal{L}_{cons}$ to encourage straight trajectories and stabilize one-step inference.



## 📂 Repository Structure

```text
Fave/
├── best for amazon_beauty/          # Saved checkpoints for Amazon Beauty
│   ├── pretrain_best.pt             #   Stage 1 weights
│   └── pretrain_best_finetuned.pt   #   Stage 2 weights
├── best for ml-100k/                # Saved checkpoints for MovieLens-100K
│   ├── pretrain_best.pt
│   └── pretrain_best_finetuned.pt
├── best for steam/                  # Saved checkpoints for Steam
│   ├── pretrain_best.pt
│   └── pretrain_best_finetuned.pt
├── datasets/
│   ├── readme.md                    # Data preprocessing instructions
│   └── data/
│       ├── amazon_beauty/
│       │   └── dataset.pkl
│       ├── ml-100k/
│       │   └── dataset.pkl
│       └── steam/
│           └── dataset.pkl
├── src/
│   ├── auto_main.py                 # Entry point: argument parsing and training orchestration
│   ├── auto_trainer.py              # Stage 1 & Stage 2 training loops, metric computation
│   ├── model.py                     # Main model definition and loss computation
│   ├── fave.py                      # Core flow matching modules and samplers
│   ├── utils.py                     # Data loading, masking, and evaluation utilities
│   └── ckpt_inference.ipynb         # Notebook for checkpoint-based one-step inference
└── README.md


## 🗃️ Data Preparation

Dataset preprocessing follows [ICLRRec](https://github.com/salesforce/ICLRec/tree/master/data) and [DuoRec](https://github.com/RuihongQiu/DuoRec).

Each dataset should be placed as:

```bash
datasets/data/{dataset_name}/dataset.pkl
```

The pickle file should contain a dictionary with the following keys:

- `train`
- `val`
- `test`
- `smap` (item ID mapping)

### Supported datasets

- `ml-100k`
- `amazon_beauty`
- `steam`


## 📝 Citation

If you find this work useful in your research, please cite our SIGIR 2026 paper:

```bibtex
@misc{shi2026faveflowbasedaveragevelocity,
      title={FAVE: Flow-based Average Velocity Establishment for Sequential Recommendation}, 
      author={Ke Shi and Yao Zhang and Feng Guo and Jinyuan Zhang and JunShuo Zhang and Shen Gao and Shuo Shang},
      year={2026},
      eprint={2604.04427},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2604.04427}, 
}
