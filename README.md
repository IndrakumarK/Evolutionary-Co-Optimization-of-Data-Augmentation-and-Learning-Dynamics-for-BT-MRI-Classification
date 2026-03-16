
# 🧠 ETDACVO: Adaptive Evolutionary Optimization for Medical Image Learning

**ETDACVO (Enhanced Tasmanian Devil Anti-Conservative Variable Optimization)**  
is a hybrid evolutionary optimization framework designed to improve **training stability, cross-domain robustness, and anatomical fidelity** in deep learning systems for **medical image analysis**.

The framework co-optimizes **data augmentation strategies and optimizer dynamics** within a unified evolutionary loop, enabling robust learning under heterogeneous medical imaging conditions.

---

# 📌 Overview

Deep learning models in medical imaging often face challenges such as:

- Scanner variability
- Domain shift between hospitals
- Limited labeled datasets
- Class imbalance
- Overfitting
- Unstable optimization dynamics

ETDACVO addresses these issues by **jointly evolving**:

- **Data augmentation parameters**
- **Optimizer hyperparameters** (learning rate, momentum, weight decay)

The framework integrates:

- 🐾 **Tasmanian Devil Optimization (TDO)** – global exploration via Lévy flight search  
- 🎯 **Anti-Conservative Variable Optimization (ACVO)** – diversity preservation via population covariance  
- 📉 **EWMA Smoothing** – stabilization of evolutionary updates  

This hybrid mechanism balances **exploration, diversity, and convergence stability**.

---

# 🧬 Core Algorithm

ETDACVO updates the parameter vector using:

θₜ₊₁ = θₜ + α₁Tₜ + α₂Aₜ + (φₜ − θₜ)

Where:

| Symbol | Meaning |
|------|------|
| Tₜ | Tasmanian Devil exploration step |
| Aₜ | ACVO diversity perturbation |
| φₜ | EWMA smoothed parameter trajectory |
| α₁, α₂ | exploration and diversity scaling coefficients |

This update rule allows the optimizer to simultaneously:

- explore the search space
- maintain population diversity
- stabilize parameter evolution.

---

# 🚀 Key Results (Reported in Paper)

Across four MRI datasets:

- **+1.0–1.4% accuracy improvement**
- **+0.03–0.04 Dice score improvement**
- **19–22 fewer epochs to convergence (~30% faster)**
- **~45% reduction in performance variance**
- **92.8% cross-domain retention**

---

# 🧩 Main Components

| Component | Description |
|------|------|
| **TDO** | Lévy-flight exploration mechanism |
| **ACVO** | covariance-driven diversity injection |
| **EWMA** | temporal smoothing of parameter updates |
| **Structural Fidelity Fitness** | SSIM / PSNR / LPIPS constraints |
| **ECC** | Evolutionary convergence confidence |
| **CA-EA-GradCAM** | convergence-aware explainability |

---

# 🧠 Explainability: CA-EA-GradCAM

ETDACVO introduces **CA-EA-GradCAM**, a convergence-aware explainability method combining:

- CNN Grad-CAM localization
- Transformer attention maps
- Evolutionary Convergence Confidence (ECC)

Fusion equation:

H_EA = Ψ [ σ(E_CNN) M_CNN + (1 − σ(E_CNN)) M_Trans ]

Where:

- **M_CNN** – GradCAM saliency map
- **M_Trans** – transformer attention map
- **Ψ** – convergence confidence score
- **σ(E_CNN)** – entropy-based gating function

This produces **trust-calibrated tumor localization maps**.

---

# 📊 Datasets

Experiments were conducted on four public MRI datasets.

| Dataset | Images | Classes |
|------|------|------|
| Nickparvar | 7,023 | Glioma, Meningioma, Pituitary, Normal |
| Mendeley | 12,064 | Glioma, Meningioma, Pituitary, Normal |
| BRISC | ~6,000 | Glioma, Meningioma, Pituitary, Normal |
| Figshare | 3,064 | Brain tumor categories |

Dataset directory structure:

datasets/
    nickparvar/
    mendeley/
    brisc/
    figshare/

Dataset paths can be configured in:

configs/dataset_config.yaml

---

# ⚡ Quick Start

Install dependencies:

pip install -r requirements.txt

or

pip install -e .

Run ETDACVO training:

python experiments/run_training.py

Run explainability analysis:

python analysis/run_gradcam.py

---

# 🔁 Reproducibility

All experiments were repeated using five random seeds:

{42, 52, 62, 72, 82}

Seed control is implemented in:

training/train.py

Runtime per evolutionary generation is logged automatically and saved to:

experiments/runtime_log.csv

---

# ## 📂 Repository Structure

```
ETDACVO-Medical-Image-Learning/
│
├── configs/           # Experiment configuration files
├── preprocessing/     # MRI preprocessing pipeline
├── augmentation/      # Data augmentation modules
├── optimizer/         # ETDACVO optimizer (TDO, ACVO, EWMA)
├── models/            # CNN, ViT, and hybrid architectures
├── training/          # Training and evolutionary optimization
├── experiments/       # Scripts to reproduce experiments
├── analysis/          # Performance analysis and plots
├── utils/             # Utility functions
│
├── requirements.txt   # Python dependencies
├── setup.py           # Package installer
└── README.md          # Project documentation
```


---

# 📚 Citation

If you use this repository in research, please cite:

@article{indrakumar2026etdacvo,
  title={ETDACVO: Enhanced Tasmanian Devil Anti-Conservative Variable Optimization for Cross-Domain Medical Image Learning},
  author={Indrakumar, K. et al.},
  journal={Under Review},
  year={2026}
}

---

# 🛡 License

MIT License
