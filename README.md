# Cerebral Tumor Quantitative Simulation

## Project Summary

This repository contains code and resources for solving inverse parameter identification problems in nonlinear parabolic PDEs, specifically for modeling tumor growth dynamics. Our machine learning framework integrates neural networks with PDE solvers to infer latent parameters from observable data.

For detailed notes, please refer to the [Project Notes Document](https://docs.google.com/document/d/1iC35VlKeHJFTMue7btkZ2kINdtwvVeS2nBEdWYb2N0g/edit?usp=sharing).

---

## Abstract

### Overview

We aim to estimate model parameters (such as the diffusion coefficient $\alpha$) from noisy observations of the solution to a PDE (e.g., tumor cell density at time $t = 1$). This is framed as an inverse problem using a physics-informed machine learning approach.

### Key Features
- Solves a reaction–diffusion PDE with Neumann boundary conditions
- Combines data-driven and model-based approaches
- Custom loss function including parameter loss and forward consistency loss

---

### Mathematical Formulation

#### Forward Problem

We consider the semi-linear parabolic PDE
$$
\partial_t u(x,t)
- \nabla \cdot \big( \tilde{\alpha}(x)\nabla u(x,t) \big)
+ \tilde{\rho}(x)\,u(x,t)\big(1-u(x,t)\big)
= 0,
$$

for $(x,t) \in \Omega_B \times (0,1]$, with initial condition
$$
u(x,0) = u_0(x), \quad x \in \Omega_B,
$$
and Neumann boundary conditions
$$
\partial_n u = 0 \quad \text{on } \partial\Omega_B \times (0,1].
$$

Here, $\Omega_B \subset \mathbb{R}^d$, $d \in \{1,2,3\}$, denotes the brain domain.

The spatially varying diffusion coefficient is modeled as
$$
\tilde{\alpha}(x) = \alpha\,\pi_W(x) + 0.2\,\alpha\,\pi_G(x),
$$
where $\alpha > 0$ is a scalar parameter, and
$\pi_W, \pi_G : \Omega_B \to [0,1]$
denote probability maps for white matter and gray matter, respectively.

The proliferation term is given by
$$
\tilde{\rho}(x) = \rho\,\pi_W(x) + 0.2\,\rho\,\pi_G(x),
$$
with scalar proliferation rate $\rho > 0$.

The tissue probability maps $\pi_j$, $j \in \{W,G\}$, are obtained from the ICBM MNI dataset.

#### Inverse Problem and Loss Function

The inverse problem consists of estimating parameters (e.g., $\alpha$)
from observations of the state $u$ at final time $t = 1$.

The total loss combines:

- **Parameter loss**
$$
\frac{1}{2}\,\big\|\Gamma^{-1/2}(\alpha_{\text{true}} - \alpha_{\text{pred}})\big\|^2
$$

- **Data loss**
$$
\frac{\lambda}{2}\,\big\|\Lambda^{-1/2}\big(u_{\text{true}}(t=1) - u_{\text{pred}}(t=1)\big)\big\|^2
$$

---

## Code Structure

| File | Description |
|------|-------------|
| `genNNData.py` | Generates training/test data using the PDE solver |
| `RDPDE.py` | Implements the PDE forward solver with Crank–Nicolson and analytical reaction steps |
| `run_DNN_MC.py` | Trains the NN with custom loss (parameter + forward consistency) |
| `run_DNN.py` | Trains the NN with only parameter loss |
| `SetupNN.py` | Functions for data handling and network construction |
| `PerfMeasures.py` | Includes evaluation metrics such as CMSE and squared bias |
| `viz_data.py` | Visualization scripts for predictions and loss curves |

---

## Training Details

- **Input**: $u(t = 1)$
- **Output**: Predicted $\alpha$
- **Loss**: Combination of parameter and data loss
- **Optimizer**: Adam
- **Batch Size**: 8 (configurable)
- **Epochs**: 300 (default)

---

## Results

[Download SIAM '24 @ Baylor University Poster PDF](https://github.com/user-attachments/files/17637383/mayank_sciml_poster.pdf)

![SIAM '24 @ Baylor University](https://github.com/user-attachments/assets/c0e91500-fb08-4c13-b943-b32e09cbe005)

[Download SIAM '25 @ Oden Institute Poster PDF](https://github.com/user-attachments/files/24265389/Mayank_SIAM.25_Poster.pdf)

![SIAM '25 @ Oden Institute](https://github.com/user-attachments/assets/56993617-650e-47dd-9da7-898a53fe37b4)

---

## References

[1] Mang, A., Gholami, A., & Biros, G. (1996). *An inverse problem formulation for parameter estimation of a reaction–diffusion model for low-grade gliomas*. Journal of Mathematical Biology, 72(1), 409–433.

---

## Acknowledgments

This work is conducted under the supervision of Prof. Andreas Mang. The code structure and PDE formulation are based on collaborative research efforts in scientific machine learning and biomedical modeling.

This project is funded by **NSF CAREER Award #2145845:** *Scalable Algorithms for Nonlinear, Large-Scale Inverse Problems Governed by Dynamical Systems*.
