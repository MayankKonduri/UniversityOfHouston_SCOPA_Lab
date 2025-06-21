# Cerebral Tumor Quantitative Simulation

## Project Summary

This repository contains code and resources for solving inverse parameter identification problems in nonlinear parabolic PDEs, specifically for modeling tumor growth dynamics. Our machine learning framework integrates neural networks with PDE solvers to infer latent parameters from observable data.

For detailed notes, please refer to the [Project Notes Document](https://docs.google.com/document/d/1iC35VlKeHJFTMue7btkZ2kINdtwvVeS2nBEdWYb2N0g/edit?usp=sharing).

---

## Abstract

### Overview

We aim to estimate model parameters (such as the diffusion coefficient α) from noisy observations of the solution to a PDE (e.g., tumor cell density at time t = 1). This is framed as an inverse problem using a physics-informed machine learning approach.

### Key Features
- Solves a reaction-diffusion PDE with Neumann boundary conditions
- Combines data-driven and model-based approaches
- Custom loss function including parameter loss and forward consistency loss

### Mathematical Formulation

The total loss combines:
- **Parameter loss**: (1/2) * ‖Γ^(-1/2) (α_true − α_pred)‖²
- **Data loss**: (λ/2) * ‖Λ^(-1/2) (u_true(t=1) − u_pred(t=1))‖²


## Code Structure

| File | Description |
|------|-------------|
| `genNNData.py` | Generates training/test data using the PDE solver |
| `RDPDE.py` | Implements the PDE forward solver with Crank-Nicholson and analytical reaction steps |
| `run_DNN_MC.py` | Trains the NN with custom loss (parameter + forward consistency) |
| `run_DNN.py` | Trains the NN with only parameter loss |
| `SetupNN.py` | Functions for data handling and network construction |
| `PerfMeasures.py` | Includes evaluation metrics such as CMSE and squared bias |
| `viz_data.py` | Visualization scripts for predictions and loss curves |

---

## Training Details

- **Input**: \( u(t=1) \)
- **Output**: Predicted \( \alpha \)
- **Loss**: Combination of parameter and data loss
- **Optimizer**: Adam
- **Batch Size**: 8 (configurable)
- **Epochs**: 300 (default)

---

## Results

![Poster](https://github.com/user-attachments/assets/c0e91500-fb08-4c13-b943-b32e09cbe005)

[Download Poster PDF](https://github.com/user-attachments/files/17637383/mayank_sciml_poster.pdf)

---

## References

[1] Mang, A., Gholami, A., & Biros, G. (1996). *An inverse problem formulation for parameter estimation of a reaction-diffusion model for low-grade gliomas*. Journal of Mathematical Biology, 72(1), 409–433.

---

## Acknowledgments

This work is conducted under the supervision of Prof. Andreas Mang. The code structure and PDE formulation are based on collaborative research efforts in scientific machine learning and biomedical modeling.
