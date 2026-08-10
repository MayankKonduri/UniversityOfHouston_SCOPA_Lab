<div align="center">

# 🧠 Cerebral Tumor Quantitative Simulation

### Learning PDE Parameters from Noisy Observations of Tumor Growth

<p>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/SciPy-Sparse_Solvers-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy" />
  <img src="https://img.shields.io/badge/NumPy-Arrays-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
</p>
 
<p>
  <img src="https://img.shields.io/badge/domain-1D_prototype-blue?style=flat-square" alt="Domain" />
  <img src="https://img.shields.io/badge/samples-10,000-informational?style=flat-square" alt="Samples" />
  <img src="https://img.shields.io/badge/scheme-Strang_splitting-green?style=flat-square" alt="Scheme" />
  <img src="https://img.shields.io/badge/NSF-CAREER_%232145845-red?style=flat-square" alt="NSF" />
</p>

<sub>SCOPA Lab · University of Houston · Advised by Prof. Andreas Mang</sub>

</div>

---

## Project Summary

This repository contains code and resources for solving inverse parameter
identification problems in nonlinear parabolic PDEs, specifically for modeling
tumor growth dynamics. The framework integrates neural networks with PDE
solvers to infer latent parameters from observable data.

A network is trained to read the tumor cell density field at the final time
$t = 1$ and regress the scalar diffusion coefficient $\alpha$ that produced it.
Training data comes from thousands of forward PDE solves at randomly sampled
$\alpha$; the inverse map is learned rather than solved by optimization at
inference time.

For detailed notes, see the
[Project Notes Document](https://docs.google.com/document/d/1iC35VlKeHJFTMue7btkZ2kINdtwvVeS2nBEdWYb2N0g/edit?usp=sharing).

> [!IMPORTANT]
> **Scope of the current code.** The formulation below describes the target
> problem: a 3D brain domain $\Omega_B$ with spatially varying tissue maps
> $\pi_W, \pi_G$ from ICBM MNI. **The code in this repository implements a 1D
> prototype of that formulation** — a single spatial dimension on $[0,1]$ with
> $n = 128$ nodes, a scalar $\alpha$, and proliferation $\rho$ held fixed at
> `0.05`. No tissue probability maps are loaded, and $d = 1$ throughout.
>
> This is the intended order of work, not a defect — the 1D case establishes
> the learning approach before the anatomical geometry is introduced. The
> distinction is called out here so the equations aren't mistaken for a
> description of what runs today.

---

## Mathematical Formulation

### Forward Problem

We consider the semi-linear parabolic PDE

$$\partial_t u - \nabla \cdot \left( \tilde{\alpha}(x) \nabla u(x,t) \right) + \tilde{\rho}(x)\, u(x,t)\left(1 - u(x,t)\right) = 0$$

for $(x,t) \in \Omega_B \times (0,1]$, with initial condition

$$u(x,0) = u_0(x), \quad x \in \Omega_B,$$

and Neumann boundary conditions

$$\frac{\partial u}{\partial n} = 0 \quad \text{on } \partial\Omega_B \times (0,1].$$

Here $\Omega_B \subset \mathbb{R}^d$, with $d \in \\{1,2,3\\}$, denotes the
brain domain.

The spatially varying diffusion coefficient is modeled as

$$\tilde{\alpha}(x) = \alpha\, \pi_W(x) + 0.2\, \alpha\, \pi_G(x),$$

where $\alpha > 0$ is a scalar parameter, and
$\pi_W, \pi_G : \Omega_B \to [0,1]$ denote probability maps for white matter
and gray matter respectively.

The proliferation term is given by

$$\tilde{\rho}(x) = \rho\, \pi_W(x) + 0.2\, \rho\, \pi_G(x),$$

with scalar proliferation rate $\rho > 0$.

The tissue probability maps $\pi_j$, $j \in \\{W, G\\}$, are obtained from the
ICBM MNI dataset.

### Inverse Problem and Loss Function

The inverse problem consists of estimating parameters (e.g. $\alpha$) from
observations of the state $u$ at final time $t = 1$.

The total loss combines:

**Parameter loss**

$$\tfrac{1}{2} \left\lVert \Gamma^{-1/2} \left( \alpha_{\text{true}} - \alpha_{\text{pred}} \right) \right\rVert^2$$

**Data loss**

$$\tfrac{\lambda}{2} \left\lVert \Lambda^{-1/2} \left( u_{\text{true}}(t{=}1) - u_{\text{pred}}(t{=}1) \right) \right\rVert^2$$

In `run_DNN_MC.py`, $\Gamma^{-1/2}$ and $\Lambda^{-1/2}$ are set empirically to
the inverse standard deviations of the training parameters and observations
respectively, and $\lambda = 1$.

---

## How It Works

```
   ┌──────────────────────────────────────────────────────────┐
   │  genNNData.py  /  genNNNoisyData.py                      │
   │                                                          │
   │  α ~ U(1e-4, 1e-2), 10,000 draws                         │
   │        │                                                 │
   │        ▼                                                 │
   │  RDPDE.fwd_sol(u0, α, ρ=0.05)   ← Strang splitting       │
   │        │                          CN diffusion ½ step    │
   │        │                          analytic logistic step │
   │        │                          CN diffusion ½ step    │
   │        ▼                                                 │
   │  keep u(·, t=1)   [+ δ·noise]                            │
   └────────────────────────┬─────────────────────────────────┘
                            │  rdiffEQ-nn-spl-10000.npz
                            │  y: (10000, 129)   θ: (10000,)
                            ▼
   ┌──────────────────────────────────────────────────────────┐
   │  SetupNN.build_dnn      Dense(32) + swish, 1 hidden layer │
   │                                                          │
   │      u(t=1) ∈ R^129  ──────────────►  α̂ ∈ R              │
   └────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      run_DNN.py                   run_DNN_MC.py
      parameter loss only          parameter + data loss
                                   (re-solves PDE in loop)
              └─────────────┬─────────────┘
                            ▼
                  PerfMeasures.eval_pred
                  squared bias · CMSE
```

The two training scripts differ only in the loss. `run_DNN.py` is the baseline:
plain MSE against the true $\alpha$. `run_DNN_MC.py` adds the forward
consistency term, re-solving the PDE for each predicted $\alpha$ inside the loss
and comparing the resulting state against the observation.

---

## Repository Layout

```text
.
├── RDPDE.py                  # reaction–diffusion solver (the core numerics)
├── HeatPDE.py                # pure diffusion solver, no reaction — reference case
│
├── genNNData.py              # generate 10k clean samples
├── genNNNoisyData.py         # same, with additive Gaussian noise (δ = 0.1)
│
├── SetupNN.py                # data loading + Keras model construction
├── run_DNN.py                # train with parameter loss only
├── run_DNN_MC.py             # train with parameter + forward consistency loss
├── PerfMeasures.py           # squared bias and CMSE
├── viz_data.py               # plot the observation ensemble
│
├── xmpls/                    # standalone sanity checks
│   ├── solHeatPDE1D.py       #   forward solve, heat equation
│   ├── solRDPDE1D.py         #   forward solve, reaction–diffusion
│   └── vizNNData.py          #   plot random samples from a dataset
│
├── rdiffEQ-nn-spl-10000.npz  # 10k clean samples (~10 MB, committed)
├── chkpts/chkpts.keras       # saved model checkpoint
└── prediction-vs-true-4k-delta0d1.pdf
```

---

## Numerical Method

`RDPDE.py` is where the physics lives. Two integrators are implemented:

| Method | Function | Notes |
|:---|:---|:---|
| **Strang splitting** | `fwd_sol_os` | Default. Crank–Nicolson half-step for diffusion, closed-form logistic update for reaction, second CN half-step. |
| **RK2** | `fwd_sol_rk2` | Explicit two-stage Runge–Kutta on the full right-hand side. Present but not called. |

The Laplacian is a standard three-point stencil on $n+1$ nodes, scaled by
$1/h^2$, with the first and last rows modified to impose homogeneous Neumann
conditions via ghost-node reflection.

The reaction step is solved analytically rather than numerically, using the
closed-form logistic solution

$$u^{k+1} = \left[ 1 + \frac{1 - u^k}{u^k} e^{-\rho \Delta t} \right]^{-1}$$

with $u^k$ clamped to $[10^{-6}, 1 - 10^{-6}]$ to avoid division by zero.

<details>
<summary><b>Default discretization and sampling parameters</b></summary>

<br>

| Parameter | Value | Set in |
|:---|:---|:---|
| Spatial nodes $n$ | `128` (129 grid points) | `genNNData.py` |
| Time steps $n_t$ | `16` | `genNNData.py` |
| $\alpha$ range | `[1e-4, 1e-2]`, uniform | `genNNData.py` |
| $\rho$ | `0.05`, fixed | `genNNData.py` |
| Samples | `10000` | `genNNData.py` |
| Initial condition | Gaussian, $\mu = 0.5$, $\sigma^2 = 5\times10^{-4}$ | `RDPDE.setup_u0` |
| Noise level $\delta$ | `0.1` | `genNNNoisyData.py` |

</details>

---

## Training Details

- **Input:** $u(t = 1)$ — a 129-dimensional vector
- **Output:** predicted $\alpha$ — scalar
- **Architecture:** Flatten → Dense(32) + swish → Dense(1) + swish
- **Loss:** parameter loss (`run_DNN.py`) or parameter + data loss (`run_DNN_MC.py`)
- **Optimizer:** Adam, learning rate `1e-3`
- **Batch size:** 8
- **Epochs:** 300
- **Split:** 2,000 test / 4,000 train (of 10,000 available)

> [!NOTE]
> The final layer uses a **swish** activation, which is not non-negative — it
> dips to roughly $-0.28$ for negative inputs. Since $\alpha > 0$ by
> construction, `softplus` or `exp` would enforce positivity by design. Worth
> checking whether any predictions come back negative.

### Evaluation

`PerfMeasures.comp_cmse` reports two quantities:

- **Squared bias** — $(\overline{\alpha_{\text{true}}} - \overline{\alpha_{\text{pred}}})^2$, systematic offset
- **CMSE** — centered mean squared error, the error remaining after both series are mean-subtracted

Reporting these separately separates a constant offset from genuine
sample-to-sample error, which a single MSE would conflate.

---

## Quickstart

```bash
git clone https://github.com/MayankKonduri/UniversityOfHouston_SCOPA_Lab.git
cd UniversityOfHouston_SCOPA_Lab

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

pip install numpy scipy matplotlib tensorflow
```

> [!WARNING]
> Before running either training script, **edit the hardcoded `path` variable
> at the top** — see [Known Issues](#-known-issues). As committed, both scripts
> write checkpoints to an absolute path on another machine and will fail.

```bash
# Sanity check: forward solve, no ML
python xmpls/solRDPDE1D.py

# Inspect the shipped dataset
python viz_data.py

# Train (dataset already committed — regeneration is optional)
python run_DNN.py

# Regenerate data from scratch (~10k PDE solves, slow)
python genNNData.py
```

<details>
<summary><b>Generating the noisy datasets</b></summary>

<br>

Only the clean dataset is committed. Both training scripts contain commented
references to noisy variants (`δ = 0.001`, `0.01`, `0.1`) that are **not in the
repository**. To produce one, set `delta` in `genNNNoisyData.py` and run it:

```bash
python genNNNoisyData.py    # writes rdiffEQ-nn-spl-10000-addnoise-delta=0.1.npz
```

</details>

---

## Results

[Download SIAM '24 @ Baylor University Poster PDF](https://github.com/user-attachments/files/17637383/mayank_sciml_poster.pdf)

![SIAM '24 @ Baylor University](https://github.com/user-attachments/assets/c0e91500-fb08-4c13-b943-b32e09cbe005)

[Download SIAM '25 @ Oden Institute Poster PDF](https://github.com/user-attachments/files/24265389/Mayank_SIAM.25_Poster.pdf)

![SIAM '25 @ Oden Institute](https://github.com/user-attachments/assets/56993617-650e-47dd-9da7-898a53fe37b4)

---

## ⚠️ Known Issues

These are observations from reading the code, ordered by how much they block a
fresh clone from running. Nothing here has been fixed.

> [!CAUTION]
> **Hardcoded absolute paths break both training scripts.**
> `run_DNN.py` sets `path = '/Users/mayank/Documents/Github/pdeml'` and
> `run_DNN_MC.py` sets `path = '/Users/amang/Research/code/pdeml'`. Both feed
> `ModelCheckpoint(filepath=path + "/chkpts/" + chkfn)`, so training fails at
> the first epoch boundary on any other machine. Replacing both with
> `os.path.join(os.path.dirname(__file__), "chkpts")` would make the repo
> clone-and-run.

> [!CAUTION]
> **`run_DNN.py` evaluates on a dataset that isn't in the repository.** After
> `fit()`, line ~73 reloads
> `rdiffEQ-nn-spl-10000-addnoise-delta=0.01.npz`, which is not committed —
> `FileNotFoundError`. Note also that this trains on clean data and tests on
> noisy data. If that generalization gap is intentional it deserves a comment;
> if it's a leftover, the reload should be deleted.

> [!WARNING]
> **The forward-consistency loss may contribute no gradient.** In
> `MCDNNLoss.call`, the PDE solve is wrapped in `tf.py_function` around a pure
> NumPy routine (`pde_eval`). TensorFlow cannot differentiate through NumPy, so
> `∂(data_loss)/∂θ_pred` is likely `None` — meaning `run_DNN_MC.py` may be
> training on the parameter loss alone while paying the cost of 10k PDE solves
> per epoch. Worth verifying directly: print
> `tape.gradient(data_loss, model.trainable_weights)` and check for `None`.
> If confirmed, the options are a differentiable solver, an adjoint via
> `tf.custom_gradient`, or a surrogate.

> [!WARNING]
> **`mode='accuracy'` is not a valid `ModelCheckpoint` mode.** `run_DNN.py`
> passes it alongside `monitor='loss'`. Keras accepts only `auto`, `min`, and
> `max`, and silently falls back to `auto`. Should be `mode='min'`.

> [!NOTE]
> **Duplicate layer names for `n_layers ≥ 2`.** In `SetupNN.build_dnn`, the
> first layer is named `dense_1`, then the loop
> `for i in range(0, n_layers-1)` names layers `dense_{i+1}` — so `i=0`
> produces a second `dense_1`. Currently masked because `n_layers = 1` makes
> the loop empty. Starting the loop index at 2 would fix it before it bites.

> [!NOTE]
> **`HeatPDE` and `RDPDE` scale the Laplacian differently.** `RDPDE.do_setup`
> divides by $h^2$; `HeatPDE.do_setup` has that division commented out
> (`ld = ld # / (h*h)`). The two solvers are therefore not directly comparable
> at the same $\alpha$ — which is consistent with `xmpls/solHeatPDE1D.py`
> needing $\alpha = 20$ where the RD example uses $0.001$.

> [!NOTE]
> **`fwd_sol` ignores its `u0` argument.** It accepts `u0` but uses
> `self.u0` from `setup_u0()` instead. Every caller passes `u0 = 1.0`, which
> has no effect. Either wire it through or drop the parameter.

> [!NOTE]
> **`xmpls/vizNNData.py` points at a file that doesn't exist.** It loads
> `../heatEQ-nn-spl-10000.npz`; the repo ships `rdiffEQ-nn-spl-10000.npz`.

> [!NOTE]
> **Repository hygiene.** `__pycache__/*.pyc` and `.DS_Store` are tracked, and
> there is no `.gitignore`. A minimal one:
> ```gitignore
> __pycache__/
> *.py[cod]
> .DS_Store
> .venv/
> *.npz
> ```
> The 10 MB `rdiffEQ-nn-spl-10000.npz` is also committed directly. Since
> `genNNData.py` regenerates it deterministically given a seed, Git LFS or a
> download script would keep the clone small — though seeding
> `np.random.uniform` first would be needed for exact reproducibility.

---

## References

[1] Gholami, A., Mang, A., & Biros, G. (2016). *An inverse problem formulation
for parameter estimation of a reaction–diffusion model of low grade gliomas*.
Journal of Mathematical Biology, 72(1), 409–433.
[doi:10.1007/s00285-015-0888-x](https://doi.org/10.1007/s00285-015-0888-x)

---

## Acknowledgments

This work is conducted under the supervision of Prof. Andreas Mang. The code
structure and PDE formulation are based on collaborative research efforts in
scientific machine learning and biomedical modeling.

This project is funded by **NSF CAREER Award #2145845:** *Scalable Algorithms
for Nonlinear, Large-Scale Inverse Problems Governed by Dynamical Systems*.

---

<div align="center">
<sub>SCOPA Lab · Department of Mathematics · University of Houston</sub>
</div>
