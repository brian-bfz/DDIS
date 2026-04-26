# Decoupled Diffusion Inverse Solver

![whole pipeline](https://img.i207m.top/2026/04/1a9ad02ad7b011039e17dbcce9575bb4.png)

Official implementation for **[Decoupled Diffusion Sampling for Inverse Problems on Function Spaces](https://arxiv.org/abs/2601.23280)**, an **oral** presentation at the **ICLR 2026 Workshop on AI & PDEs**.

DDIS is a physics-aware diffusion framework for inverse PDE problems with sparse observations and limited paired supervision. Instead of learning a joint distribution over coefficients and solutions, DDIS decouples the problem into two parts: an unconditional diffusion model learns the prior over unknown coefficient fields, and a neural operator learns the forward PDE map used for likelihood guidance. This gives dense, physics-informed guidance during posterior sampling while avoiding the guidance attenuation and over-smoothing seen in joint-embedding diffusion solvers under data scarcity.

## Highlights

- **State-of-the-art inverse PDE reconstruction** on Poisson, Helmholtz, and Navier-Stokes benchmarks with sparse observations.
- **Data-efficient physics guidance**: DDIS keeps strong accuracy even when the neural operator sees only 1% of paired coefficient-solution data.
- **Sharper high-frequency recovery**: average spectral error improves by 54% over prior diffusion-based solvers reported in the paper.
- **Modular design**: swap diffusion priors, learned neural-operator surrogates, and posterior samplers such as DPS/DAPS.

### Results at a Glance

Under standard supervision and comparable inference budgets, DDIS improves both reconstruction error and spectral fidelity:

| Method | Poisson L2 (%) | Poisson Es | Helmholtz L2 (%) | Helmholtz Es | N-S L2 (%) | N-S Es |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DiffusionPDE | 74.68 | 0.566 | 46.10 | 0.315 | 32.78 | 2.099 |
| FunDPS | 19.96 | 0.192 | 17.16 | 0.140 | 8.99 | 0.382 |
| **DDIS** | **15.78** | **0.074** | **15.08** | **0.044** | **8.93** | **0.165** |

DDIS is especially robust when paired PDE solves are scarce:

| Paired Data | Poisson L2 (%) | Helmholtz L2 (%) | N-S L2 (%) |
| --- | ---: | ---: | ---: |
| FunDPS, 1% data | 35.81 | 41.69 | 13.65 |
| **DDIS, 1% data** | **18.70** | **16.40** | **12.05** |
| **DDIS + physics loss, 1% data** | **16.56** | **16.05** | / |

For details, theory, and full ablations, please see the [paper](https://arxiv.org/abs/2601.23280).

## Setup

We support two tested setup paths depending on your hardware:

- **x86_64 Linux with NVIDIA GPU** (e.g., RTX 4090): use `environment.yml`.
- **VISTA aarch64/ARM64 GPU nodes**: use `environment_platform.yml` plus a small PyTorch post-step.

### 1. x86_64 (e.g., RTX 4090)

```shell
conda env create -f environment.yml
conda activate edm_no

python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
# Expected (tested) output on an RTX 4090 machine:
# 2.7.1+cu126 12.6
# CUDA available: True
```

If your output differs significantly, you may need to adapt CUDA / PyTorch versions for your particular GPU, but this configuration is known to work on an RTX 4090 machine.

### 2. VISTA aarch64/ARM64 GPU nodes

From a VISTA interactive job:

```shell
idev -p ...
module load gcc cuda

conda env create -f environment_platform.yml
conda activate edm_no

# One-time setup per VISTA machine: always use the env's libstdc++ (fixes GLIBCXX_* issues)
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat >> "$CONDA_PREFIX/etc/conda/activate.d/edm_no_libstdcxx.sh" << 'EOF'
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
EOF

# Install a tested CUDA 12.4 PyTorch stack
python -m pip install --force-reinstall --no-deps \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
# Expected (tested) output on VISTA:
# 2.5.1 12.4
# CUDA available: True
```

This keeps `environment_platform.yml` portable across architectures, while the post-step installs a matching GPU-capable PyTorch build on VISTA aarch64 and ensures the environment's C++ runtime is consistently used.

### Install `neuraloperator` from Git

After the environment is created and activated (either path above), install the `neuraloperator` used in this project:

```shell
pip install "git+https://github.com/neuraloperator/neuraloperator.git@98cd305099f4a2b232ed85773984f3e5991f9b1a"
```

It is important to use this specific commit, because [#702](https://github.com/neuraloperator/neuraloperator/pull/702) fixed a critical bug in the FNO implementation that would cause reproducibility issues and poor surrogate performance.

### Download data

We support both an **automatic Hugging Face download** (recommended) and the **original manual download**.

- **Option A (recommended, automatic from Hugging Face)**

  This repository includes `utils/download_dataset.py`, which pulls the normalized PDE datasets from the Hugging Face dataset `jcy20/DiffusionPDE-normalized` and places them under `data/DiffPDE`:

  ```shell
  # From the project root
  python utils/download_dataset.py all --output-dir data/DiffPDE
  ```

  - By default this downloads the **training** splits for all supported datasets (`darcy`, `helmholtz`, `ns-bounded`, `ns-nonbounded`, `poisson`).
  - To download the **test** splits instead, add `--test`:

    ```shell
    python utils/download_dataset.py all --output-dir data/DiffPDE --test
    ```

  - You can also download a **single dataset** (train or test) by name, e.g.:

    ```shell
    # Single dataset (Darcy, train split)
    python utils/download_dataset.py darcy --output-dir data/DiffPDE

    # Single dataset (Darcy, test split)
    python utils/download_dataset.py darcy --output-dir data/DiffPDE --test
    ```

  After this step, you should have a structure like:

  - `data/DiffPDE/darcy_hf`, `data/DiffPDE/darcy_test_hf`
  - `data/DiffPDE/helmholtz_hf`, `data/DiffPDE/helmholtz_test_hf`
  - `data/DiffPDE/ns-bounded_hf`, `data/DiffPDE/ns-bounded_test_hf`
  - `data/DiffPDE/ns-nonbounded_hf`, `data/DiffPDE/ns-nonbounded_test_hf`
  - `data/DiffPDE/poisson_hf`, `data/DiffPDE/poisson_test_hf`

- **Option B (manual, original DiffusionPDE instructions)**

  You can alternatively follow the original data download instructions in the DiffusionPDE repository and then copy the data into `data/DiffPDE`:

  ```text
  https://github.com/jhhuangchloe/DiffusionPDE
  ```

  To generate the processed data used by this repo, run:

  ```shell
  python utils/dataset_process.py all
  ```

To initialize the wandb environment, run:

```shell
wandb init
```

## Usage

We will release trained checkpoints soon. Stay tuned and hit star if you find this repo useful!

### Training

**Diffusion model** (backbone for generation):

```shell
python scripts/train/train.py -c configs/training/poisson.yml --name poisson-run
```

Configs for each PDE are in `configs/training/` (e.g. `poisson.yml`, `helmholtz.yml`, `ns-nonbounded.yml`).

**FNO surrogate** (physics guidance model, trained separately):

```shell
python scripts/train/training_fno.py --config configs/training/no_surrogate/fno_pad_poisson_forward.yml
```

Surrogate configs are in `configs/training/no_surrogate/`, one per PDE and direction (forward/backward).

### Inference

```shell
python scripts/generate/generate_pde.py --config configs/generation/poisson_backward.yaml
```

Generation configs are in `configs/generation/`. Key options to set in the config:
- `surrogate_type`: `fno_pad` (learned surrogate) or `numerical_poisson` (exact solver, Poisson only)
- `surrogate_path`: path to trained FNO `.pth` file
- `guidance.type`: `daps` (recommended) or `dps`

## Unit Tests

Validation scripts live in `unit_tests/`. They test individual components without running the full generation pipeline.

```shell
# Test NumericalPoissonWrapper logic (no data required)
python unit_tests/test_poisson_solver.py

# Validate NumericalPoissonWrapper against real Poisson dataset
python unit_tests/test_poisson_solver.py --data-path data/DiffPDE/poisson_test_hf --save-dir exps/tests/poisson_solver

# Validate a trained FNO/FNO_pad surrogate (architecture auto-detected from checkpoint)
python unit_tests/test_fno_surrogate.py --model-path <path-to-model.pth> --data-path data/DiffPDE/helmholtz_test_hf

# Validate a trained PINO surrogate
python unit_tests/test_pino_surrogate.py --model-path <path-to-model.pth>
```

Outputs (visualizations, metrics) are saved under `exps/tests/`.

## Visualization

Analysis and plotting scripts live in `visualization/scripts/`.

```shell
# Aggregate and plot sweep results
python visualization/scripts/aggregate_sweep_results.py

# Power spectrum comparison across methods
python visualization/scripts/power_spectrum.py
```
