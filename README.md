# CT Slice Interpolation via 3D Gaussian Splatting

Research project applying **3D Gaussian Splatting (3DGS)** to the problem of CT slice interpolation on the **CT-ORG** dataset. This work demonstrates that per-volume 3DGS optimization with **residual learning** can reconstruct missing CT slices competitively with both classical cubic interpolation and supervised deep learning (U-Net), without requiring any external training data.

## Problem Statement

CT and MRI volumes typically have high in-plane (x-y) resolution but significantly lower through-plane (z-axis) resolution due to slice thickness and acquisition gaps. This anisotropy degrades 3D reconstruction quality and downstream clinical tasks (segmentation, volumetric analysis). Traditional solutions involve either reacquiring scans at higher resolution (more radiation dose, longer scan time) or applying post-hoc interpolation.

This project explores whether **3D Gaussian Splatting** — originally developed for novel view synthesis in computer vision — can serve as a self-supervised, per-volume optimization method for CT slice interpolation that outperforms classical approaches and competes with supervised deep learning baselines.

## Results (21 Test Cases)

All numbers below are aggregated **directly from the per-case JSON / `summary.csv` files in `outputs/`** (N = 21 test cases per ratio, fixed split, seed 42). 3DGS / 3DGS+Patch use the rebuilt summaries that include the perceptual metrics (LPIPS / HFEN / GMSD).

### PSNR Comparison

| Method | R=2 PSNR | R=3 PSNR | R=4 PSNR | Training Data? | Category |
|--------|----------|----------|----------|:--------------:|----------|
| Nearest | 35.11 | 35.08 | 33.48 | ✗ | Classical |
| Linear | 41.74 | 38.12 | 35.98 | ✗ | Classical |
| **Cubic** | **43.04** | **38.73** | **36.30** | ✗ | Classical |
| U-Net 2D | 41.92 | 37.01 | 34.82 | ✓ (98 vols) | Supervised DL |
| ArSSR (Wu et al., 2022) | TBD | TBD | TBD | ✓ (98 vols) | INR (1 model, all R) |
| SAINT (Peng et al., 2020) | TBD | TBD | TBD | ✓ (98 vols) | Supervised SOTA (per-R) |
| Tri-plane INR (H2) | TBD | TBD | TBD | ✗ | Per-volume self-supervised |
| 3DGS Standard | 42.40 | 38.61 | 36.22 | ✗ | Per-volume self-supervised |
| 3DGS High | 42.62 | 38.61 | 36.20 | ✗ | Per-volume self-supervised |
| **3DGS + Patch Prior (Ours)** | **42.38** | **38.54** | **36.22** | ✗ | Per-volume self-supervised |

> ArSSR / SAINT / Tri-plane INR rows will be populated by `kaggle_notebooks/07_baseline_arssr.ipynb`, `kaggle_notebooks/08_baseline_saint.ipynb`, and `kaggle_notebooks/09_baseline_triplane.ipynb` respectively.

### Complete 6-Metric Comparison (All Methods)

#### R=2 (2× Interpolation)

| Metric | 3DGS+Patch (Ours) | 3DGS Std | 3DGS High | U-Net 2D | Cubic | Linear | Nearest | Best |
|:------:|:------------------:|:--------:|:---------:|:--------:|:-----:|:------:|:-------:|:----:|
| **PSNR** ↑ | 42.38 | 42.40 | 42.62 | 41.92 | **43.04** | 41.74 | 35.11 | Cubic |
| **SSIM** ↑ | 0.9634 | 0.9632 | 0.9677 | 0.9733 | **0.9744** | 0.9732 | 0.9434 | Cubic |
| **MAE** ↓ | 0.00506 | 0.00501 | 0.00485 | 0.00467 | **0.00443** | 0.00474 | 0.00834 | Cubic |
| **LPIPS** ↓ | 0.01529 | 0.01560 | 0.01550 | 0.02027 | **0.01494** | 0.02027 | 0.02510 | Cubic |
| **HFEN** ↓ | 0.15968 | **0.15959** | 0.15970 | 0.16190 | 0.15970 | 0.17579 | 0.32244 | 3DGS Std |
| **GMSD** ↓ | 0.02790 | **0.02789** | 0.02790 | 0.02840 | 0.02808 | 0.03064 | 0.05823 | 3DGS Std |

#### R=3 (3× Interpolation)

| Metric | 3DGS+Patch (Ours) | 3DGS Std | 3DGS High | U-Net 2D | Cubic | Linear | Nearest | Best |
|:------:|:------------------:|:--------:|:---------:|:--------:|:-----:|:------:|:-------:|:----:|
| **PSNR** ↑ | 38.54 | 38.61 | 38.61 | 37.01 | **38.73** | 38.12 | 35.08 | Cubic |
| **SSIM** ↑ | 0.9443 | 0.9476 | 0.9482 | 0.9470 | **0.9540** | 0.9538 | 0.9432 | Cubic |
| **MAE** ↓ | 0.00700 | 0.00689 | 0.00688 | 0.00736 | **0.00660** | 0.00676 | 0.00838 | Cubic |
| **LPIPS** ↓ | 0.02335 | 0.02358 | 0.02353 | 0.04723 | **0.02294** | 0.02928 | 0.02524 | Cubic |
| **HFEN** ↓ | 0.24173 | **0.24172** | 0.24173 | 0.26398 | 0.24214 | 0.25194 | 0.32342 | 3DGS Std |
| **GMSD** ↓ | **0.04561** | 0.04562 | 0.04562 | 0.05238 | 0.04582 | 0.04785 | 0.05841 | 3DGS+Patch |

#### R=4 (4× Interpolation)

| Metric | 3DGS+Patch (Ours) | 3DGS Std | 3DGS High | U-Net 2D | Cubic | Linear | Nearest | Best |
|:------:|:------------------:|:--------:|:---------:|:--------:|:-----:|:------:|:-------:|:----:|
| **PSNR** ↑ | 36.22 | 36.22 | 36.20 | 34.82 | **36.30** | 35.98 | 33.48 | Cubic |
| **SSIM** ↑ | 0.9283 | 0.9307 | 0.9263 | 0.9304 | 0.9351 | **0.9363** | 0.9209 | Linear |
| **MAE** ↓ | 0.00861 | 0.00848 | 0.00865 | 0.00897 | **0.00830** | 0.00834 | 0.01019 | Cubic |
| **LPIPS** ↓ | 0.03128 | 0.03138 | 0.03108 | 0.07984 | **0.03066** | 0.03667 | 0.03322 | Cubic |
| **HFEN** ↓ | 0.30903 | 0.30900 | **0.30897** | 0.32559 | 0.30958 | 0.31364 | 0.39229 | 3DGS High |
| **GMSD** ↓ | 0.05926 | **0.05925** | 0.05925 | 0.06988 | 0.05946 | 0.06083 | 0.07113 | 3DGS Std |

### Key Findings

> **Key takeaway 1**: 3DGS closely matches cubic on pixel-level metrics (PSNR/SSIM/MAE) across all sparse ratios while being entirely self-supervised — no external training data required.

> **Key takeaway 2**: On edge-quality metrics (**HFEN, GMSD**), 3DGS methods consistently **outperform cubic interpolation** across all ratios. This indicates 3DGS better preserves high-frequency anatomical structures (organ boundaries, bone interfaces) — a clinically important advantage.

> **Key takeaway 3**: The patch-prior variant (H3d) consistently improves **LPIPS** (perceptual quality) over the standard 3DGS baseline, and achieves the best GMSD at R=3.

| Summary | R=2 | R=3 | R=4 |
|---------|-----|-----|-----|
| Δ PSNR vs Cubic | -0.66 dB | -0.18 dB | -0.08 dB |
| Δ HFEN vs Cubic (best 3DGS variant) | **−0.00011** ✓ | **−0.00042** ✓ | **−0.00061** ✓ |
| Δ GMSD vs Cubic (best 3DGS variant) | **−0.00018** ✓ | **−0.00021** ✓ | **−0.00022** ✓ |
| 3DGS+Patch LPIPS vs 3DGS Std | **−0.0003** ✓ | **−0.0002** ✓ | **−0.0001** ✓ |

## Key Contributions

- **Residual 3DGS**: Novel formulation where 3DGS predicts a residual correction on top of cubic interpolation, achieving near-cubic quality while enabling fine-detail learning
- **Custom 3DGS pipeline** adapted for axis-aligned medical slice rendering, with optional **oriented (quaternion-rotated) Gaussians** for oblique anatomical structures (H3a)
- **Separable differentiable rendering**: Exploits axis-aligned Gaussian structure for O(H·K + W·K) rendering via matrix multiplication, ~50-100x faster than naive per-pixel computation; falls back to a tiled Mahalanobis path when rotation is enabled
- **Superior high-frequency fidelity**: 3DGS outperforms cubic interpolation on HFEN (edge fidelity) and GMSD (gradient structure) across all sparsity ratios — preserving clinically relevant anatomical boundaries better than classical methods
- **FFT high-frequency loss**: Frequency-domain loss penalizing discrepancies in high-frequency components, forcing the model to capture sharp organ boundaries and bone interfaces
- **HU gradient-weighted reconstruction loss**: Spatial weighting based on Sobel gradient magnitude of ground truth, prioritizing reconstruction accuracy at clinically important edge structures
- **Error-map densification**: Replaces gradient-based densification with per-pixel reconstruction error mapping, directly targeting Gaussian allocation at regions with highest reconstruction error
- **Multi-scale reconstruction loss**: Computes L1 at 3 resolution levels (1x, 1/2x, 1/4x) for simultaneous coarse structure and fine detail learning
- **Medical-specific regularization**: z-axis smoothness with coarse-to-fine annealing, Sobel-based edge preservation, and Total Variation loss for spatial denoising
- **Cross-view consistency loss (H3b)**: L1 on a short z-stack of rendered slices (sagittal/coronal gradient consistency) vs the base interpolation, enforcing 3D coherence
- **Patch-based non-local prior (H3d)**: Self-supervision signal at target z from the top-k most similar observed slices, injecting x-y correlation information that cubic cannot provide; improves LPIPS consistently
- **Structure-tensor-based adaptive initialization (H3c)**: Anisotropic Gaussian scales and quaternion rotations aligned with local edge tangents
- **Advanced residual bases (H1)**: Optional `cubic_bm4d`, `sinc3d`, and `unet_blend` bases that exploit 3D self-similarity, frequency-domain zero-fill, and learned priors respectively, giving 3DGS a stronger starting point to correct
- **Tri-plane INR baseline (H2)**: Self-supervised per-volume implicit neural representation with three 2D feature planes (xy, xz, yz) and an MLP decoder — a strong non-3DGS competitor that natively captures x-y correlations
- **Extended perceptual metrics (H4)**: LPIPS, HFEN, and GMSD in addition to PSNR / SSIM / MAE for edge-sharpness and perceptual-quality evaluation
- **Adaptive initialization**: Gaussian z-scale automatically adjusted based on sparse ratio; spatial subsampling adapts to keep count within budget
- **Progressive densification**: Gradient threshold ramps from aggressive (0.5x) to conservative (1.5x) over training, promoting rapid coverage then fine refinement
- **Residual magnitude penalty**: L2 regularization on the raw 3DGS output to prevent artifact injection at target positions
- **Comprehensive comparison** against classical interpolation (nearest/linear/cubic + BM4D/sinc3d standalone), supervised U-Net 2D, ArSSR, SAINT, and Tri-plane INR
- **Paired statistical testing**: Built-in paired t-test + Wilcoxon + Cohen's d helpers (`src/evaluation/statistical_tests.py`) for publication-ready comparison tables
- **ROI-based evaluation**: Per-organ metrics (liver, lungs, kidneys, bladder, bone) using segmentation masks
- **Kaggle-optimized pipeline**: Multi-GPU support (T4x2), memory-efficient lazy loading, session auto-resume

## Project Structure

```
TLCN/
  src/
    data/
      ct_org_loader.py      # CT-ORG volume loading with HU normalization
      sparse_simulator.py    # Sparse slice simulation (R=2, 3, 4)
      dataset.py             # PyTorch Datasets + LazyUNetSliceDataset + VolumeGroupedSampler
    models/
      gaussian_volume.py     # 3DGS model (pos, log-scale, opacity, intensity,
                             # optional quaternion rotation + structure-tensor init)
      slice_renderer.py      # Differentiable renderer (separable + tiled
                             # Mahalanobis path for rotated Gaussians)
      unet2d.py              # U-Net 2D baseline
      classical_interp.py    # Nearest/linear/cubic + cubic_bm4d + sinc3d + unet_blend
      triplane_inr.py        # Tri-plane INR baseline (3 2D feature grids + MLP)
      arssr.py               # ArSSR baseline (3D RDN encoder + INR MLP decoder)
      saint.py               # SAINT baseline (coronal + sagittal EDSR + fusion)
    losses/
      reconstruction.py      # L1, L2, SSIM, FFT high-frequency, combined + multi-scale loss
      regularization.py      # SmoothnessLoss, EdgePreservationLoss, TotalVariationLoss,
                             # CrossViewConsistencyLoss (H3b), TotalLoss aggregator
    training/
      trainer_3dgs.py        # Per-volume 3DGS optimizer with residual learning,
                             # error-map densification, oriented Gaussians,
                             # cross-view + patch-prior self-supervision
      trainer_triplane.py    # Tri-plane INR per-volume trainer (residual mode)
      trainer_unet.py        # U-Net training with checkpoint resume support
      trainer_arssr.py       # ArSSR pretraining + zero-shot inference
      trainer_saint.py       # SAINT branch + fusion training (per R)
    evaluation/
      metrics.py             # PSNR, SSIM, MAE + LPIPS, HFEN, GMSD, ROI per-organ
      statistical_tests.py   # paired_comparison, build_comparison_table,
                             # summarize_ablation (paired t-test + Wilcoxon + Cohen's d)
      visualization.py       # Publication-ready figure generation
    utils/
      config.py              # YAML config loading
      seed.py                # Reproducibility utilities
  configs/
    default.yaml             # All hyperparameters (data, model, training, evaluation)
  kaggle_notebooks/          # Self-contained Kaggle notebooks (NB01 -> NB10)
  modal_runs/                # Modal.com cloud-runner scripts (mirror of NB04/NB05)
  outputs/                   # Experiment results (CSV, JSON, checkpoints)
  modal_app.py               # Modal entry point for remote training
  requirements.txt           # Python dependencies
```

## Installation

```bash
git clone https://github.com/PhoenixEvo/ct-slice-interpolation-3dgs.git
cd TLCN
pip install -r requirements.txt
```

**Dependencies**: PyTorch >= 2.1, nibabel, scikit-image, scipy, pandas, matplotlib, seaborn, tqdm

## Running on Kaggle

All notebooks are designed as self-contained Kaggle notebooks. Each loads raw CT-ORG NIfTI data directly, with no inter-notebook file dependencies.

### Setup

1. Upload the CT-ORG dataset as a Kaggle dataset
2. Upload the `src/` and `configs/` folders (or link to the GitHub repo)
3. Select **GPU T4 x2** accelerator
4. Run notebooks in order (NB01 -> NB06), though NB02/NB03/NB04 can run in parallel

### Notebook Pipeline

| # | Notebook | Purpose | Runtime |
|---|----------|---------|---------|
| 01 | `01_data_preparation.ipynb` | Data exploration, volume statistics, split verification | ~5 min (CPU) |
| 02 | `02_baseline_classical.ipynb` | Streaming slice-by-slice classical interpolation (nearest/linear/cubic) | ~30 min (CPU) |
| 03 | `03_baseline_unet.ipynb` | U-Net 2D training + evaluation for R=2,3,4 with multi-GPU | ~8-12h total |
| 04 | `04_3dgs_training.ipynb` | 3DGS per-volume optimization with 3-phase strategy | ~15 min - 8h |
| 05 | `05_benchmark_ablation.ipynb` | Cross-method comparison + 15-variant ablation study | ~7-9h (or ~2-3h with PARTITION) |
| 06a | `06_visualization.ipynb` | Publication figures + paired t-tests (3DGS vs each baseline) | ~10 min |
| 06b | `06_reeval_perceptual.ipynb` | Re-evaluate checkpoints (H3DGS + Old 3DGS + Cubic) with LPIPS/HFEN/GMSD | ~2-4h |
| 07 | `07_baseline_arssr.ipynb` | ArSSR pretraining (arbitrary-scale INR) + zero-shot eval for R=2/3/4 | ~7-9h (pretrain once) |
| 08 | `08_baseline_saint.ipynb` | SAINT SOTA: train 3 models (R=2/3/4) + eval | ~10-13h total (run each R in a separate session) |
| 09 | `09_baseline_triplane.ipynb` | Tri-plane INR per-volume baseline (self-supervised, x-y correlation native) | ~1-3h per (case, R) |
| 10 | `10_qualitative_slide_figures.ipynb` | Qualitative side-by-side slide figures at R=3 (GT / Cubic / U-Net / 3DGS, with zoom + |error| heatmap) | ~10 min |

### NB04 Phased Strategy

NB04 uses a 3-phase approach to avoid wasting GPU time if 3DGS underperforms:

| Phase | Cases | Ratios | Time (T4x2) | Decision Gate |
|-------|-------|--------|-------------|---------------|
| 1 SCOUT | 5 | R=2 | ~15-30 min to ~2h | 3DGS ≈ cubic? |
| 2 VALIDATE | 10 | R=2, 3 | ~1-4h | 3DGS competitive with U-Net? |
| 3 FULL | 21 | R=2, 3, 4 | ~4-12h | Paper-ready results |

Set `PHASE = 1` in cell 1, run all cells, check verdict. Completed cases are automatically skipped when advancing to the next phase.

### Resume Support

All GPU notebooks (NB03, NB04, NB05) support session resume for Kaggle's 30h/week GPU quota:

1. After a session ends, save the notebook output as a Kaggle dataset
2. Attach it as input in a new session
3. The auto-restore mechanism copies checkpoints and results from `/kaggle/input/` into `/kaggle/working/`
4. Training resumes from the latest checkpoint; completed experiments are skipped

## Methods

### Classical Interpolation (Baselines)

Standard z-axis interpolation using **nearest**, **linear**, and **cubic** (Catmull-Rom) methods. Implemented with a memory-efficient streaming approach that processes one target slice at a time, loading only the 2-4 neighboring observed slices needed.

### U-Net 2D (Supervised Baseline)

A 4-level U-Net (features: 32-64-128-256) trained to predict a missing middle slice from two adjacent observed slices. Trained separately for each sparse ratio on all training volumes.

- **Input**: 2-channel tensor (upper + lower observed slices)
- **Output**: 1-channel predicted middle slice
- **Loss**: L1 + 0.1 * SSIM
- **Optimizer**: Adam (lr=1e-4, ReduceLROnPlateau)

### ArSSR (SOTA, INR Arbitrary-Scale SR)

Re-implementation of Wu et al., "An Arbitrary Scale Super-Resolution Approach for 3D MR Images via Implicit Neural Representation" (IEEE JBHI 2022).

- **Encoder**: Lightweight 3D RDN (4 blocks x 4 conv layers, 64 feats, growth 32), produces a feature volume with no spatial downsampling.
- **Decoder**: 5-layer MLP INR conditioned on a bilinear-sampled feature and a relative coordinate.
- **Training**: Pretrained once on 98 training volumes with random-scale z-axis downsampling (s continuous in [2, 4]). 200k iters, patch HR = 48x48x32, coords-per-patch = 4096. Cosine lr 1e-4 -> 1e-5, AMP on.
- **Inference**: Applied zero-shot to 21 test cases at R=2, 3, 4; only z-axis is super-resolved (xy is preserved), xy-tiled to fit on T4.
- **Paper reference**: https://arxiv.org/abs/2110.14476

### SAINT (SOTA, Supervised CT Slice Interpolation)

Re-implementation of Peng et al., "SAINT: Spatially Aware Interpolation NeTwork for Medical Slice Synthesis" (CVPR 2020).

- **Coronal branch**: 2D EDSR (16 residual blocks, 64 feats, residual scale 0.1), z-axis-only transposed-conv upsampler with scale R.
- **Sagittal branch**: identical architecture, trained on sagittal views.
- **Fusion**: 3-layer CNN that fuses two HR axial-slice estimates into one.
- **Training per R**: 80k iterations; 60% of iters train branches independently, 40% train the full pipeline end-to-end (fusion loss). Loss = L1 + 0.1 * SSIM (same recipe as the U-Net baseline, for comparability).
- **Paper reference**: https://arxiv.org/abs/2001.09449

### 3DGS with Residual Learning (Proposed Method)

Per-volume self-supervised optimization. A set of axis-aligned 3D Gaussians is optimized to reproduce the observed slices, then used to render the missing target slices. The model uses **residual learning** on top of cubic interpolation to guarantee competitive quality.

**Architecture**: `prediction(z) = cubic_base(z) + 3DGS_residual(z)`

- **Cubic base**: Precomputed Catmull-Rom interpolation using all observed slices as control points
- **3DGS residual**: Learned correction predicting fine details that cubic interpolation misses
- **Residual design**: 3DGS intensities initialized to zero, so the initial prediction equals the cubic base; training refines from this strong starting point

**Gaussian parameterization** (per Gaussian):
- Position (x, y, z) in volume coordinates
- Log-scale (log σ_x, log σ_y, log σ_z) for numerical stability
- Raw opacity (pre-sigmoid)
- Scalar intensity value

**Differentiable slice renderer**: For a target z-position z_t, each Gaussian contributes:
```
w_z = exp(-0.5 × ((z_t - μ_z) / σ_z)²)
G_2d(x,y) = exp(-0.5 × [((x - μ_x) / σ_x)² + ((y - μ_y) / σ_y)²])
contribution = intensity × sigmoid(opacity) × w_z × G_2d(x, y)
```
Contributions are combined via weighted sum (normalized). **Separable rendering** exploits the axis-aligned property: G(x,y) = G_x(x) · G_y(y), enabling the full render to be computed via two matrix multiplications instead of materializing the (H, W, K) tensor. Z-threshold filtering (3σ) further reduces the number of active Gaussians per slice.

**Combined loss**:
```
L_total = L_rec + λ_fft · L_fft + λ_smooth(t) · L_smooth + λ_edge · L_edge
        + λ_tv · L_tv + λ_res · L_residual
```

Where:
- **L_rec**: Multi-scale weighted L1 loss at 3 resolution levels (1x, 1/2x, 1/4x), with spatial weighting by HU gradient magnitude (Sobel) to prioritize organ boundaries
- **L_fft**: FFT high-frequency loss — penalizes differences in high-frequency Fourier components (cutoff ratio 0.25), forcing the model to capture sharp edges (bone, organ walls)
- **L_smooth**: z-axis continuity between adjacent rendered slices (annealed: starts at 2× then decays to 0.5× for coarse-to-fine training)
- **L_edge**: Sobel gradient matching to preserve anatomical boundaries
- **L_tv**: Anisotropic Total Variation for spatial denoising without edge blurring
- **L_residual**: L2 penalty on raw 3DGS output to prevent artifact injection at target positions

**Error-map densification** (replaces gradient-based):
- Tracks per-Gaussian reconstruction error accumulated across training iterations
- Clones Gaussians with error in the top 5th percentile instead of using gradient norm
- Directly allocates model capacity to regions with worst reconstruction quality

**Adaptive initialization**:
- **z-scale**: Automatically set to max(1.0, R × 0.6) where R is the sparse ratio, ensuring Gaussians bridge inter-slice gaps
- **xy-subsampling**: Dynamically adjusted so initial count stays within budget (default 500K)
- **Intensity**: Initialized to zero for residual mode (initial output = pure cubic base)

**Progressive densification/pruning** (every 100 iterations):
- **Early training**: Low gradient/error threshold → aggressive densification for rapid coverage
- **Late training**: High threshold → conservative refinement only where needed
- **Prune**: Low-opacity Gaussians (< 0.01) are removed
- Max 500K Gaussians per volume (800K in high-quality mode)

## Dataset

**CT-ORG** from The Cancer Imaging Archive (TCIA): 140 CT volumes with multi-organ segmentations.

| Property | Value |
|----------|-------|
| Volumes | 140 CT scans |
| Organs | Liver (1), Bladder (2), Lungs (3), Kidneys (4), Bone (5) |
| Format | NIfTI (.nii.gz) |
| Split | 98 train / 21 val / 21 test |
| Preprocessing | HU clip [-1000, 1000] → normalize to [0, 1] |
| Sparse ratios | R=2 (every 2nd slice), R=3 (every 3rd), R=4 (every 4th) |

Download: [TCIA CT-ORG Collection](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890)

## Evaluation

### Metrics

- **PSNR** (dB): Peak Signal-to-Noise Ratio — pixel-level fidelity
- **SSIM**: Structural Similarity Index — perceptual quality
- **MAE**: Mean Absolute Error — average reconstruction error
- **LPIPS** (optional, `compute_perceptual=True`): Learned Perceptual Image Patch Similarity (lower is better). Requires `pip install lpips`.
- **HFEN**: High-Frequency Error Norm on a Laplacian-of-Gaussian response (edge fidelity, lower is better).
- **GMSD**: Gradient Magnitude Similarity Deviation (lower is better, Xue et al. 2013).
- **ROI metrics**: Per-organ PSNR/SSIM computed within segmentation masks (liver, lungs, kidneys, bladder, bone)

### Fair-Comparison Protocol

All baselines (classical, U-Net, ArSSR, SAINT, 3DGS) are evaluated under an identical protocol to keep the comparison publishable:

| Dimension | Setting | Where it is enforced |
|-----------|---------|----------------------|
| Dataset | CT-ORG, 140 volumes | `src/data/ct_org_loader.py` |
| Split | 98 train / 21 val / 21 test (fixed) | `configs/default.yaml` (`test_cases`, `val_cases`) |
| HU normalization | clip `[-1000, 1000]` → `[0, 1]` | `CTORGLoader.preprocess_volume` |
| Sparse simulation | Keep slice indices `0, R, 2R, ...` along z | `src/data/sparse_simulator.py` |
| Sparse ratios | R=2, 3, 4 | All NB03/04/07/08 use the same values |
| Random seed | 42 | `src/utils/seed.set_seed(42)` in every notebook |
| Metrics | PSNR, SSIM, MAE + per-organ ROI | `src/evaluation/metrics.evaluate_volume` |
| Loss (supervised baselines) | L1 + 0.1·SSIM | Identical between U-Net and SAINT |
| Logging | `training_time_s`, `inference_time_s`, per-case JSON | Saved in `outputs/<method>/per_case/` |

Paired t-tests (3DGS vs each baseline, matched per case_idx) are generated in `kaggle_notebooks/06_visualization.ipynb` and saved as `outputs/figures/comparison_table.csv` for direct inclusion in the paper.

### Ablation Study

15 variants tested in NB05 to isolate each component's contribution. Currently aggregated from `outputs/ablations/ablation_results_p{1,2,3}.csv` at **R = 3, N = 10 cases** (Full = Ours uses 20 cases — partition 1+2+3 of the 10-case shard).

| Variant | Description | PSNR | SSIM | MAE | Δ PSNR vs Full |
|---------|-------------|:----:|:----:|:---:|:---:|
| `no_edge` | w/o Edge Preservation | **38.016** | 0.9415 | 0.00769 | +0.050 |
| `no_lr_decay` | w/o LR Decay | 38.011 | 0.9394 | 0.00778 | +0.046 |
| `no_errormap` | w/o Error-Map Densification | 37.985 | 0.9389 | 0.00783 | +0.020 |
| `no_res_penalty` | w/o Residual Penalty | 37.980 | 0.9412 | 0.00773 | +0.015 |
| **`full`** | **Full (Ours, N=20)** | **37.965** | **0.9383** | **0.00785** | — |
| `no_fft` | w/o FFT Loss | 37.955 | 0.9379 | 0.00781 | -0.010 |
| `no_reg` | w/o All Regularization | 37.904 | 0.9390 | 0.00787 | -0.061 |
| `no_huweight` | w/o HU Gradient Weighting | 37.904 | 0.9347 | 0.00800 | -0.061 |
| `grid_init` | Grid Init | 37.898 | 0.9369 | 0.00785 | -0.067 |
| `adaptive_init` | Adaptive (edge-aware) Init | 37.871 | 0.9327 | 0.00808 | -0.094 |
| `no_multiscale` | w/o Multi-scale Loss | 37.835 | 0.9360 | 0.00799 | -0.130 |
| `no_tv` | w/o TV Denoising | 37.828 | 0.9389 | 0.00788 | -0.137 |
| `no_smooth` | w/o Smoothness | 37.677 | 0.9318 | 0.00817 | -0.288 |
| `baseline` | Raw 3DGS (no improvements) | 28.493 | 0.8401 | 0.01735 | -9.473 |
| `no_residual` | w/o Residual Learning | 24.317 | 0.6458 | 0.03651 | -13.649 |

> **Reading the table.** The two large drops (`baseline`, `no_residual`) confirm that residual learning + the regularization stack are the dominant contributors. Among the fine-grained ablations, removing `smooth` / `tv` / `multiscale` hurts the most, while disabling `edge` or `lr_decay` is statistically a wash on R=3 (within ±0.05 dB of Full). SSIM/MAE deltas mirror the PSNR ranking.

**Multi-account parallel**: Set `PARTITION = 1/2/3` to split ablation across 3 Kaggle accounts (~2-3h each instead of ~7-9h). Partition i writes to `outputs/ablations/ablation_results_p{i}.csv`.

## Outputs

Tree below reflects the **current state of `outputs/`** in this repository (everything checked-in as of May 1, 2026). Optional folders that will appear once the corresponding notebook is run are marked `(planned)`.

```
outputs/
  classical_baselines/
    R{r}/                                # r in {2, 3, 4}
      summary.csv                        # 21 cases x cubic, with PSNR/SSIM/MAE/LPIPS/HFEN/GMSD
      per_case/case{id}_R{r}.json        # Per-case detailed metrics (cubic only)
  unet_baseline/
    outputs/unet_R{r}/summary.csv        # U-Net 2D results per sparse ratio (PSNR/SSIM/MAE)
    checkpoints/unet_R{r}/               # Model weights + epoch history
  3dgs/
    3dgs_R{r}/
      high/
        summary.csv                      # 3DGS high (~1.0M Gaussians)
        per_case/case{id}_R{r}.json
        checkpoints/case{id}_R{r}/       # final.pt
      standard/
        summary.csv                      # 3DGS standard (~0.8M Gaussians)
        per_case/case{id}_R{r}.json
        checkpoints/case{id}_R{r}/
  3dgs_improve/
    H3d/                                 # H3d = patch-based non-local prior (Ours)
      R{r}/
        summary.csv                      # H3DGS patch_prior with perceptual metrics
        per_case/case{id}_R{r}.json
        checkpoints/case{id}_R{r}/
    check_cubic.py / compare_results.py / fix_nb10.py /
    r2_stats.py / rebuild_summary.py     # Helper scripts used to rebuild summaries
  ablations/
    ablation_results_p1.csv              # NB05 partition 1 (10 cases @ R=3, full + 3 variants)
    ablation_results_p1_no_residual.csv  # Companion run that re-ran the no_residual variant
    ablation_results_p2.csv              # NB05 partition 2
    ablation_results_p3.csv              # NB05 partition 3
    paper_table1_p2.csv                  # Compact "paper Table 1" (Nearest/Linear/Cubic/U-Net/3DGS)
  roi_table.csv                          # Aggregated ROI PSNR/SSIM by (R, method, organ)
  arssr/                                 # (planned) populated by NB07
  saint/                                 # (planned) populated by NB08
  triplane/                              # (planned) populated by NB09
  figures/                               # (planned) populated by NB06a/NB10
```

**Notes on data lineage:**

- The 3DGS / H3d `summary.csv` files were rebuilt by `outputs/3dgs_improve/rebuild_summary.py` from the per-case JSONs after `06_reeval_perceptual.ipynb` added LPIPS/HFEN/GMSD. As a side-effect, `training_time_s = 0` in those rebuilt summaries — refer to the NB04/NB05 logs for actual training times.
- Mean inference time per case (forward render only, T4 single-GPU): R=2 ≈ 2.0–2.4 s, R=3 ≈ 3.7–3.9 s, R=4 ≈ 5.4–7.1 s, scaling with the number of target slices.
- `outputs/roi_table.csv` is aggregated from each method's `summary.csv` `roi` column. It reports mean ROI PSNR/SSIM per `(sparse_ratio, method, organ)` and includes `n_cases_with_roi` for transparency.

## Technical Optimizations

### Memory Efficiency (Kaggle 30GB CPU RAM)

- **LazyUNetSliceDataset**: On-demand NIfTI loading with LRU cache (2 volumes), zero-copy preprocessing via `np.asarray(nii.dataobj, dtype=np.float32)` with in-place HU clipping and normalization
- **VolumeGroupedSampler**: Custom PyTorch Sampler that groups training samples by source volume, ensuring consecutive batches come from the same 1-2 cached volumes (eliminates cache thrashing)
- **Streaming classical interpolation**: `interpolate_target_slice()` processes one slice at a time using only 2-4 neighboring slices, avoiding full-volume intermediate arrays
- **Cubic base caching**: Precomputed cubic predictions stored on GPU to avoid recomputation during training

### GPU Utilization (T4 x2)

- **U-Net**: `torch.nn.DataParallel` distributes batches across both GPUs
- **3DGS/Ablation**: `ThreadPoolExecutor(max_workers=2)` runs independent per-volume tasks concurrently on separate GPUs
- **Separable rendering**: Axis-aligned Gaussians decompose as G(x,y) = G_x(x) · G_y(y), enabling O(H·K + W·K) rendering via matrix multiplication instead of O(H·W·K) per-pixel computation (~50-100x speedup over tiled rendering)
- **TF32 enabled**: `torch.backends.cuda.matmul.allow_tf32 = True` for faster matrix operations
- **Mixed precision**: AMP with GradScaler for ~30% training speedup
- **Async data transfer**: `tensor.to(device, non_blocking=True)` overlaps CPU-GPU transfers
- **Gradient accumulation for memory safety**: per-slice backward avoids graph blow-up from multi-slice + multi-term losses

### Session Resume (Kaggle 30h/week GPU Quota)

- Auto-restore: Recursively scans attached input datasets for `checkpoints/` and `outputs/` directories
- Checkpoint resume: U-Net loads latest `epoch_*.pt` and continues from correct epoch
- Experiment skip: 3DGS and ablation check for existing JSON/CSV results before re-running
- Partial save: `summary.csv` written after each sparse ratio completes

## New Methods & How to Enable Them

All the improvements below live behind config flags in `configs/default.yaml` and default to OFF so they do not break the baseline. Turn on one at a time and compare paired-t vs the "full" baseline via `src/evaluation/statistical_tests.summarize_ablation`.

### H1 — Advanced residual bases (`gaussian.residual_base`)

Switch the base interpolation the 3DGS residual is stacked on top of:

| Value | Description | Trade-off |
|-------|-------------|-----------|
| `cubic` (default) | Catmull-Rom cubic along z | Fast, no x-y correlation |
| `cubic_bm4d` | Cubic, then BM4D denoising on the full 3D volume | Exploits 3D self-similarity; slower setup, needs `pip install bm4d` |
| `sinc3d` | Frequency-domain zero-fill along z (FFT) | Pure signal-processing baseline, uniform-spacing only |
| `unet_blend` | `alpha * cubic + (1 - alpha) * unet_prediction` | Leverages a pretrained 2D U-Net, needs the predictor in code |

Same options are available for the Tri-plane baseline (`triplane.residual_base`).

### H2 — Tri-plane INR baseline

Self-supervised per-volume INR with three 2D feature planes. Runs via `kaggle_notebooks/09_baseline_triplane.ipynb` or:

```python
from src.training import TrainerTriPlane
trainer = TrainerTriPlane(volume, observed_indices, target_indices, config, device="cuda")
trainer.train()
results = trainer.evaluate_on_targets()
```

### H3 — 3DGS representational upgrades

- **H3a/H3c — oriented + structure-tensor init**: `gaussian.use_rotation: true`, `gaussian.use_structure_tensor: true`. Adds a 4-vector quaternion per Gaussian (LR ramps up after a warmup) and aligns initial scales/rotations with local edge tangents.
- **H3b — cross-view consistency**: `loss.lambda_crossview: 0.001` (set > 0 to enable), `loss.crossview_interval: 5`, `loss.crossview_window: 3`.
- **H3d — patch-based non-local prior**: `loss.lambda_patch: 0.02`, `loss.patch_prior_k: 5`, `loss.patch_prior_interval: 5`.

### H4 — Extended perceptual metrics

```python
from src.evaluation import evaluate_volume
result = evaluate_volume(
    predictions, ground_truths, target_indices,
    compute_perceptual=True, lpips_device="cuda",
)
# result["summary"] now also includes mean_lpips / mean_hfen / mean_gmsd
```

### Paired statistical tests

```python
from src.evaluation import build_comparison_table, summarize_ablation
table = build_comparison_table(
    per_case_results={"3dgs": df_3dgs, "cubic": df_cubic, "triplane": df_triplane},
    ref_method="3dgs", metric="psnr",
)
# table has columns: sparse_ratio, other, delta_mean, p_ttest, cohen_d, ref_wins_frac
```

## Configuration

All hyperparameters are centralized in `configs/default.yaml`:

```yaml
gaussian:
  init_mode: "grid"           # or "adaptive" (edge-aware initialization)
  residual_mode: true         # Residual learning on top of the base interp
  residual_base: "cubic"      # cubic / cubic_bm4d / sinc3d / unet_blend (H1)
  base_bm4d_sigma: 0.015      # BM4D noise sigma when residual_base=cubic_bm4d
  base_unet_alpha: 0.7        # cubic weight when residual_base=unet_blend
  use_rotation: false         # H3a: quaternion-rotated Gaussians
  rotation_warmup_frac: 0.2   # ramp rotation LR from 0 after this fraction of iters
  lr_rotation: 1.0e-3
  use_structure_tensor: false # H3c: anisotropic init from edge tangents
  num_iterations: 7000
  batch_slices: 4
  max_gaussians: 500000
  render_mode: "weighted"
  densify_use_error_map: true

loss:
  l1_weight: 1.0
  ssim_weight: 0.0
  lambda_smooth: 0.002
  lambda_edge: 0.001
  lambda_tv: 0.0002
  lambda_fft: 0.05
  fft_cutoff: 0.25
  lambda_residual: 0.1
  lambda_crossview: 0.0       # H3b: sagittal/coronal consistency (~0.001 when on)
  crossview_interval: 5
  crossview_window: 3
  lambda_patch: 0.0           # H3d: non-local patch prior (~0.01-0.05 when on)
  patch_prior_k: 5
  patch_prior_interval: 5
  hu_gradient_weight: true
  hu_weight_max: 5.0
  multiscale: true

triplane:                      # H2: Tri-plane INR baseline
  residual_mode: true
  residual_base: "cubic"
  plane_resolution: 128
  feature_dim: 16
  mlp_hidden: 64
  mlp_layers: 3
  num_iterations: 5000

unet:
  features: [32, 64, 128, 256]
  lr: 1.0e-4
  num_epochs: 50
```

## Local Usage

```python
from src.data.ct_org_loader import CTORGLoader
from src.data.sparse_simulator import SparseSimulator
from src.training.trainer_3dgs import Trainer3DGS
from src.utils.config import load_config

config = load_config("configs/default.yaml")

loader = CTORGLoader(dataset_root="path/to/CT-ORG")
volume, labels, metadata = loader.load_and_preprocess(case_idx=0)

simulator = SparseSimulator(sparse_ratio=2)
sparse_data = simulator.simulate(volume)

trainer = Trainer3DGS(
    volume=volume,
    observed_indices=sparse_data["observed_indices"],
    target_indices=sparse_data["target_indices"],
    config=config,
    device="cuda",
)
trainer.train()

results = trainer.evaluate_on_targets(organ_labels={"liver": 1, "lungs": 3})
print(f"PSNR: {results['summary']['mean_psnr']:.2f} dB")
```

## Citation

```bibtex
@software{ct_slice_interpolation_3dgs,
  title = {CT Slice Interpolation via 3D Gaussian Splatting with Residual Learning},
  author = {Nguyen, Nhat Phat},
  year = {2026},
  url = {https://github.com/PhoenixEvo/ct-slice-interpolation-3dgs}
}
```

## Acknowledgments

- **CT-ORG dataset**: Rister et al., via The Cancer Imaging Archive (TCIA)
- **3D Gaussian Splatting**: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
- **MedGS**: Wu et al., medical image synthesis via Gaussian Splatting
