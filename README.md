# CT Slice Interpolation via 3D Gaussian Splatting

Research project applying **3D Gaussian Splatting (3DGS)** to the problem of CT slice interpolation on the **CT-ORG** dataset. This work demonstrates that per-volume 3DGS optimization can reconstruct missing CT slices competitively with supervised deep learning methods, without requiring any training data.

## Problem Statement

CT and MRI volumes typically have high in-plane (x-y) resolution but significantly lower through-plane (z-axis) resolution due to slice thickness and acquisition gaps. This anisotropy degrades 3D reconstruction quality and downstream clinical tasks (segmentation, volumetric analysis). Traditional solutions involve either reacquiring scans at higher resolution (more radiation dose, longer scan time) or applying post-hoc interpolation.

This project explores whether **3D Gaussian Splatting** -- originally developed for novel view synthesis in computer vision -- can serve as a self-supervised, per-volume optimization method for CT slice interpolation that outperforms classical approaches and competes with supervised deep learning baselines.

## Key Contributions

- **Custom 3DGS pipeline** adapted for axis-aligned medical slice rendering (no rotation parameters, reducing model complexity)
- **Separable differentiable rendering**: Exploits axis-aligned Gaussian structure for O(H·K + W·K) rendering via matrix multiplication, ~50-100x faster than naive per-pixel computation
- **Multi-scale reconstruction loss**: Computes L1 + SSIM at 3 resolution levels (1x, 1/2x, 1/4x) for simultaneous coarse structure and fine detail learning
- **Medical-specific regularization**: z-axis smoothness with coarse-to-fine annealing, Sobel-based edge preservation, and Total Variation loss for spatial denoising
- **Adaptive initialization**: Gaussian z-scale automatically adjusted based on sparse ratio; spatial subsampling adapts to keep count within budget
- **Progressive densification**: Gradient threshold ramps from aggressive (0.5x) to conservative (1.5x) over training, promoting rapid coverage then fine refinement
- **Comprehensive comparison** against classical interpolation (nearest/linear/cubic) and supervised U-Net 2D
- **ROI-based evaluation**: Per-organ metrics (liver, lungs, kidneys, bladder, bone) using segmentation masks
- **Ablation studies**: Quantifying the impact of each regularization and rendering component
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
      gaussian_volume.py     # 3DGS model (position, log-scale, opacity, intensity)
      slice_renderer.py      # Differentiable tile-based slice renderer
      unet2d.py              # U-Net 2D baseline
      classical_interp.py    # Nearest/linear/cubic + streaming slice-by-slice mode
    losses/
      reconstruction.py      # L1, L2, SSIM, combined + multi-scale reconstruction loss
      regularization.py      # SmoothnessLoss, EdgePreservationLoss, TotalVariationLoss
    training/
      trainer_3dgs.py        # Per-volume 3DGS optimizer with densification/pruning
      trainer_unet.py        # U-Net training with checkpoint resume support
    evaluation/
      metrics.py             # PSNR, SSIM, ROI-based per-organ metrics
      visualization.py       # Publication-ready figure generation
    utils/
      config.py              # YAML config loading
      seed.py                # Reproducibility utilities
  configs/
    default.yaml             # All hyperparameters (data, model, training, evaluation)
  kaggle_notebooks/          # Self-contained Kaggle notebooks (6 notebooks)
  outputs/                   # Experiment results (CSV, JSON, figures)
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
| 04 | `04_3dgs_training.ipynb` | 3DGS per-volume optimization with 3-phase strategy | ~5 min - 2h |
| 05 | `05_benchmark_ablation.ipynb` | Cross-method comparison + ablation (regularization variants) | ~1-3h |
| 06 | `06_visualization.ipynb` | Publication figures: bar charts, slice comparisons, per-organ plots | ~10 min |

### NB04 Phased Strategy

NB04 uses a 3-phase approach to avoid wasting GPU time if 3DGS underperforms:

| Phase | Cases | Ratios | Time (T4x2) | Decision Gate |
|-------|-------|--------|-------------|---------------|
| 1 SCOUT | 5 | R=2 | ~3-5 min | 3DGS > cubic by 1+ dB? |
| 2 VALIDATE | 10 | R=2, 3 | ~10-15 min | 3DGS competitive with U-Net? |
| 3 FULL | 21 | R=2, 3, 4 | ~1-2h | Paper-ready results |

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

### 3DGS (Proposed Method)

Per-volume self-supervised optimization. A set of axis-aligned 3D Gaussians is optimized to reproduce the observed slices, then used to render the missing target slices.

**Gaussian parameterization** (per Gaussian):
- Position (x, y, z) in volume coordinates
- Log-scale (log σ_x, log σ_y, log σ_z) for numerical stability
- Raw opacity (pre-sigmoid)
- Scalar intensity value

**Differentiable slice renderer**: For a target z-position z_t, each Gaussian contributes:
```
w_z = exp(-0.5 * ((z_t - μ_z) / σ_z)²)
G_2d(x,y) = exp(-0.5 * [((x - μ_x) / σ_x)² + ((y - μ_y) / σ_y)²])
contribution = intensity × sigmoid(opacity) × w_z × G_2d(x, y)
```
Contributions are combined via weighted sum (normalized). **Separable rendering** exploits the axis-aligned property: G(x,y) = G_x(x) · G_y(y), enabling the full render to be computed via two matrix multiplications instead of materializing the (H, W, K) tensor. Z-threshold filtering (3σ) further reduces the number of active Gaussians per slice.

**Combined loss** (multi-scale with regularization annealing):
```
L_rec = Σ_{s=0}^{2} w_s · [L1(pred_s, gt_s) + 0.1 · SSIM(pred_s, gt_s)]
L = L_rec + λ_smooth(t) · L_smooth + 0.005 · L_edge + 0.001 · L_tv
```
Where:
- **L_rec**: Multi-scale reconstruction loss at 3 resolution levels (1x, 1/2x, 1/4x), ensuring both coarse anatomy and fine detail are captured
- **L_smooth**: z-axis continuity between adjacent rendered slices (annealed: starts at 2× then decays to 0.5× for coarse-to-fine training)
- **L_edge**: Sobel gradient matching to preserve anatomical boundaries (organ edges, bone margins)
- **L_tv**: Anisotropic Total Variation for spatial denoising without edge blurring

**Adaptive initialization**:
- **z-scale**: Automatically set to max(1.0, R × 0.6) where R is the sparse ratio, ensuring Gaussians bridge inter-slice gaps
- **xy-subsampling**: Dynamically adjusted so initial count stays within budget (default 500K)

**Progressive densification/pruning** (every 100 iterations):
- **Early training**: Low gradient threshold (0.5× base) → aggressive densification for rapid coverage
- **Late training**: High gradient threshold (1.5× base) → conservative refinement only where needed
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

- **PSNR** (dB): Peak Signal-to-Noise Ratio -- pixel-level fidelity
- **SSIM**: Structural Similarity Index -- perceptual quality
- **MAE**: Mean Absolute Error -- average reconstruction error
- **ROI metrics**: Per-organ PSNR/SSIM computed within segmentation masks (liver, lungs, kidneys, bladder, bone)

### Ablation Study

Variants tested in NB05 to isolate the contribution of each component:

| Variant | Multi-scale | Smoothness | Edge | TV | Annealing | Purpose |
|---------|:-----------:|:----------:|:----:|:--:|:---------:|---------|
| full | yes | yes | yes | yes | yes | Complete model |
| no_multiscale | no | yes | yes | yes | yes | Effect of multi-scale loss |
| no_smooth | yes | no | yes | yes | yes | Effect of z-axis smoothness |
| no_edge | yes | yes | no | yes | yes | Effect of edge preservation |
| no_tv | yes | yes | yes | no | yes | Effect of TV denoising |
| no_reg | no | no | no | no | no | Raw 3DGS baseline |

## Outputs

```
outputs/
  classical_baselines/
    summary.csv                   # All classical results
    per_case/case{id}_R{r}.json   # Per-case detailed metrics
  unet_baseline/
    outputs/unet_R{r}/summary.csv # U-Net results per sparse ratio
    checkpoints/unet_R{r}/        # Model weights, history
  3dgs/
    summary.csv                   # All 3DGS results
    per_case/case{id}_R{r}.json   # Per-case detailed metrics + Gaussian stats
  figures/                        # Publication-ready visualizations
```

## Technical Optimizations

### Memory Efficiency (Kaggle 30GB CPU RAM)

- **LazyUNetSliceDataset**: On-demand NIfTI loading with LRU cache (2 volumes), zero-copy preprocessing via `np.asarray(nii.dataobj, dtype=np.float32)` with in-place HU clipping and normalization
- **VolumeGroupedSampler**: Custom PyTorch Sampler that groups training samples by source volume, ensuring consecutive batches come from the same 1-2 cached volumes (eliminates cache thrashing)
- **Streaming classical interpolation**: `interpolate_target_slice()` processes one slice at a time using only 2-4 neighboring slices, avoiding full-volume intermediate arrays

### GPU Utilization (T4 x2)

- **U-Net**: `torch.nn.DataParallel` distributes batches across both GPUs
- **3DGS/Ablation**: `ThreadPoolExecutor(max_workers=2)` runs independent per-volume tasks concurrently on separate GPUs
- **Separable rendering**: Axis-aligned Gaussians decompose as G(x,y) = G_x(x) · G_y(y), enabling O(H·K + W·K) rendering via matrix multiplication instead of O(H·W·K) per-pixel computation (~50-100x speedup over tiled rendering)
- **TF32 enabled**: `torch.backends.cuda.matmul.allow_tf32 = True` for faster matrix operations
- **cudnn.benchmark**: Auto-selects optimal convolution algorithms
- **Async data transfer**: `tensor.to(device, non_blocking=True)` overlaps CPU-GPU transfers

### Session Resume (Kaggle 30h/week GPU Quota)

- Auto-restore: Recursively scans attached input datasets for `checkpoints/` and `outputs/` directories
- Checkpoint resume: U-Net loads latest `epoch_*.pt` and continues from correct epoch
- Experiment skip: 3DGS and ablation check for existing JSON/CSV results before re-running
- Partial save: `summary.csv` written after each sparse ratio completes

## Configuration

All hyperparameters are centralized in `configs/default.yaml`:

```yaml
gaussian:
  init_mode: "grid"          # or "adaptive" (edge-aware initialization)
  num_iterations: 5000       # 10000+ for high quality
  batch_slices: 4            # Multi-slice batch training
  max_gaussians: 500000
  render_mode: "weighted"    # Separable rendering via matmul

loss:
  l1_weight: 1.0
  ssim_weight: 0.1
  lambda_smooth: 0.01       # Annealed: 2x -> 0.5x during training
  lambda_edge: 0.005
  lambda_tv: 0.001           # Total variation for spatial denoising
  multiscale: true           # Multi-scale loss (1x, 1/2x, 1/4x)

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
  title = {CT Slice Interpolation via 3D Gaussian Splatting},
  author = {Nguyen, Nhat Phat},
  year = {2026},
  url = {https://github.com/PhoenixEvo/ct-slice-interpolation-3dgs}
}
```

## Acknowledgments

- **CT-ORG dataset**: Rister et al., via The Cancer Imaging Archive (TCIA)
- **3D Gaussian Splatting**: Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
- **MedGS**: Wu et al., medical image synthesis via Gaussian Splatting
