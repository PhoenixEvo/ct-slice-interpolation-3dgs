# CT Slice Interpolation via 3D Gaussian Splatting

Research project for CT slice interpolation using 3D Gaussian Splatting (3DGS) on the CT-ORG dataset.

## Overview

CT/MRI volumes often have high in-plane resolution but low through-plane (z-axis) resolution, resulting in anisotropic volumes. This project applies 3D Gaussian Splatting to interpolate missing slices, improving z-axis resolution without additional scanning.

## Features

- **Custom 3DGS implementation** optimized for axis-aligned CT slice rendering
- **Medical-specific regularization**: Smoothness along z-axis and edge preservation
- **Comprehensive baselines**: Classical interpolation (nearest/linear/cubic) and U-Net 2D
- **Full evaluation pipeline**: PSNR, SSIM, ROI-based metrics, ablation studies
- **Google Colab ready**: All notebooks configured for L4 GPU with Drive mounting

## Project Structure

```
TLCN/
  src/                    # Python source package
    data/                 # Data loading, preprocessing, sparse simulation
    models/               # 3DGS model, U-Net baseline, classical interpolation
    losses/               # Reconstruction + regularization losses
    training/             # Training loops for 3DGS and U-Net
    evaluation/           # Metrics (PSNR, SSIM, ROI) and visualization
    utils/                # Config management, seed utilities
  configs/                # YAML configuration files
  notebooks/              # Colab notebooks for experiments
  requirements.txt        # Python dependencies
```

## Installation

### Local Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd TLCN

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Setup

1. Upload the project folder to Google Drive at `MyDrive/TLCN/`
2. Ensure CT-ORG dataset is at `MyDrive/TLCN/PKG-CT-ORG/CT-ORG/OrganSegmentations/`
3. Open notebooks in Google Colab
4. Select GPU runtime (L4 recommended, 24GB VRAM)
5. Run notebooks in order: `01_data_preparation.ipynb` → `06_visualization.ipynb`

## Usage

### Quick Start (Colab)

1. **Data Preparation** (`01_data_preparation.ipynb`):
   - Mount Google Drive
   - Load and preprocess CT-ORG volumes
   - Create sparse slice simulations (R=2, R=3)

2. **Classical Baselines** (`02_baseline_classical.ipynb`):
   - Evaluate nearest/linear/cubic interpolation
   - Generate baseline metrics

3. **U-Net Baseline** (`03_baseline_unet.ipynb`):
   - Train U-Net 2D model
   - Evaluate on test set

4. **3DGS Training** (`04_3dgs_training.ipynb`):
   - Train per-volume 3DGS models
   - Core experiment

5. **Benchmark & Ablation** (`05_benchmark_ablation.ipynb`):
   - Aggregate all results
   - Ablation studies (with/without regularization)

6. **Visualization** (`06_visualization.ipynb`):
   - Generate publication-ready figures
   - Qualitative comparisons

### Local Usage

```python
from src.data.ct_org_loader import CTORGLoader
from src.data.sparse_simulator import SparseSimulator
from src.training.trainer_3dgs import Trainer3DGS
from src.utils.config import load_config

# Load config
config = load_config("configs/default.yaml")

# Load data
loader = CTORGLoader(dataset_root="path/to/CT-ORG")
volume, labels, metadata = loader.load_and_preprocess(case_idx=0)

# Simulate sparse slices
simulator = SparseSimulator(sparse_ratio=2)
sparse_data = simulator.simulate(volume)

# Train 3DGS
trainer = Trainer3DGS(
    volume=volume,
    observed_indices=sparse_data["observed_indices"],
    target_indices=sparse_data["target_indices"],
    config=config,
    device="cuda"
)
trainer.train()

# Evaluate
results = trainer.evaluate_on_targets()
```

## Methods

- **Classical baselines**: Nearest, linear, cubic interpolation along z-axis
- **U-Net 2D baseline**: Two adjacent slices as input, middle slice as output
- **3DGS (Ours)**: Per-volume optimization with axis-aligned 3D Gaussians
  - Custom differentiable slice renderer
  - Medical-specific regularization (smoothness + edge preservation)
  - Adaptive densification and pruning

## Dataset

**CT-ORG** from TCIA: 140 CT volumes with multi-organ segmentations (liver, lungs, kidneys, bladder, bone).

- Download: [TCIA CT-ORG Collection](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890)
- Format: NIfTI (.nii.gz)
- Split: 98 train / 21 val / 21 test (as per dataset documentation)

## Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio) in dB
- **SSIM** (Structural Similarity Index)
- **ROI-based metrics**: Per-organ PSNR/SSIM using segmentation masks
- **Ablation studies**: Effect of regularization components

## Results

Results are saved to `outputs/` directory:
- `classical_baselines/` - Classical interpolation results
- `unet_baseline/` - U-Net training and evaluation
- `3dgs/` - 3DGS model checkpoints and results
- `figures/` - Publication-ready visualizations

## Technical Details

### 3DGS Architecture

- **Gaussian representation**: Position (x,y,z), log-scale (sx,sy,sz), opacity, intensity
- **Axis-aligned**: No rotation (simpler, fewer parameters, suitable for CT)
- **Slice renderer**: Differentiable, tile-based, optimized for axis-aligned rendering
- **Training**: Per-volume optimization with adaptive densification/pruning

### Optimization

- Mixed precision training (FP16) for L4 GPU efficiency
- Tile-based rendering for large volumes
- Gradient accumulation for densification decisions
- Checkpointing every 500 iterations

## Citation

If you use this code in your research, please cite:

```bibtex
@software{ct_slice_interpolation_3dgs,
  title = {CT Slice Interpolation via 3D Gaussian Splatting},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/YOUR_REPO_NAME}
}
```

## License

[Specify your license here, e.g., MIT License]

## Acknowledgments

- CT-ORG dataset from TCIA
- Inspired by MedGS and original 3D Gaussian Splatting paper
