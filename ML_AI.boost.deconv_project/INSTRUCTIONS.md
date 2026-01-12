# ML_AI.boost.deconv — Microscopy-Guided Pseudobulk Deconvolution Pipeline

A fully functional pipeline for deconvolving bulk RNA/ATAC-seq samples into cell-type proportions using single-cell references with optional microscopy-guided priors.

## Quick Start (1–5 minutes)

### 1. Set Up Environment

```bash
# Create and activate conda environment
conda create -n deconv_env python=3.10 -y
conda activate deconv_env
cd /path/to/ML_AI.boost.deconv_project
pip install -r requirements.txt
```

### 2. Configure Data Paths

Edit `config.yaml` with your data paths:

```yaml
sc_paths:
  merged: "/path/to/your/anndata.h5ad"  # Single-cell reference (AnnData format)

bulk_paths:
  merged: "/path/to/your/bulk_counts.tsv"  # Bulk counts (genes × samples)

microscopy_csv: "data/metadata/microscopy.csv"  # Optional: observed cell fractions
```

**AnnData requirements:**
- Must contain a clustering label column (default: `Seurat_Cluster`)
- Must contain a `genotype` column for grouping samples (used in plots)

**Microscopy CSV format (optional):**
```
Sample,Seurat_Cluster,Observed_Fraction
sample1,0,0.15
sample1,1,0.35
sample1,2,0.50
```

### 3. Run the Pipeline

#### Full Pipeline (Fast mode, 100 pseudobulk samples):
```bash
PYTHONPATH=. python train_fast.py
PYTHONPATH=. python predict_fast.py
PYTHONPATH=. python scripts/plot_bulk_deconv.py
```

#### Full Pipeline (Production mode, 500 pseudobulk samples):
```bash
PYTHONPATH=. python pipeline/models/train_deconv.py
PYTHONPATH=. python predict_fast.py
PYTHONPATH=. python scripts/plot_bulk_deconv.py
```

### 4. View Results

All outputs saved to `results/`:

- **Model**: `results/models/model_deconv.joblib` (trained Multi-Output XGBoost)
- **Predictions**: `results/predictions_bulk_deconv.csv` (predicted fractions per bulk sample)
- **Plots**:
  - `results/figs_deconv/bulk_deconv_stacked.png` (stacked barplot)
  - `results/figs_deconv/bulk_deconv_cluster_grid.pdf` (per-cluster grid)
  - `results/figs_deconv/cluster_frac_*.png` (individual cluster plots)

---

## Detailed Pipeline Steps

### Step 1: Pseudobulk Generation

Generate synthetic mixtures with known fractions from your single-cell reference:

```bash
# Train script automatically calls this:
from pipeline.pseudobulk.advanced_pseudobulk import generate_fractional_pseudobulk
pb = generate_fractional_pseudobulk(adata, config)
```

**What it does:**
- Samples cells from each cluster
- Draws fractions from a Dirichlet distribution
- Aggregates counts and simulates sequencing read-depth and dropout
- Optionally biases fractions using microscopy-derived priors

**Configuration:**
```yaml
pseudo_bulk_samples: 500  # Number of synthetic samples to generate
augmentation_factor: 2.0  # Dirichlet concentration (higher = more uniform)

pseudobulk:
  read_depth_mean: 1000000  # Target sequencing depth
  dropout_rate: 0.1  # Fraction of genes to randomly zero

# Microscopy guidance (optional)
use_microscopy_augmentation: true
microscopy_mode: "global"  # or "per_sample"
microscopy_concentration_k: 10.0  # Prior strength (higher = closer to microscopy fractions)
```

### Step 2: Deconvolution Model Training

Train a multi-output regressor to predict cluster fractions:

```bash
PYTHONPATH=. python pipeline/models/train_deconv.py
```

**Model**: `sklearn.multioutput.MultiOutputRegressor(xgboost.XGBRegressor)`

**Training:**
- Input: pseudobulk gene expression (samples × genes)
- Output: cluster fractions (samples × K clusters)
- Loss: Mean Squared Error per cluster
- Split: 80% train, 20% test (on simulated data)

### Step 3: Bulk Sample Prediction

Apply the trained model to predict fractions for real bulk samples:

```bash
PYTHONPATH=. python predict_fast.py
```

**What it does:**
1. Loads trained model and feature list
2. Reads bulk sample counts
3. Aligns genes to model features (pads/trims as needed)
4. Scales bulk to pseudobulk read-depth (preserving proportions)
5. Predicts fractions per bulk sample
6. Normalizes predictions (clipped to [0,1], summed to 1)
7. Writes `results/predictions_bulk_deconv.csv`

### Step 4: Visualization

Generate barplots showing predicted fractions:

```bash
PYTHONPATH=. python scripts/plot_bulk_deconv.py
```

**Outputs:**
- **Stacked barplot**: All clusters stacked per sample
- **Per-cluster grid**: Individual rows per cluster, samples on x-axis
- **Per-cluster detail**: Separate PNG per cluster with single-cell reference bars

---

## Advanced: Evaluation & Comparison

Compare augmentation modes (unguided vs. global microscopy vs. per-sample):

```bash
PYTHONPATH=. python scripts/evaluate_deconv.py
```

**Outputs:**
- `results/eval_none/metrics.json` (unguided Dirichlet)
- `results/eval_global/metrics.json` (global microscopy prior)
- `results/eval_per_sample/metrics.json` (per-sample microscopy prior)
- `results/eval_summary.json` (aggregate summary)

Each mode trains a separate model and predicts on bulk; metrics include overall RMSE and per-cluster RMSE on held-out pseudobulk test set.

---

## Configuration Reference

### Data Paths
```yaml
sc_paths:
  merged: "path/to/merged_anndata.h5ad"  # Required

bulk_paths:
  merged: "path/to/bulk_counts.tsv"  # Required

microscopy_csv: "path/to/microscopy.csv"  # Optional
```

### Pseudobulk Generation
```yaml
pseudo_bulk_samples: 500  # Number of synthetic samples
augmentation_factor: 2.0  # Dirichlet concentration parameter

pseudobulk:
  read_depth_mean: 1000000  # Sequencing depth for read simulation
  dropout_rate: 0.1  # Fraction of low-count genes to randomly zero
  simulate_read_depth: true  # Enable multinomial read-depth simulation
```

### Microscopy Augmentation
```yaml
use_microscopy_augmentation: false  # Enable microscopy-guided priors
microscopy_mode: "global"  # "global" (mean) or "per_sample" (random sample)
microscopy_concentration_k: 10.0  # Strength of microscopy prior (higher = stronger)
```

### Model Training
```yaml
model_type: "xgboost"
hyperparameter_search: false  # GridSearch (slow) or fixed params

output_dir: "results/"  # Output directory for all artifacts
```

---

## File Structure

```
ML_AI.boost.deconv_project/
├── config.yaml                    # Main configuration
├── requirements.txt               # Python dependencies
├── INSTRUCTIONS.md               # This file
│
├── train_fast.py                 # Fast training (100 pseudobulk samples)
├── predict_fast.py               # Bulk prediction script
├── pipeline/
│   ├── io/load_sc.py             # Load single-cell AnnData
│   ├── preprocessing/filter_genes.py  # Filter MT/rpl genes
│   ├── pseudobulk/advanced_pseudobulk.py  # Pseudobulk generation
│   ├── models/train_deconv.py    # Training script
│   └── validators.py             # Input validation
│
├── scripts/
│   ├── predict_bulk_deconv.py    # Prediction (older version)
│   ├── plot_bulk_deconv.py       # Plotting routines
│   └── evaluate_deconv.py        # Evaluation across modes
│
└── results/
    ├── models/
    │   └── model_deconv.joblib  # Trained model
    ├── predictions_bulk_deconv.csv  # Predictions
    ├── eval_summary.json         # Evaluation metrics
    └── figs_deconv/
        ├── bulk_deconv_stacked.png
        ├── bulk_deconv_cluster_grid.pdf
        └── cluster_frac_*.png
```

---

## Troubleshooting

### ImportError: No module named 'anndata'
```bash
pip install anndata scanpy
```

### Gene mismatch between bulk and model
The pipeline automatically handles this by:
- Padding missing genes with zeros
- Trimming extra genes
- This preserves compositional data (proportions)

### Predictions are uniform across clusters
Possible causes:
- Model underfitted (increase `pseudo_bulk_samples`)
- Bulk data quality issues (check normalization)
- Feature scaling problem (check read-depth simulation)

Try:
- Set `pseudo_bulk_samples: 500` (production mode)
- Regenerate with `use_microscopy_augmentation: true` and valid microscopy CSV
- Check bulk sample quality and distribution

### Slow pseudobulk generation
The `train_fast.py` script uses 100 samples for quick iteration. For production:
```yaml
pseudo_bulk_samples: 500  # Longer training, better model
```

If still slow, try subsetting single-cell data to frequently expressed genes.

---

## Citation

If you use this pipeline, please cite:

> [Citation info for the microscopy-guided deconvolution method]

---

## Contact

For questions or issues, refer to the code comments and config examples above.

