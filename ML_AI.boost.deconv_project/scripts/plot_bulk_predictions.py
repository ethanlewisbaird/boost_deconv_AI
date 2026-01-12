#!/usr/bin/env python3
"""
Plot bulk prediction probabilities: one barplot per cluster (class).
Reads: `results/predictions_bulk_proba.csv` (index = sample names, columns = cluster labels)
Writes: `results/figs/cluster_<col>.png` and `results/figs/bulk_cluster_prob_grid.pdf` (combined grid)
"""
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

PRED_PATH = 'results/predictions_bulk_proba.csv'
OUT_DIR = 'results/figs'
os.makedirs(OUT_DIR, exist_ok=True)

# Read predictions
try:
    df = pd.read_csv(PRED_PATH, index_col=0)
except Exception as e:
    raise SystemExit(f"Failed to read {PRED_PATH}: {e}")

# Ensure index are strings for plotting
df.index = df.index.map(str)

# Individual barplots per cluster
for col in df.columns:
    safe_col = str(col).replace('/', '_').replace(' ', '_')
    fig, ax = plt.subplots(figsize=(max(6, 0.3 * len(df.index)), 4))
    ax.bar(df.index, df[col], color='C0')
    ax.set_xlabel('Bulk sample')
    ax.set_ylabel('Predicted probability')
    ax.set_title(f'Cluster {col} predicted probability across bulk samples')
    plt.xticks(rotation=90, fontsize=8)
    plt.ylim(0, 1)
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f'cluster_{safe_col}.png')
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

# Combined grid of all clusters (arrange roughly square)
n = len(df.columns)
cols = 4
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), squeeze=False)
axes = axes.flatten()

for ax_i, col in enumerate(df.columns):
    ax = axes[ax_i]
    ax.bar(df.index, df[col], color='C0')
    ax.set_title(str(col), fontsize=10)
    ax.set_ylim(0,1)
    ax.tick_params(axis='x', labelrotation=90, labelsize=6)
    ax.set_xticklabels(df.index)

# Turn off any unused axes
for j in range(n, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
out_pdf = os.path.join(OUT_DIR, 'bulk_cluster_prob_grid.pdf')
fig.savefig(out_pdf, dpi=200)
plt.close(fig)

# Print summary
print(f"Read predictions: {PRED_PATH} -> shape {df.shape}")
print(f"Wrote individual PNGs to {OUT_DIR} and combined PDF: {out_pdf}")
