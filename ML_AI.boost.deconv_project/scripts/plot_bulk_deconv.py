#!/usr/bin/env python3
"""
Plot deconvolution results:
- Stacked barplot: one bar per bulk sample, stacks = clusters (fractions)
- One barplot per cluster: bulk samples on x-axis showing predicted fraction for that cluster
Writes PNGs to `results/figs_deconv/` and a combined PDF `results/figs_deconv/bulk_deconv_grid.pdf`.
"""
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path

P = 'results/predictions_bulk_deconv.csv'
CONFIG = Path('config.yaml')
OUT = 'results/figs_deconv'
os.makedirs(OUT, exist_ok=True)

# read
df = pd.read_csv(P, index_col=0)
# Try to append single-cell reference proportions (if config and h5ad available)# Try to append single-cell reference proportions (per-genotype) (if config and h5ad available)
sc_ref_df = None
try:
    cfg = yaml.safe_load(open(CONFIG))
    sc_path = cfg.get('sc_paths', {}).get('merged')
    if sc_path:
        import anndata as ad
        adata = ad.read_h5ad(sc_path)
        # detect cluster column variants
        cluster_cols = ['seurat_clusters', 'seurat_cluster', 'Seurat_Cluster', 'cluster', 'louvain']
        col = None
        for c in cluster_cols:
            if c in adata.obs and c in adata.obs.columns:
                col = c
                break
        # detect genotype column variants
        genotype_cols = ['genotype','condition','group','sample_group']
        gcol = None
        for g in genotype_cols:
            if g in adata.obs and g in adata.obs.columns:
                gcol = g
                break
        if col is not None:
            # If genotype column present, compute per-genotype fractions, else compute global
            mapped_rows = []
            if gcol is not None:
                for g in sorted(adata.obs[gcol].dropna().unique()):
                    subset = adata.obs[adata.obs[gcol]==g]
                    if subset.shape[0] == 0:
                        continue
                    counts = subset[col].astype(str).value_counts(normalize=True)
                    mapped = {}
                    for c in df.columns:
                        if str(c).startswith('frac_'):
                            cluster_label = str(c).split('frac_')[-1]
                        else:
                            cluster_label = str(c)
                        mapped[c] = float(counts.get(cluster_label, 0.0))
                    mapped_rows.append((f'SC_{g}', mapped))
            else:
                # global reference
                counts = adata.obs[col].astype(str).value_counts(normalize=True)
                mapped = {}
                for c in df.columns:
                    if str(c).startswith('frac_'):
                        cluster_label = str(c).split('frac_')[-1]
                    else:
                        cluster_label = str(c)
                    mapped[c] = float(counts.get(cluster_label, 0.0))
                mapped_rows.append(('SC_reference', mapped))
            if len(mapped_rows) > 0:
                import pandas as _pd
                sc_ref_df = _pd.DataFrame([m for _,m in mapped_rows], index=[name for name,_ in mapped_rows])
except Exception:
    sc_ref_df = None

if sc_ref_df is not None:
    # ensure columns order matches df
    sc_ref_df = sc_ref_df[df.columns]
    df = pd.concat([df, sc_ref_df])
# ensure samples are strings and preserve current order
samples = list(df.index.astype(str))
clusters = list(df.columns)

# Stacked barplot (samples on x, clusters stacked)
fig, ax = plt.subplots(figsize=(max(8, 0.35*len(samples)), 6))
bottom = [0.0]*len(samples)
colors = sns.color_palette('tab20', n_colors=len(clusters))
for i, cl in enumerate(clusters):
    vals = df[cl].values
    ax.bar(samples, vals, bottom=bottom, label=cl, color=colors[i % len(colors)])
    bottom = [b + v for b, v in zip(bottom, vals)]
ax.set_xticks(range(len(samples)))
ax.set_xticklabels(samples, rotation=90, fontsize=8)
ax.set_ylabel('Predicted fraction')
ax.set_xlabel('Bulk sample')
ax.set_title('Stacked predicted fractions per bulk sample')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, fontsize=8)
plt.tight_layout()
out_stack = os.path.join(OUT, 'bulk_deconv_stacked.png')
fig.savefig(out_stack, dpi=200)
plt.close(fig)

# One barplot per cluster
n = len(clusters)
cols = 4
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), squeeze=False)
axes = axes.flatten()
for ax_i, cl in enumerate(clusters):
    ax = axes[ax_i]
    ax.bar(samples, df[cl].values, color=colors[ax_i % len(colors)])
    ax.set_title(cl)
    ax.set_ylim(0, df.max().max()*1.05)
    ax.tick_params(axis='x', labelrotation=90, labelsize=6)

# turn off unused axes
for j in range(n, len(axes)):
    axes[j].axis('off')
plt.tight_layout()
out_grid = os.path.join(OUT, 'bulk_deconv_cluster_grid.pdf')
fig.savefig(out_grid, dpi=200)
plt.close(fig)

# Also save individual PNGs per cluster
for cl in clusters:
    safe = str(cl).replace('/', '_').replace(' ', '_')
    fig, ax = plt.subplots(figsize=(max(6, 0.35*len(samples)), 3))
    ax.bar(samples, df[cl].values, color=colors[clusters.index(cl) % len(colors)])
    ax.set_xticklabels(samples, rotation=90, fontsize=8)
    ax.set_ylim(0, df.max().max()*1.05)
    ax.set_title(f'Predicted fraction for {cl} across bulk samples')
    ax.set_xlabel('Bulk sample')
    ax.set_ylabel('Predicted fraction')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f'cluster_{safe}.png'), dpi=200)
    plt.close(fig)

print('Wrote:', out_stack, out_grid, 'and individual cluster PNGs to', OUT)
