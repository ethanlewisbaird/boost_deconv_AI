import numpy as np
import pandas as pd
from anndata import AnnData
import scipy.sparse as sp


def simulate_read_depth(counts_array, mean_depth=1e6):
    """Scale counts to a simulated library size then draw from multinomial.

    This function sanitizes the input counts to handle zeros/NaNs and
    floating point rounding that can make probabilities sum slightly
    > 1 after casting. It returns an integer array of drawn counts.
    """
    # Convert to numpy array of floats
    arr = np.asarray(counts_array, dtype=np.float64).ravel()
    # Replace NaN/inf with 0
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    total = arr.sum()
    if total <= 0 or not np.isfinite(total):
        # nothing to draw from - return zeros with integer dtype
        return np.zeros_like(arr, dtype=int)
    # compute proportions
    prop = arr / total
    # guard against tiny negative/positive rounding errors
    prop = np.clip(prop, 0.0, 1.0)
    # ensure sum(prop) == 1.0 exactly (within float tolerances) by renormalizing
    s = prop.sum()
    if s == 0.0:
        # fallback: uniform probabilities
        prop = np.ones_like(prop, dtype=np.float64) / prop.size
    else:
        prop = prop / s
    # To avoid issues with rounding after casting, force last element = 1 - sum(rest)
    if prop.size > 1:
        rest = prop[:-1].astype(np.float64)
        last = 1.0 - rest.sum()
        # clip last to [0,1]
        last = float(np.clip(last, 0.0, 1.0))
        prop = np.concatenate([rest, np.array([last], dtype=np.float64)])
    else:
        prop = np.array([1.0], dtype=np.float64)
    # final safety: normalize again (should sum to 1)
    prop = prop / float(prop.sum())
    # draw from multinomial with integer lib size
    lib_size = int(max(0, np.random.poisson(mean_depth)))
    if lib_size == 0:
        return np.zeros_like(arr, dtype=int)
    drawn = np.random.multinomial(lib_size, prop)
    return drawn


def apply_dropout(counts, dropout_rate=0.01):
    mask = np.random.rand(*counts.shape) < dropout_rate
    counts[mask] = 0
    return counts


def generate_advanced_pseudobulk(adata: AnnData, config: dict):
    n_samples = config.get('pseudo_bulk_samples', 500)
    cells_per = config.get('cells_per_pseudobulk', 200)
    simulate_rd = config.get('pseudobulk', {}).get('simulate_read_depth', True)
    mean_depth = config.get('pseudobulk', {}).get('read_depth_mean', 1e6)
    dropout = config.get('pseudobulk', {}).get('dropout_rate', 0.01)

    # Accept common cluster column variants
    cluster_cols = ['seurat_clusters', 'seurat_cluster', 'Seurat_Cluster', 'cluster', 'louvain']
    col = None
    for c in cluster_cols:
        if c in adata.obs and c in adata.obs.columns:
            col = c
            break
    if col is None:
        raise ValueError(f'cluster column required; tried {cluster_cols}')
    clusters = adata.obs[col].unique()
    cluster_map = {c: list(adata.obs_names[adata.obs[col]==c]) for c in clusters}
    genes = adata.var_names
    pb_list = []
    meta = []
    for i in range(n_samples):
        dominant = np.random.choice(clusters)
        n_dom = int(cells_per * 0.6)
        n_other = cells_per - n_dom
        cells = []
        cells += list(np.random.choice(cluster_map[dominant], size=min(n_dom, len(cluster_map[dominant])), replace=True))
        other_clusters = [c for c in clusters if c!=dominant]
        for _ in range(n_other):
            c = np.random.choice(other_clusters)
            cells.append(np.random.choice(cluster_map[c]))
        X = adata[cells].X
        if sp.issparse(X):
            agg = np.asarray(X.sum(axis=0)).ravel()
        else:
            agg = np.sum(X, axis=0)
        if simulate_rd:
            agg = simulate_read_depth(agg, mean_depth)
        if dropout > 0:
            agg = apply_dropout(agg, dropout)
        pb_list.append(agg)
        meta.append({'sample_id': f'pb_{i}', 'dominant_cluster': str(dominant), 'n_cells': len(cells)})

    pb_df = pd.DataFrame(np.vstack(pb_list).T, index=genes, columns=[m['sample_id'] for m in meta])
    meta_df = pd.DataFrame(meta).set_index('sample_id')
    return {'pseudobulk': pb_df, 'meta': meta_df}


def generate_fractional_pseudobulk(adata, config: dict):
    """Generate pseudobulk samples with explicit fractional composition (ground truth).

    Returns dict with 'pseudobulk' DataFrame (genes x samples) and 'meta' DataFrame
    with columns for each cluster fraction (one column per cluster, sum to 1).
    """
    import numpy as _np
    import pandas as _pd
    import scipy.sparse as _sp

    n_samples = int(config.get('pseudo_bulk_samples', 500))
    cells_per = int(config.get('cells_per_pseudobulk', 200))
    mean_depth = float(config.get('pseudobulk', {}).get('read_depth_mean', 1e6))
    dropout = float(config.get('pseudobulk', {}).get('dropout_rate', 0.01))
    rng = _np.random.default_rng()

    # find cluster column
    cluster_cols = ['seurat_clusters', 'seurat_cluster', 'Seurat_Cluster', 'cluster', 'louvain']
    col = None
    for c in cluster_cols:
        if c in adata.obs and c in adata.obs.columns:
            col = c
            break
    if col is None:
        raise ValueError(f'cluster column required; tried {cluster_cols}')
    clusters = list(adata.obs[col].unique())
    cluster_map = {c: list(adata.obs_names[adata.obs[col] == c]) for c in clusters}
    genes = adata.var_names

    pb_list = []
    meta_rows = []

    # Microscopy-guided augmentation: optional
    use_mic = bool(config.get('use_microscopy_augmentation', False))
    mic_mode = config.get('microscopy_mode', 'global')  # 'global' or 'per_sample'
    mic_k = float(config.get('microscopy_concentration_k', 50.0))
    alpha_base = float(config.get('augmentation_factor', 2.0))

    # if microscopy is requested, try to read microscopy csv and build priors
    mic_priors = None
    if use_mic:
        mic_path = config.get('microscopy_csv')
        try:
            if mic_path:
                mic_df = _pd.read_csv(mic_path)
                # Expected columns (validator enforces these earlier): Sample, Seurat_Cluster, Observed_Fraction
                # normalize cluster labels as strings
                mic_df['Seurat_Cluster'] = mic_df['Seurat_Cluster'].astype(str)
                # build per-sample mapping
                per_sample = {}
                for s, g in mic_df.groupby('Sample'):
                    vec = {r['Seurat_Cluster']: float(r['Observed_Fraction']) for _, r in g.iterrows()}
                    per_sample[str(s)] = vec
                # build global mean
                all_clusters = list({str(x) for x in mic_df['Seurat_Cluster'].unique()})
                global_vec = {}
                for c in all_clusters:
                    vals = mic_df[mic_df['Seurat_Cluster'].astype(str) == c]['Observed_Fraction'].astype(float)
                    if len(vals) == 0:
                        global_vec[c] = 0.0
                    else:
                        global_vec[c] = float(vals.mean())
                mic_priors = {'per_sample': per_sample, 'global': global_vec}
        except Exception:
            mic_priors = None

    for i in range(n_samples):
        # decide fractions
        if use_mic and mic_priors is not None:
            # build alpha according to chosen mode
            if mic_mode == 'per_sample' and len(mic_priors.get('per_sample', {})) > 0:
                # choose a microscopy sample at random and form alpha = observed * k
                mic_sample = rng.choice(list(mic_priors['per_sample'].keys()))
                obs = mic_priors['per_sample'][mic_sample]
                # align to cluster order
                obs_vec = _np.array([float(obs.get(str(cl), 0.0)) for cl in clusters], dtype=float)
                obs_vec = obs_vec / (obs_vec.sum() + 1e-12)
                alpha = (obs_vec + 1e-6) * mic_k
            else:
                # global prior
                obs = mic_priors.get('global', {})
                obs_vec = _np.array([float(obs.get(str(cl), 0.0)) for cl in clusters], dtype=float)
                obs_vec = obs_vec / (obs_vec.sum() + 1e-12)
                alpha = (obs_vec + 1e-6) * mic_k
            # fallback if alpha invalid
            if not _np.all(_np.isfinite(alpha)) or alpha.sum() <= 0:
                alpha = _np.ones(len(clusters)) * (1.0 / max(1e-3, alpha_base))
            fracs = rng.dirichlet(alpha)
            fracs = fracs / fracs.sum()
        else:
            # Dirichlet concentration: uses augmentation_factor as dispersion control if present
            alpha = _np.ones(len(clusters)) * (1.0 / max(1e-3, alpha_base))
            fracs = rng.dirichlet(alpha)
            fracs = fracs / fracs.sum()

        # choose cells according to fractions (multinomial counts of cells_per)
        counts_cells = rng.multinomial(cells_per, fracs)
        chosen_cells = []
        for ci, ccount in enumerate(counts_cells):
            cl = clusters[ci]
            pool = cluster_map.get(cl, [])
            if len(pool) == 0 or ccount == 0:
                continue
            # sample with replacement when pool smaller
            replace = ccount > len(pool)
            chosen = list(rng.choice(pool, size=ccount, replace=replace))
            chosen_cells.extend(chosen)

        if len(chosen_cells) == 0:
            # fallback sample: pick one random cell
            chosen_cells = [rng.choice(adata.obs_names, size=1)[0]]

        X = adata[chosen_cells].X
        if _sp.issparse(X):
            agg = _np.asarray(X.sum(axis=0)).ravel()
        else:
            agg = _np.sum(X, axis=0)

        # simulate read depth and dropout like earlier
        if config.get('pseudobulk', {}).get('simulate_read_depth', True):
            agg = simulate_read_depth(agg, mean_depth)
        if dropout > 0:
            agg = apply_dropout(agg, dropout)

        pb_list.append(agg)
        meta_rows.append({'sample_id': f'pb_frac_{i}', **{f'frac_{clusters[j]}': float(fracs[j]) for j in range(len(clusters))}})

    pb_df = _pd.DataFrame(_np.vstack(pb_list).T, index=genes, columns=[r['sample_id'] for r in meta_rows])
    meta_df = _pd.DataFrame(meta_rows).set_index('sample_id')
    return {'pseudobulk': pb_df, 'meta': meta_df}
