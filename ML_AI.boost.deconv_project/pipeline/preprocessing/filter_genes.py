import re
from typing import Iterable


def _is_mito_gene(g: str) -> bool:
    gup = str(g).upper()
    # common patterns: MT-*, MT*, COX*, ND* (mitochondrial genes vary by organism)
    return gup.startswith('^mt:') or gup.startswith('MT') or gup.startswith('COX') or re.match(r'^ND\d', gup) is not None


def _is_ribosomal_gene(g: str) -> bool:
    gup = str(g).upper()
    return gup.startswith('RpL') or gup.startswith('RpS') or 'RIBOSOMAL' in gup


def filter_mt_rpl(adata):
    """Return a copy of `adata` with mitochondrial and ribosomal genes removed.

    Uses simple heuristics to detect MT and RPL/RPS genes. For other organisms
    adjust heuristics as needed.
    """
    # operate on var_names (AnnData.var_names)
    try:
        var_names = list(adata.var_names)
    except Exception:
        # fallback for older AnnData structures
        var_names = list(adata.var.index)

    remove_idx = [i for i, g in enumerate(var_names) if _is_mito_gene(g) or _is_ribosomal_gene(g)]

    if not remove_idx:
        return adata

    mask = [i not in remove_idx for i in range(len(var_names))]
    # slice and return a copy
    return adata[:, mask].copy()
