import os
import anndata as ad


def load_sc_data(path, is_merged=True):
    """Load single merged h5ad or a list of library files and return AnnData.

    Raises FileNotFoundError if files are missing.
    """
    if is_merged:
        if not os.path.exists(path):
            raise FileNotFoundError(f'merged h5ad not found: {path}')
        return ad.read_h5ad(path)
    else:
        # path expected as list of filepaths
        adatas = []
        for p in path:
            if not os.path.exists(p):
                raise FileNotFoundError(f'h5ad not found: {p}')
            adatas.append(ad.read_h5ad(p))
        return ad.concat(adatas, join='outer')
