import numpy as np
import pandas as pd


def to_cpm(df_counts: pd.DataFrame):
    # columns = samples
    col_sums = df_counts.sum(axis=0)
    return df_counts.divide(col_sums, axis=1) * 1e6


def to_tpm(df_counts: pd.DataFrame, gene_lengths: pd.Series):
    # gene_lengths indexed by gene
    rpk = df_counts.divide(gene_lengths, axis=0)
    per_sample = rpk.sum(axis=0)
    return rpk.divide(per_sample, axis=1) * 1e6


def to_log1p(df_counts: pd.DataFrame):
    cpm = to_cpm(df_counts)
    return np.log1p(cpm)