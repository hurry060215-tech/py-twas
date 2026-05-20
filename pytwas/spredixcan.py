"""S-PrediXcan: summary-statistics transcriptome-wide association study.

High-level entry point reproducing the original MetaXcan ``SPrediXcan.py``
(``M03_betas`` GWAS harmonisation + ``M04_zscores`` association).  Given:

* a prediction-model ``.db`` (elastic-net SNP -> expression weights),
* a reference SNP-covariance file, and
* a GWAS summary-statistics file,

it produces a per-gene association table identical in layout to
S-PrediXcan output:

``gene, gene_name, zscore, effect_size, pvalue, var_g, pred_perf_r2,
pred_perf_pval, pred_perf_qval, n_snps_used, n_snps_in_cov,
n_snps_in_model`` (plus ``best_gwas_p, largest_weight`` with
``additional_output=True``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

from .association import associate
from .gwas import load_gwas
from .model import load_covariance, load_model

__all__ = ["spredixcan"]

#: column order of the S-PrediXcan result table
_RESULT_COLUMNS = [
    "gene",
    "gene_name",
    "zscore",
    "effect_size",
    "pvalue",
    "var_g",
    "pred_perf_r2",
    "pred_perf_pval",
    "pred_perf_qval",
    "n_snps_used",
    "n_snps_in_cov",
    "n_snps_in_model",
]


def _to_int(x):
    """Coerce a value to ``int`` where possible (mirrors MetaXcan's ``_to_int``)."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def spredixcan(
    model_db_path: str,
    covariance: str,
    gwas_file: Optional[str] = None,
    *,
    gwas: Optional[pd.DataFrame] = None,
    snp_column: str = "SNP",
    effect_allele_column: str = "A1",
    non_effect_allele_column: str = "A2",
    zscore_column: Optional[str] = None,
    beta_column: Optional[str] = None,
    se_column: Optional[str] = None,
    or_column: Optional[str] = None,
    beta_sign_column: Optional[str] = None,
    pvalue_column: Optional[str] = None,
    chromosome_column: Optional[str] = None,
    position_column: Optional[str] = None,
    separator: Optional[str] = None,
    keep_non_rsid: bool = False,
    input_pvalue_fix: Optional[float] = 1e-50,
    model_db_snp_key: Optional[str] = None,
    additional_output: bool = False,
    remove_ens_version: bool = False,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """Run an S-PrediXcan summary-statistics TWAS.

    Parameters
    ----------
    model_db_path:
        Path to the PrediXcan/MetaXcan prediction-model ``.db`` file.
    covariance:
        Path to the reference SNP-covariance ``.txt`` / ``.txt.gz`` file.
    gwas_file:
        Path to the GWAS summary-statistics file.  Either this or
        ``gwas`` must be given.
    gwas:
        Pre-loaded GWAS data frame (alternative to ``gwas_file``);
        useful when you already harmonised the GWAS yourself.
    snp_column, effect_allele_column, non_effect_allele_column:
        GWAS column names for SNP id and the two alleles.
    zscore_column, beta_column, se_column, or_column, beta_sign_column,
    pvalue_column:
        GWAS association columns.  Provide a ``zscore`` column, OR a
        ``pvalue`` column with one of ``beta`` / ``or`` / ``beta_sign``,
        OR an ``se`` column with ``beta`` / ``or``.
    chromosome_column, position_column:
        Optional positional columns.
    separator:
        GWAS field separator; defaults to any whitespace.
    keep_non_rsid:
        Keep SNPs whose id does not contain ``rs`` (default ``False``).
    input_pvalue_fix:
        Threshold used to repair divergent z-scores from tiny p-values.
    model_db_snp_key:
        Alternative ``weights``-table column to use as the SNP id.
    additional_output:
        Append ``best_gwas_p`` and ``largest_weight`` columns.
    remove_ens_version:
        Drop the ``.NN`` Ensembl version suffix from gene ids.
    output_file:
        If given, write the result table to this CSV path.

    Returns
    -------
    pandas.DataFrame
        Per-gene association table sorted by ascending p-value.
    """
    if gwas is None and gwas_file is None:
        raise ValueError("Provide either gwas_file or gwas")

    model = load_model(model_db_path, snp_key=model_db_snp_key)
    cov = load_covariance(covariance)

    if gwas is None:
        gwas = load_gwas(
            gwas_file,
            snp_column=snp_column,
            effect_allele_column=effect_allele_column,
            non_effect_allele_column=non_effect_allele_column,
            zscore_column=zscore_column,
            beta_column=beta_column,
            se_column=se_column,
            or_column=or_column,
            beta_sign_column=beta_sign_column,
            pvalue_column=pvalue_column,
            chromosome_column=chromosome_column,
            position_column=position_column,
            separator=separator,
            keep_non_rsid=keep_non_rsid,
            input_pvalue_fix=input_pvalue_fix,
        )

    results = associate(model, gwas, cov, additional=additional_output)

    rows = []
    for r in results:
        rows.append(
            {
                "gene": r.gene,
                "zscore": r.zscore,
                "effect_size": r.effect_size,
                "var_g": r.var_g,
                "n_snps_in_model": r.n_snps_in_model,
                "n_snps_in_cov": r.n_snps_in_cov,
                "n_snps_used": r.n_snps_used,
                "best_gwas_p": r.best_gwas_p,
                "largest_weight": r.largest_weight,
            }
        )
    df = pd.DataFrame(rows)

    df = _format_output(df, model.extra, remove_ens_version, additional_output)

    if output_file:
        df.to_csv(output_file, index=False, na_rep="NA")
    return df


def _format_output(
    df: pd.DataFrame,
    extra: pd.DataFrame,
    remove_ens_version: bool,
    additional_output: bool,
) -> pd.DataFrame:
    """Compute p-values, merge gene metadata and order columns.

    Faithful port of ``metax.metaxcan.Utilities.format_output``.
    """
    if df.shape[0] == 0:
        cols = list(_RESULT_COLUMNS)
        if additional_output:
            cols += ["best_gwas_p", "largest_weight"]
        return pd.DataFrame(columns=cols)

    # two-sided normal p-value, avoiding cdf on non-finite zscores
    pvalue = np.full(df.shape[0], np.nan)
    finite = np.isfinite(df["zscore"].values)
    pvalue[finite] = 2.0 * stats.norm.sf(np.abs(df.loc[finite, "zscore"].values))
    df = df.assign(pvalue=pvalue)

    # the association frame and the `extra` table both carry an
    # `n_snps_in_model` column; the model's `extra` value is authoritative
    # (it is what S-PrediXcan reports), so drop the association copy.
    df = df.drop(columns=["n_snps_in_model"], errors="ignore")
    merged = pd.merge(df, extra, how="inner", on="gene")
    if remove_ens_version:
        merged["gene"] = merged["gene"].str.split(".").str.get(0)

    column_order = list(_RESULT_COLUMNS)
    if additional_output:
        column_order += ["best_gwas_p", "largest_weight"]
    merged = merged[column_order]

    merged["n_snps_in_cov"] = merged["n_snps_in_cov"].apply(_to_int)
    merged = merged.sort_values(by="pvalue").reset_index(drop=True)
    return merged
