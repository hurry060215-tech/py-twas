"""S-PrediXcan summary-statistics TWAS association.

This is the workhorse of py-twas: the closed-form S-PrediXcan formula of
Barbeira *et al.* (*Nature Communications* 2018), reproducing
``metax.metaxcan.AssociationCalculation`` from the original MetaXcan.

For each gene the elastic-net prediction-model weights ``w``, the GWAS
per-SNP z-scores ``z`` and betas ``b``, and the reference SNP covariance
``Sigma`` are combined into a gene-level association:

.. math::

    \\sigma_g^2 = w^{T}\\,\\Sigma\\,w

    Z_g = \\frac{\\sum_l w_l\\,z_l\\,\\sigma_l}{\\sqrt{\\sigma_g^2}}

    \\text{effect}_g =
        \\frac{\\sum_l w_l\\,b_l\\,\\sigma_l^2}{\\sigma_g^2}

where :math:`\\sigma_l = \\sqrt{\\Sigma_{ll}}` is the reference standard
deviation of SNP *l*.  The gene p-value is the two-sided normal tail
``2 * Phi(-|Z_g|)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

from .gwas import BETA, SNP, ZSCORE, align_to_model
from .model import CovarianceDB, PredictionModel

__all__ = [
    "GeneAssociation",
    "gene_association",
    "associate"]


@dataclass
class GeneAssociation:
    """Per-gene S-PrediXcan association result.

    Attributes
    ----------
    gene:
        Gene identifier.
    zscore:
        Gene-level S-PrediXcan z-score (``NaN`` if not computable).
    effect_size:
        Gene-level effect size (``NaN`` if no betas / not computable).
    var_g:
        Predicted-expression variance :math:`\\sigma_g^2 = w^T \\Sigma w`.
    n_snps_in_model:
        SNPs in the gene's prediction model.
    n_snps_in_cov:
        SNPs with covariance data for the gene.
    n_snps_used:
        SNPs actually used (in model AND GWAS AND covariance).
    """

    gene: str
    zscore: float
    effect_size: float
    var_g: float
    n_snps_in_model: int
    n_snps_in_cov: float
    n_snps_used: int
    snps_used: Tuple[str, ...] = ()
    best_gwas_p: float = np.nan
    largest_weight: float = np.nan


def _gene_data(
    gene: str,
    model: PredictionModel,
    gwas_index: dict,
    covariance: CovarianceDB,
) -> Tuple[int, List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Assemble aligned (weight, zscore, beta, covariance) arrays for a gene.

    Returns ``(n_snps_in_model, snps_used, weights, zscores, betas, cov,
    n_snps_in_cov)``.  ``snps_used`` is exactly the order in which SNPs
    appear in the covariance matrix, with model weight and GWAS data
    aligned to it.
    """
    entries = model.weight_entries(gene)  # (rsid, weight, eff, non_eff)
    n_in_model = len(entries)
    n_in_cov = covariance.n_snps(gene)

    # weight map; only SNPs present in the GWAS survive
    w_map = {rsid: w for rsid, w, _, _ in entries}
    snps_with_gwas = [s for s in w_map if s in gwas_index]

    snps, cov = covariance.get(gene, snps_with_gwas)
    if snps is None or len(snps) == 0:
        return n_in_model, [], np.array([]), np.array([]), np.array([]), \
            np.zeros((0, 0)), n_in_cov

    weights = np.array([w_map[s] for s in snps], dtype=np.float64)
    zscores = np.array([gwas_index[s][0] for s in snps], dtype=np.float64)
    betas = np.array([gwas_index[s][1] for s in snps], dtype=np.float64)
    return n_in_model, list(snps), weights, zscores, betas, cov, n_in_cov


def gene_association(
    gene: str,
    model: PredictionModel,
    gwas_index: dict,
    covariance: CovarianceDB,
    additional: bool = False,
) -> GeneAssociation:
    """Compute the S-PrediXcan association for a single gene.

    Parameters
    ----------
    gene:
        Gene identifier (must be present in ``model``).
    model:
        Prediction model.
    gwas_index:
        ``rsid -> (zscore, beta)`` mapping of the (model-aligned) GWAS.
    covariance:
        Reference SNP covariance store.
    additional:
        If ``True`` also compute ``best_gwas_p`` and ``largest_weight``.

    Returns
    -------
    GeneAssociation
    """
    n_in_model, snps, w, z, b, cov, n_in_cov = _gene_data(
        gene, model, gwas_index, covariance
    )
    n_used = len(snps)
    zscore = effect_size = var_g = np.nan

    if n_used > 0:
        variances = np.diag(cov)
        sigma_l = np.sqrt(variances)
        # sigma_g^2 = w^T Sigma w
        var_g = float(w.dot(cov).dot(w))
        if var_g > 0:
            denom = np.sqrt(var_g)
            zscore = float(np.sum(w * z * sigma_l) / denom)
            if not np.all(np.isnan(b)):
                effect_size = float(np.sum(w * b * (sigma_l ** 2)) / var_g)

    result = GeneAssociation(
        gene=gene,
        zscore=zscore,
        effect_size=effect_size,
        var_g=var_g,
        n_snps_in_model=n_in_model,
        n_snps_in_cov=n_in_cov,
        n_snps_used=n_used,
        snps_used=tuple(snps),
    )

    if additional and n_used > 0:
        best_z = float(np.max(np.abs(z)))
        result.best_gwas_p = 2.0 * stats.norm.cdf(-best_z)
        result.largest_weight = float(np.max(np.abs(w)))

    return result


def _build_gwas_index(gwas: pd.DataFrame) -> dict:
    """Build the ``rsid -> (zscore, beta)`` lookup from a GWAS frame.

    Non-finite z-scores are discarded (as in MetaXcan's ``_sanitized_gwas``).
    """
    has_beta = BETA in gwas.columns
    index: dict = {}
    for row in gwas.itertuples(index=False):
        z = float(getattr(row, ZSCORE))
        if not np.isfinite(z):
            continue
        b = float(getattr(row, BETA)) if has_beta else np.nan
        index[getattr(row, SNP)] = (z, b)
    return index


def associate(
    model: PredictionModel,
    gwas: pd.DataFrame,
    covariance: CovarianceDB,
    genes: Optional[List[str]] = None,
    additional: bool = False,
    align: bool = True,
) -> List[GeneAssociation]:
    """Run S-PrediXcan association across all genes of a model.

    Parameters
    ----------
    model:
        Prediction model.
    gwas:
        GWAS summary statistics (output of :func:`pytwas.load_gwas`).
    covariance:
        Reference SNP covariance.
    genes:
        Optional subset of genes; defaults to all genes in the model
        ``extra`` table.
    additional:
        Whether to compute ``best_gwas_p`` / ``largest_weight``.
    align:
        Whether to harmonise GWAS alleles to the model first (default).

    Returns
    -------
    list of GeneAssociation
    """
    if align:
        gwas = align_to_model(gwas, model.weights)
    gwas_index = _build_gwas_index(gwas)

    if genes is None:
        genes = model.genes()

    results: List[GeneAssociation] = []
    for gene in genes:
        entries = model.weight_entries(gene)
        if not entries:
            continue
        # mirror MetaXcan's data-intersection: only genes with at least one
        # model SNP present in the GWAS are reported at all.
        if not any(rsid in gwas_index for rsid, *_ in entries):
            continue
        results.append(
            gene_association(gene, model, gwas_index, covariance, additional)
        )
    return results
