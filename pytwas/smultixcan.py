"""S-MultiXcan: multi-tissue / cross-model joint TWAS.

Reproduces the original MetaXcan ``SMulTiXcan.py`` together with
``metax.cross_model`` and ``metax.genotype.GeneExpressionMatrixManager``.

S-MultiXcan aggregates the per-tissue S-PrediXcan z-scores of a gene into
a single multi-tissue test.  For a gene measured in *k* tissues with
z-score vector ``z``, the tissue-tissue correlation matrix ``R`` is
estimated from the shared SNP covariance and the elastic-net weights:

.. math::

    R_{st} =
        \\frac{w_s^{T}\\,\\Sigma\\,w_t}
             {\\sqrt{(w_s^{T}\\Sigma w_s)(w_t^{T}\\Sigma w_t)}}

``R`` is regularised through a truncated SVD pseudo-inverse (the
``cutoff_*`` options control which singular values survive), and the
joint statistic is the chi-square form

.. math::

    T = z^{T}\\,R^{+}\\,z \\;\\sim\\; \\chi^2_{n_\\text{indep}}

with ``n_indep`` the number of retained components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

from .model import CovarianceDB, PredictionModel, load_covariance, load_model

__all__ = [
    "MultiXcanResult",
    "smultixcan",
    "capinv",
    "tissue_correlation_matrix",
]


# ----------------------------------------------------------------------
# Truncated pseudo-inverse (port of metax.misc.Math)
# ----------------------------------------------------------------------
def capinv(
    a: np.ndarray, rcond: float = 1e-15, epsilon: Optional[float] = None
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Modified Moore-Penrose pseudo-inverse with an *absolute* cutoff.

    Faithful port of ``metax.misc.Math.capinv`` / ``_inv``.  Singular
    values below ``rcond`` are zeroed (the largest is always kept), so
    the resulting inverse has reduced rank.

    Parameters
    ----------
    a:
        Square matrix (typically a tissue-tissue correlation matrix).
    rcond:
        Absolute singular-value cutoff.
    epsilon:
        Optional ridge term added to the diagonal before inversion.

    Returns
    -------
    inv:
        The (truncated) pseudo-inverse.
    n_indep:
        Number of retained (non-zero) singular values.
    eigen:
        The original singular-value spectrum.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        raise RuntimeError("Arrays cannot be empty")
    if epsilon is not None:
        a = a + np.diag(np.repeat(epsilon, a.shape[0]))
    a = a.conjugate()
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    m, n = u.shape[0], vt.shape[1]
    eigen = np.copy(s)
    cutoff = rcond  # absolute cutoff
    inv_s = np.zeros_like(s)
    for i in range(min(n, m)):
        # the first singular value is always kept
        if s[i] >= cutoff or i == 0:
            inv_s[i] = 1.0 / s[i]
        else:
            inv_s[i] = 0.0
    n_indep = int(np.count_nonzero(inv_s))
    res = vt.T.dot(inv_s[:, np.newaxis] * u.T)
    return res, n_indep, eigen


# ----------------------------------------------------------------------
# Cutoff strategies (port of metax.cross_model.Utilities._cutoff)
# ----------------------------------------------------------------------
def _cutoff_eigen_ratio(ratio: float) -> Callable[[np.ndarray], float]:
    """Cutoff = ``ratio`` * largest eigenvalue."""
    ratio = float(ratio)

    def _f(matrix: np.ndarray) -> float:
        if ratio == 0:
            return 0.0
        w = np.linalg.eigh(matrix)[0]
        w = -np.sort(-w)
        return ratio * w[0]

    return _f


def _cutoff_trace_ratio(ratio: float) -> Callable[[np.ndarray], float]:
    """Cutoff = ``ratio`` * trace of the matrix."""
    ratio = float(ratio)

    def _f(matrix: np.ndarray) -> float:
        if ratio == 0:
            return 0.0
        return ratio * np.trace(matrix)

    return _f


def _cutoff_threshold(threshold: float) -> Callable[[np.ndarray], float]:
    """Cutoff = smallest eigenvalue keeping ``threshold`` of total variance."""
    threshold = float(threshold)

    def _f(matrix: np.ndarray) -> float:
        if threshold == 0:
            return 0.0
        eigen = sorted(np.linalg.eigh(matrix)[0], reverse=True)
        trace = np.sum(eigen)
        cumsum = np.cumsum(eigen)
        objective = trace * (1 - threshold)
        last = eigen[0]
        for i in range(len(cumsum)):
            if cumsum[i] > objective:
                break
            last = eigen[i]
        return last

    return _f


def _resolve_cutoff(
    cutoff_condition_number: Optional[float],
    cutoff_eigen_ratio: Optional[float],
    cutoff_threshold: Optional[float],
    cutoff_trace_ratio: Optional[float],
) -> Callable[[np.ndarray], float]:
    """Pick the cutoff strategy from the supplied ``cutoff_*`` options."""
    if cutoff_eigen_ratio is not None:
        return _cutoff_eigen_ratio(cutoff_eigen_ratio)
    if cutoff_trace_ratio is not None:
        return _cutoff_trace_ratio(cutoff_trace_ratio)
    if cutoff_threshold is not None:
        return _cutoff_threshold(cutoff_threshold)
    if cutoff_condition_number is not None:
        return _cutoff_eigen_ratio(1.0 / cutoff_condition_number)
    raise ValueError(
        "Specify one of cutoff_condition_number / cutoff_eigen_ratio / "
        "cutoff_threshold / cutoff_trace_ratio"
    )


# ----------------------------------------------------------------------
# Tissue-tissue correlation
# ----------------------------------------------------------------------
def _gene_variance(
    weights: Dict[str, float], covariance: CovarianceDB, gene: str
) -> Optional[float]:
    """Compute ``w^T Sigma w`` for one tissue's model of a gene."""
    snps, matrix = covariance.get(gene, list(weights.keys()))
    if snps is None or len(snps) == 0:
        return None
    w = np.array([weights[s] for s in snps], dtype=np.float64)
    return float(w.dot(matrix).dot(w))


def _gene_coef(
    w1: Dict[str, float],
    w2: Dict[str, float],
    covariance: CovarianceDB,
    gene: str,
    var1: float,
    var2: float,
) -> Optional[float]:
    """Compute the correlation coefficient between two tissues' models."""
    s1, s2, matrix = covariance.get_2(gene, list(w1.keys()), list(w2.keys()))
    if len(s1) == 0 or len(s2) == 0:
        return None
    a = np.array([w1[s] for s in s1], dtype=np.float64)
    b = np.array([w2[s] for s in s2], dtype=np.float64)
    denom = np.sqrt(var1 * var2)
    if denom == 0:
        return np.nan
    return float(a.dot(matrix).dot(b)) / denom


def tissue_correlation_matrix(
    gene: str,
    tissue_models: Dict[str, Dict[str, float]],
    covariance: CovarianceDB,
    tissues: Sequence[str],
) -> Tuple[List[str], np.ndarray]:
    """Build the tissue-tissue correlation matrix for a gene.

    Faithful port of
    ``metax.genotype.GeneExpressionMatrixManager._build_matrix``.

    Parameters
    ----------
    gene:
        Gene identifier (as keyed in ``covariance``).
    tissue_models:
        ``tissue -> {rsid: weight}`` elastic-net weights.
    covariance:
        A single :class:`~pytwas.model.CovarianceDB` holding the SNP
        covariance for the gene across *all* tissues -- exactly the
        merged ``--snp_covariance`` input that the original S-MultiXcan
        consumes.  Off-diagonal tissue correlations need the
        cross-covariance between two tissues' SNP sets, so a single
        combined covariance is required.
    tissues:
        Candidate tissues to include (intersected with available data).

    Returns
    -------
    labels:
        Tissues actually represented in the matrix.
    matrix:
        Symmetric correlation matrix in ``labels`` order.
    """
    tissues = sorted(set(tissues) & set(tissue_models.keys()))
    variances: Dict[str, float] = {}
    for t in tissues:
        v = _gene_variance(tissue_models[t], covariance, gene)
        if v is not None:
            variances[t] = v

    coefs: Dict[str, Dict[str, float]] = {}
    labels: List[str] = []
    seen: set = set()
    for i in range(len(tissues)):
        for j in range(i, len(tissues)):
            t1, t2 = tissues[i], tissues[j]
            if t1 not in variances or t2 not in variances:
                continue
            value = _gene_coef(
                tissue_models[t1],
                tissue_models[t2],
                covariance,
                gene,
                variances[t1],
                variances[t2],
            )
            if value is None or value is np.nan:
                continue
            coefs.setdefault(t1, {})[t2] = value
            coefs.setdefault(t2, {})[t1] = value
            if t1 not in seen:
                seen.add(t1)
                labels.append(t1)

    matrix = np.array(
        [[coefs[a][b] for b in labels] for a in labels], dtype=np.float64
    ).reshape(len(labels), len(labels))
    return labels, matrix


# ----------------------------------------------------------------------
# Joint analysis
# ----------------------------------------------------------------------
@dataclass
class MultiXcanResult:
    """Per-gene S-MultiXcan joint-analysis result.

    Attributes mirror the columns of MetaXcan's ``SMulTiXcan`` output:
    ``gene, gene_name, pvalue, n, n_indep, p_i_best, t_i_best,
    p_i_worst, t_i_worst, eigen_max, eigen_min, eigen_min_kept,
    z_min, z_max, z_mean, z_sd, tmi, status``.
    """

    gene: str
    gene_name: str
    pvalue: float
    n: int
    n_indep: Optional[int]
    p_i_best: float
    t_i_best: Optional[str]
    p_i_worst: float
    t_i_worst: Optional[str]
    eigen_max: float
    eigen_min: float
    eigen_min_kept: float
    z_min: float
    z_max: float
    z_mean: float
    z_sd: float
    tmi: float
    status: int


_SMULTI_COLUMNS = [
    "gene",
    "gene_name",
    "pvalue",
    "n",
    "n_indep",
    "p_i_best",
    "t_i_best",
    "p_i_worst",
    "t_i_worst",
    "eigen_max",
    "eigen_min",
    "eigen_min_kept",
    "z_min",
    "z_max",
    "z_mean",
    "z_sd",
    "tmi",
    "status",
]


def _joint_analysis(
    gene: str,
    gene_name: str,
    zscores_by_tissue: Dict[str, float],
    labels: List[str],
    matrix: np.ndarray,
    cutoff_fn: Callable[[np.ndarray], float],
    epsilon: Optional[float],
) -> MultiXcanResult:
    """Combine per-tissue z-scores into the chi-square joint statistic.

    Faithful port of ``metax.cross_model.JointAnalysis.joint_analysis``.
    """
    tissues = list(zscores_by_tissue.keys())
    zs = np.array([zscores_by_tissue[t] for t in tissues], dtype=np.float64)
    n = len(zs)
    z_min = float(np.min(zs))
    z_max = float(np.max(zs))
    z_mean = float(np.mean(zs))
    z_sd = float(np.std(zs, ddof=1)) if n > 1 else np.nan

    none_result = MultiXcanResult(
        gene, gene_name, np.nan, n, None, np.nan, None, np.nan, None,
        np.nan, np.nan, np.nan, z_min, z_max, z_mean, z_sd, np.nan, -3,
    )
    if not labels:
        return none_result

    cutoff = cutoff_fn(matrix)

    # align z-scores to the matrix label order
    z = np.array([zscores_by_tissue[l] for l in labels], dtype=np.float64)
    inv, n_indep, eigen = capinv(matrix, cutoff, epsilon)
    eigen_max = float(np.max(eigen))
    eigen_min = float(np.min(eigen))
    eigen_min_kept = float(np.min(eigen[:n_indep]))

    absz = np.abs(z)
    maxi = int(np.argmax(absz))
    p_i_best = float(2.0 * stats.norm.sf(absz[maxi]))
    t_i_best = labels[maxi]
    mini = int(np.argmin(absz))
    p_i_worst = float(2.0 * stats.norm.sf(absz[mini]))
    t_i_worst = labels[mini]

    w = float(z.dot(inv).dot(z))
    chi2_p = float(stats.chi2.sf(w, n_indep))
    tmi = float(np.trace(matrix.dot(inv)))
    status = -4 if chi2_p == 0 else 0

    return MultiXcanResult(
        gene, gene_name, chi2_p, n, n_indep, p_i_best, t_i_best,
        p_i_worst, t_i_worst, eigen_max, eigen_min, eigen_min_kept,
        z_min, z_max, z_mean, z_sd, tmi, status,
    )


def smultixcan(
    spredixcan_results: Dict[str, pd.DataFrame],
    models: Dict[str, PredictionModel | str],
    snp_covariance: CovarianceDB | str,
    *,
    cutoff_condition_number: Optional[float] = None,
    cutoff_eigen_ratio: Optional[float] = None,
    cutoff_threshold: Optional[float] = None,
    cutoff_trace_ratio: Optional[float] = None,
    regularization: Optional[float] = None,
    trimmed_ensemble_id: bool = False,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """Run an S-MultiXcan multi-tissue joint TWAS.

    Parameters
    ----------
    spredixcan_results:
        ``tissue -> S-PrediXcan result data frame`` (output of
        :func:`pytwas.spredixcan`).  Supplies the per-tissue gene
        z-scores.
    models:
        ``tissue -> PredictionModel`` (or a path to a ``.db``).
    snp_covariance:
        A single merged SNP-covariance store -- a
        :class:`~pytwas.model.CovarianceDB` or a path to the combined
        covariance file holding, for each gene, the covariance of *every*
        SNP used by *any* tissue's model.  This is the
        ``--snp_covariance`` input of the original S-MultiXcan; a single
        combined file is required because the tissue-tissue correlation
        needs cross-covariance between two tissues' SNP sets.
    cutoff_condition_number, cutoff_eigen_ratio, cutoff_threshold,
    cutoff_trace_ratio:
        SVD-truncation strategy; provide exactly one.
    regularization:
        Optional ridge term (``epsilon``) added before inversion.
    trimmed_ensemble_id:
        Strip the ``.NN`` Ensembl version suffix from gene ids.
    output_file:
        If given, write the result table to this tab-separated path.

    Returns
    -------
    pandas.DataFrame
        Per-gene joint-analysis table sorted by ascending p-value.
    """
    cutoff_fn = _resolve_cutoff(
        cutoff_condition_number,
        cutoff_eigen_ratio,
        cutoff_threshold,
        cutoff_trace_ratio,
    )

    # resolve models / covariance
    loaded_models: Dict[str, PredictionModel] = {
        t: (load_model(m) if isinstance(m, str) else m) for t, m in models.items()
    }
    covariance: CovarianceDB = (
        load_covariance(snp_covariance)
        if isinstance(snp_covariance, str)
        else snp_covariance
    )

    # gene -> name and gene -> {tissue: {rsid: weight}}
    gene_names: Dict[str, str] = {}
    tissue_weight_maps: Dict[str, Dict[str, Dict[str, float]]] = {}
    for tissue, model in loaded_models.items():
        for _, row in model.extra.iterrows():
            g = row["gene"]
            if trimmed_ensemble_id:
                g = g.split(".")[0]
            gene_names.setdefault(g, row["gene_name"])
        wmap: Dict[str, Dict[str, float]] = {}
        for g, entries in model._weight_by_gene.items():
            key = g.split(".")[0] if trimmed_ensemble_id else g
            wmap[key] = {rsid: w for rsid, w, _, _ in entries}
        tissue_weight_maps[tissue] = wmap

    # gene -> {tissue: zscore} from the S-PrediXcan results
    gene_zscores: Dict[str, Dict[str, float]] = {}
    for tissue, df in spredixcan_results.items():
        for _, row in df.iterrows():
            z = row["zscore"]
            if not (isinstance(z, (int, float)) and np.isfinite(z)):
                continue
            g = row["gene"]
            if trimmed_ensemble_id:
                g = str(g).split(".")[0]
            gene_zscores.setdefault(g, {})[tissue] = float(z)

    results: List[MultiXcanResult] = []
    for gene in sorted(gene_zscores.keys()):
        zbt = gene_zscores[gene]
        tissues = sorted(zbt.keys())

        # per-tissue weight maps for this gene
        tissue_models = {
            t: tissue_weight_maps[t][gene]
            for t in tissues
            if t in tissue_weight_maps and gene in tissue_weight_maps[t]
        }
        # build correlation matrix
        labels, matrix = _build_corr(
            gene, tissue_models, covariance, trimmed_ensemble_id
        )
        zbt_used = {t: zbt[t] for t in labels} if labels else zbt
        gene_name = gene_names.get(gene, gene)
        results.append(
            _joint_analysis(
                gene, gene_name, zbt_used, labels, matrix, cutoff_fn, regularization
            )
        )

    df = _format_smultixcan(results)
    if output_file:
        df.to_csv(output_file, index=False, sep="\t", na_rep="NA")
    return df


def _build_corr(
    gene: str,
    tissue_models: Dict[str, Dict[str, float]],
    covariance: CovarianceDB,
    trimmed: bool,
) -> Tuple[List[str], np.ndarray]:
    """Build the tissue correlation matrix, handling Ensembl id trimming."""
    if not tissue_models:
        return [], np.zeros((0, 0))
    # covariance files may carry full ensembl ids; resolve a matching key
    cov_gene = gene
    if gene not in covariance.genes():
        for cg in covariance.genes():
            if cg.split(".")[0] == gene:
                cov_gene = cg
                break
    return tissue_correlation_matrix(
        cov_gene, tissue_models, covariance, list(tissue_models.keys())
    )


def _format_smultixcan(results: List[MultiXcanResult]) -> pd.DataFrame:
    """Assemble and sort the S-MultiXcan result table."""
    rows = []
    for r in results:
        rows.append(
            {
                "gene": r.gene,
                "gene_name": r.gene_name,
                "pvalue": r.pvalue,
                "n": r.n,
                "n_indep": r.n_indep,
                "p_i_best": r.p_i_best,
                "t_i_best": r.t_i_best,
                "p_i_worst": r.p_i_worst,
                "t_i_worst": r.t_i_worst,
                "eigen_max": r.eigen_max,
                "eigen_min": r.eigen_min,
                "eigen_min_kept": r.eigen_min_kept,
                "z_min": r.z_min,
                "z_max": r.z_max,
                "z_mean": r.z_mean,
                "z_sd": r.z_sd,
                "tmi": r.tmi,
                "status": r.status,
            }
        )
    if not rows:
        return pd.DataFrame(columns=_SMULTI_COLUMNS)
    df = pd.DataFrame(rows)[_SMULTI_COLUMNS]
    df = df.sort_values(by=["pvalue", "status"]).reset_index(drop=True)
    return df
