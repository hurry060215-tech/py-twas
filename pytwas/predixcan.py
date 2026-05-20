"""PrediXcan: individual-level transcriptome-wide association study.

Reproduces the original MetaXcan ``PrediXcan.py`` /
``PrediXcanAssociation.py`` / ``metax.predixcan``.

PrediXcan works in two steps:

1. **Predict expression.**  For each gene, predicted expression for an
   individual is the dot product of the model's elastic-net weights with
   the individual's (allele-harmonised) genotype dosages:
   ``E_g = sum_l w_l * dosage_l``.
2. **Associate.**  Predicted expression is regressed against the
   phenotype -- ordinary least squares for a quantitative trait, or
   logistic regression for a binary trait -- and the gene's effect size,
   standard error, z-score (= t-statistic) and p-value are reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

from .model import PredictionModel, load_model

__all__ = [
    "PredixcanResult",
    "predict_expression",
    "predixcan",
]


# ----------------------------------------------------------------------
# Expression prediction
# ----------------------------------------------------------------------
def predict_expression(
    model: PredictionModel,
    dosages: pd.DataFrame,
    genes: Optional[List[str]] = None,
    dosage_alleles: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Predict per-gene expression from individual-level genotypes.

    Parameters
    ----------
    model:
        Prediction model.
    dosages:
        ``samples x SNPs`` data frame of effect-allele dosages (0--2),
        index = sample ids, columns = rsids.
    genes:
        Optional gene subset; defaults to all genes in the model.
    dosage_alleles:
        Optional data frame indexed by rsid with columns
        ``effect_allele`` / ``non_effect_allele`` describing the dosage
        coding.  When supplied, dosages whose effect allele is swapped
        relative to the model are flipped (``dosage -> 2 - dosage``);
        SNPs whose alleles cannot be matched are dropped from that gene.

    Returns
    -------
    pandas.DataFrame
        ``samples x genes`` predicted-expression matrix.
    """
    if genes is None:
        genes = model.genes()

    out: Dict[str, np.ndarray] = {}
    n = dosages.shape[0]
    for gene in genes:
        entries = model.weight_entries(gene)
        expr = np.zeros(n, dtype=np.float64)
        for rsid, weight, eff, non_eff in entries:
            if rsid not in dosages.columns:
                continue
            d = dosages[rsid].values.astype(np.float64)
            if dosage_alleles is not None and rsid in dosage_alleles.index:
                row = dosage_alleles.loc[rsid]
                sign = _dosage_sign(
                    eff, non_eff, row["effect_allele"], row["non_effect_allele"]
                )
                if sign is None:
                    continue
                if sign < 0:
                    d = 2.0 - d
            expr = expr + weight * d
        out[gene] = expr
    return pd.DataFrame(out, index=dosages.index)


_COMPLEMENT = str.maketrans({"C": "G", "G": "C", "T": "A", "A": "T"})


def _dosage_sign(
    model_eff: str, model_non: str, dos_eff: str, dos_non: str
) -> Optional[int]:
    """Return ``+1`` / ``-1`` to align a dosage to the model, or ``None``.

    Mirrors ``GWASAndModels.match_alleles`` for the SNP (single-base)
    case, including strand-complement matching.
    """
    me, mn = model_eff.upper(), model_non.upper()
    de, dn = dos_eff.upper(), dos_non.upper()
    if me == de and mn == dn:
        return 1
    if me == dn and mn == de:
        return -1
    if len(me) == 1 and len(mn) == 1:
        cde = de.translate(_COMPLEMENT)
        cdn = dn.translate(_COMPLEMENT)
        if me == cde and mn == cdn:
            return 1
        if me == cdn and mn == cde:
            return -1
    return None


# ----------------------------------------------------------------------
# Association
# ----------------------------------------------------------------------
@dataclass
class PredixcanResult:
    """Per-gene individual-level PrediXcan association result.

    Attributes
    ----------
    gene:
        Gene identifier.
    effect:
        Regression coefficient of predicted expression on the phenotype.
    se:
        Standard error of ``effect``.
    zscore:
        Wald statistic ``effect / se`` (t-statistic for OLS).
    pvalue:
        Two-sided p-value of the coefficient.
    n_samples:
        Number of samples used after dropping missing data.
    status:
        ``None`` on success, otherwise an error / non-convergence string.
    """

    gene: str
    effect: float
    se: float
    zscore: float
    pvalue: float
    n_samples: int
    status: Optional[str]


def _ols(expr: np.ndarray, pheno: np.ndarray):
    """OLS of ``pheno ~ 1 + expr``; return ``(beta, se, t, p)`` for ``expr``."""
    n = len(pheno)
    X = np.column_stack([np.ones(n), expr])
    xtx_inv = np.linalg.inv(X.T.dot(X))
    beta = xtx_inv.dot(X.T).dot(pheno)
    resid = pheno - X.dot(beta)
    dof = n - X.shape[1]
    sigma2 = resid.dot(resid) / dof
    cov = sigma2 * xtx_inv
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2.0 * stats.t.sf(np.abs(t), dof)
    return beta[1], se[1], t[1], p[1]


def _logit(expr: np.ndarray, pheno: np.ndarray, maxiter: int = 100):
    """Newton-Raphson logistic regression of ``pheno ~ 1 + expr``.

    Returns ``(beta, se, z, p, converged)`` for the ``expr`` coefficient.
    """
    n = len(pheno)
    X = np.column_stack([np.ones(n), expr])
    beta = np.zeros(2)
    converged = False
    for _ in range(maxiter):
        eta = X.dot(beta)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1.0 - mu)
        w = np.clip(w, 1e-10, None)
        grad = X.T.dot(pheno - mu)
        hess = X.T.dot(X * w[:, None])
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + delta
        if np.max(np.abs(delta)) < 1e-8:
            converged = True
            break
    eta = X.dot(beta)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1.0 - mu), 1e-10, None)
    cov = np.linalg.inv(X.T.dot(X * w[:, None]))
    se = np.sqrt(np.diag(cov))
    z = beta / se
    p = 2.0 * stats.norm.sf(np.abs(z))
    return beta[1], se[1], z[1], p[1], converged


def predixcan_association(
    gene: str, expression: np.ndarray, pheno: np.ndarray, mode: str = "linear"
) -> PredixcanResult:
    """Associate a single gene's predicted expression with the phenotype.

    Parameters
    ----------
    gene:
        Gene identifier.
    expression:
        Predicted-expression vector.
    pheno:
        Phenotype vector (same length as ``expression``).
    mode:
        ``"linear"`` (OLS) or ``"logistic"``.

    Returns
    -------
    PredixcanResult
    """
    expr = np.asarray(expression, dtype=np.float64)
    y = np.asarray(pheno, dtype=np.float64)
    keep = np.isfinite(expr) & np.isfinite(y)
    expr, y = expr[keep], y[keep]
    n = len(y)

    if n < 3 or np.all(expr == expr[0]):
        return PredixcanResult(gene, np.nan, np.nan, np.nan, np.nan, n,
                               "insufficient_data")
    try:
        if mode == "linear":
            beta, se, z, p = _ols(expr, y)
            status = None
        elif mode == "logistic":
            beta, se, z, p, conv = _logit(expr, y)
            status = None if conv else "MLE_did_not_converge"
        else:
            raise ValueError(f"Unknown mode {mode!r}")
    except Exception as exc:  # pragma: no cover - defensive
        return PredixcanResult(gene, np.nan, np.nan, np.nan, np.nan, n,
                               str(exc).replace(" ", "_"))
    return PredixcanResult(gene, beta, se, z, p, n, status)


_PRED_COLUMNS = ["gene", "gene_name", "effect", "se", "zscore", "pvalue",
                 "n_samples", "status"]


def predixcan(
    model_db_path: str | PredictionModel,
    dosages: pd.DataFrame,
    pheno: np.ndarray,
    *,
    mode: str = "linear",
    dosage_alleles: Optional[pd.DataFrame] = None,
    genes: Optional[List[str]] = None,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """Run an individual-level PrediXcan TWAS.

    Parameters
    ----------
    model_db_path:
        Path to a prediction-model ``.db``, or a loaded
        :class:`~pytwas.model.PredictionModel`.
    dosages:
        ``samples x SNPs`` effect-allele dosage matrix.
    pheno:
        Phenotype vector, one entry per sample (row of ``dosages``).
    mode:
        ``"linear"`` (quantitative trait) or ``"logistic"`` (binary).
    dosage_alleles:
        Optional rsid-indexed allele table for dosage harmonisation
        (see :func:`predict_expression`).
    genes:
        Optional gene subset.
    output_file:
        If given, write the result table to this CSV path.

    Returns
    -------
    pandas.DataFrame
        Per-gene association table sorted by ascending p-value, with
        columns ``gene, gene_name, effect, se, zscore, pvalue,
        n_samples, status``.
    """
    model = (
        load_model(model_db_path)
        if isinstance(model_db_path, str)
        else model_db_path
    )
    if genes is None:
        genes = model.genes()

    expr = predict_expression(model, dosages, genes, dosage_alleles)
    pheno = np.asarray(pheno, dtype=np.float64)

    results = [
        predixcan_association(g, expr[g].values, pheno, mode=mode) for g in genes
    ]

    name_map = dict(zip(model.extra.gene, model.extra.gene_name))
    rows = [
        {
            "gene": r.gene,
            "gene_name": name_map.get(r.gene, r.gene),
            "effect": r.effect,
            "se": r.se,
            "zscore": r.zscore,
            "pvalue": r.pvalue,
            "n_samples": r.n_samples,
            "status": r.status,
        }
        for r in results
    ]
    df = pd.DataFrame(rows, columns=_PRED_COLUMNS)
    df = df.sort_values(by="pvalue", na_position="last").reset_index(drop=True)
    if output_file:
        df.to_csv(output_file, index=False, na_rep="NA")
    return df
