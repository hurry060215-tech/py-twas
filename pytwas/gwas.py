"""GWAS summary-statistics parsing and harmonisation.

Faithful reimplementation of ``metax.gwas.GWAS`` and
``metax.misc.GWASAndModels`` from the original MetaXcan software.

A GWAS summary-statistics file is read into a uniform data frame with
columns ``snp, effect_allele, non_effect_allele, zscore`` (plus optional
``beta``).  The harmonisation logic handles the many ways an association
can be specified:

* a ``zscore`` column directly, or
* a ``pvalue`` column with a sign source (``beta``, ``or``, or
  ``beta_sign``)  ->  ``z = sign * -Phi^-1(p/2)``, or
* a ``se`` column with ``beta`` or ``or``  ->  ``z = beta / se``.

Odds ratios are converted to betas via ``beta = log(or)``.  When aligning
to a prediction model, alleles are matched and the z-score / beta are
sign-flipped if the effect allele is swapped relative to the model.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

__all__ = [
    "load_gwas",
    "align_to_model",
    "beta_from_pvalue",
    "zscore_from_pvalue",
]

# canonical internal column names
SNP = "snp"
EFFECT_ALLELE = "effect_allele"
NON_EFFECT_ALLELE = "non_effect_allele"
ZSCORE = "zscore"
BETA = "beta"
OR = "or"
SE = "se"
PVALUE = "pvalue"
BETA_SIGN = "beta_sign"
CHROMOSOME = "chromosome"
POSITION = "position"

_NUMERIC = [BETA, OR, SE, PVALUE, ZSCORE]
_COMPLEMENT = str.maketrans({"C": "G", "G": "C", "T": "A", "A": "T"})


def zscore_from_pvalue(
    pvalue: np.ndarray, sign: np.ndarray, input_pvalue_fix: Optional[float] = 1e-50
) -> np.ndarray:
    """Convert a two-sided p-value + a sign into a signed z-score.

    ``z = sign * -Phi^-1(p / 2)``.  When p is so small that the inverse
    normal diverges (``abs_z = inf``) and ``input_pvalue_fix`` is set, the
    divergent entries are thresholded -- exactly as MetaXcan does.
    """
    p = np.asarray(pvalue, dtype=np.float64)
    abs_z = -stats.norm.ppf(p / 2.0)
    if np.any(np.isinf(abs_z)) and input_pvalue_fix:
        finite = np.logical_and(np.isfinite(abs_z), p != 0)
        the_min = np.min(p[finite]) if np.any(finite) else input_pvalue_fix
        if input_pvalue_fix < the_min:
            the_min = input_pvalue_fix
        fix_z = -stats.norm.ppf(the_min / 2.0)
        abs_z = abs_z.copy()
        abs_z[np.isinf(abs_z)] = fix_z
    return abs_z * np.asarray(sign, dtype=np.float64)


def beta_from_pvalue(
    pvalue: np.ndarray, sign: np.ndarray, input_pvalue_fix: Optional[float] = 1e-50
) -> np.ndarray:
    """Recover an (unscaled) beta-like effect from a p-value and sign.

    Equivalent to the signed z-score; provided for API symmetry.
    """
    return zscore_from_pvalue(pvalue, sign, input_pvalue_fix)


def _or_to_beta(odd: np.ndarray) -> np.ndarray:
    """Convert odds ratios to betas: ``beta = log(or)``."""
    odd = np.asarray(odd, dtype=np.float64)
    if np.any(odd < 0):
        raise ValueError("Odds ratios include negative values")
    return np.log(odd)


def _beta_sign(d: pd.DataFrame) -> np.ndarray:
    """Derive a per-row sign array from a ``beta`` or ``beta_sign`` column."""
    if BETA in d:
        return np.sign(d[BETA].values)
    if BETA_SIGN in d:
        b = d[BETA_SIGN]
        return b.apply(lambda x: 1.0 if (x == "+" or x == 1.0) else -1.0).values
    raise ValueError("No beta sign available in GWAS")


def load_gwas(
    source: str,
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
) -> pd.DataFrame:
    """Read and harmonise a GWAS summary-statistics file.

    Parameters
    ----------
    source:
        Path to the GWAS file (``.txt`` / ``.txt.gz`` / ``.tsv`` ...).
    snp_column, effect_allele_column, non_effect_allele_column:
        Column names for SNP id, effect allele and non-effect allele.
    zscore_column:
        Column with the signed z-score (used directly if present).
    beta_column, se_column, or_column, beta_sign_column, pvalue_column:
        Alternative association columns; the z-score is derived from them
        if ``zscore_column`` is not given.
    chromosome_column, position_column:
        Optional positional columns, carried through if present.
    separator:
        Field separator; defaults to any whitespace.
    keep_non_rsid:
        If ``False`` (default), rows whose SNP id lacks an ``rs`` prefix
        are dropped (mirrors MetaXcan).
    input_pvalue_fix:
        Threshold used to repair divergent z-scores from tiny p-values.

    Returns
    -------
    pandas.DataFrame
        Columns ``snp, effect_allele, non_effect_allele, zscore`` and
        optionally ``beta, chromosome, position``.
    """
    sep = r"\s+" if (separator is None or separator == "ANY_WHITESPACE") else separator
    raw = pd.read_csv(source, sep=sep, engine="python")

    rename: Dict[str, str] = {}
    for col, name in [
        (snp_column, SNP),
        (effect_allele_column, EFFECT_ALLELE),
        (non_effect_allele_column, NON_EFFECT_ALLELE),
        (chromosome_column, CHROMOSOME),
        (position_column, POSITION),
        (se_column, SE),
        (beta_column, BETA),
        (beta_sign_column, BETA_SIGN),
        (or_column, OR),
        (zscore_column, ZSCORE),
        (pvalue_column, PVALUE),
    ]:
        if col and col in raw.columns:
            rename[col] = name
    d = raw.rename(columns=rename)

    if SNP not in d.columns:
        raise ValueError(f"SNP column {snp_column!r} not found in GWAS file")

    # keep only rsids
    if d.shape[0] > 0:
        d = d[d[SNP].notnull()]
        if not keep_non_rsid:
            d = d[d[SNP].astype(str).str.contains("rs")]

    d = _enforce_numeric(d)
    d = _ensure_columns(d, input_pvalue_fix)
    return _keep_columns(d)


def _enforce_numeric(d: pd.DataFrame) -> pd.DataFrame:
    """Coerce the numeric association columns to ``float64``."""
    d = d.copy()
    for col in _NUMERIC:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").astype(np.float64)
    return d


def _ensure_columns(d: pd.DataFrame, input_pvalue_fix: Optional[float]) -> pd.DataFrame:
    """Uppercase alleles and ensure a ``zscore`` column exists."""
    d = d.copy()
    if d.shape[0] == 0:
        d[ZSCORE] = pd.Series(dtype=np.float64)
        return d
    d[EFFECT_ALLELE] = d[EFFECT_ALLELE].astype(str).str.upper()
    d[NON_EFFECT_ALLELE] = d[NON_EFFECT_ALLELE].astype(str).str.upper()

    if OR in d.columns:
        d[BETA] = _or_to_beta(d[OR].values)

    if ZSCORE in d.columns:
        pass  # use declared zscore
    elif PVALUE in d.columns:
        sign = _beta_sign(d)
        d[ZSCORE] = zscore_from_pvalue(d[PVALUE].values, sign, input_pvalue_fix)
    elif SE in d.columns and BETA in d.columns:
        d[ZSCORE] = d[BETA].values / d[SE].values
    else:
        raise ValueError("Could not derive a z-score from the GWAS file")

    # MetaXcan stores zscore as float32 at this stage
    d[ZSCORE] = np.asarray(d[ZSCORE], dtype=np.float32)
    return d


def _keep_columns(d: pd.DataFrame) -> pd.DataFrame:
    """Restrict to the canonical GWAS output columns."""
    keep = [SNP, EFFECT_ALLELE, NON_EFFECT_ALLELE, ZSCORE]
    for opt in (CHROMOSOME, POSITION, BETA, SE, PVALUE):
        if opt in d.columns:
            keep.append(opt)
    return d[keep].reset_index(drop=True)


def align_to_model(gwas: pd.DataFrame, model_weights: pd.DataFrame) -> pd.DataFrame:
    """Harmonise a GWAS data frame to a prediction model's allele coding.

    Faithful port of ``GWASAndModels.align_data_to_alleles``: an inner
    merge on rsid, then SNPs whose ``{effect, non-effect}`` allele set
    does not match the model are discarded, and z-score / beta are
    sign-flipped where the effect allele is swapped.

    Parameters
    ----------
    gwas:
        Output of :func:`load_gwas`.
    model_weights:
        The ``weights`` data frame of a :class:`~pytwas.model.PredictionModel`
        (``rsid, effect_allele, non_effect_allele`` are used).

    Returns
    -------
    pandas.DataFrame
        The GWAS rows aligned to the model, alleles updated to the model
        coding.
    """
    base = model_weights[["rsid", "effect_allele", "non_effect_allele"]].drop_duplicates()
    merged = pd.merge(
        gwas, base, left_on=SNP, right_on="rsid", suffixes=("", "_BASE")
    )
    if merged.shape[0] == 0:
        return merged

    ea, nea = EFFECT_ALLELE, NON_EFFECT_ALLELE
    ea_b, nea_b = ea + "_BASE", nea + "_BASE"
    alleles_1 = [frozenset(e) for e in zip(merged[ea], merged[nea])]
    alleles_2 = [frozenset(e) for e in zip(merged[ea_b], merged[nea_b])]
    eq = np.array([a == b for a, b in zip(alleles_1, alleles_2)])
    merged = merged[eq].copy()
    if merged.shape[0] == 0:
        return merged

    flipped = (merged[ea] != merged[ea_b]).values
    if ZSCORE in merged.columns:
        merged.loc[flipped, ZSCORE] = -merged.loc[flipped, ZSCORE]
    if BETA in merged.columns:
        merged.loc[flipped, BETA] = -merged.loc[flipped, BETA]
    merged.loc[flipped, ea] = merged.loc[flipped, ea_b]
    merged.loc[flipped, nea] = merged.loc[flipped, nea_b]

    return merged.drop(columns=[ea_b, nea_b, "rsid"]).reset_index(drop=True)
