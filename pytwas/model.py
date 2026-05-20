"""Prediction-model (PrediXcan/MetaXcan ``.db``) and SNP-covariance readers.

This module is a faithful, dependency-light reimplementation of
``metax.PredictionModel`` and ``metax.MatrixManager`` from the original
MetaXcan software (https://github.com/hakyimlab/MetaXcan).

A PrediXcan/MetaXcan elastic-net prediction model is a SQLite database with
two tables:

``weights``
    one row per SNP-in-a-gene-model:
    ``rsid, gene, weight, ref_allele, eff_allele``.
``extra``
    one row per gene:
    ``gene, genename, n.snps.in.model, pred.perf.R2, pred.perf.pval,
    pred.perf.qval`` (column names use dots, hence the back-ticks).

The SNP covariance is a (optionally gzip-compressed) whitespace-delimited
text file with columns ``GENE RSID1 RSID2 VALUE`` -- the LD covariance of
the model SNPs, estimated from a reference panel (e.g. 1000 Genomes).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "PredictionModel",
    "load_model",
    "CovarianceDB",
    "load_covariance",
]


# ----------------------------------------------------------------------
# Prediction model
# ----------------------------------------------------------------------
#: canonical column order of the ``weights`` data frame
WEIGHT_COLUMNS = ["rsid", "gene", "weight", "effect_allele", "non_effect_allele"]
#: canonical column order of the ``extra`` data frame
EXTRA_COLUMNS = [
    "gene",
    "gene_name",
    "n_snps_in_model",
    "pred_perf_r2",
    "pred_perf_pval",
    "pred_perf_qval",
]


@dataclass
class PredictionModel:
    """An elastic-net SNP -> expression prediction model.

    Attributes
    ----------
    weights:
        Data frame with columns ``rsid, gene, weight, effect_allele,
        non_effect_allele``.
    extra:
        Per-gene metadata: ``gene, gene_name, n_snps_in_model,
        pred_perf_r2, pred_perf_pval, pred_perf_qval``.
    """

    weights: pd.DataFrame
    extra: pd.DataFrame
    #: lazily-built ``gene -> [(rsid, weight, eff, non_eff), ...]`` map
    _weight_by_gene: Dict[str, List[tuple]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self._weight_by_gene:
            self._weight_by_gene = {}
            for row in self.weights.itertuples(index=False):
                self._weight_by_gene.setdefault(row.gene, []).append(
                    (row.rsid, row.weight, row.effect_allele, row.non_effect_allele)
                )

    # -- accessors -----------------------------------------------------
    def snps(self) -> set:
        """Return the set of all rsids used by any model in the database."""
        return set(self.weights.rsid.values)

    def genes(self) -> List[str]:
        """Return the genes in database (``extra``-table) order."""
        return list(self.extra.gene.values)

    def weights_for_gene(self, gene: str) -> pd.DataFrame:
        """Return the ``weights`` sub-frame for a single gene."""
        return self.weights[self.weights.gene == gene]

    def weight_entries(self, gene: str) -> List[tuple]:
        """Return ``[(rsid, weight, eff_allele, non_eff_allele), ...]``."""
        return self._weight_by_gene.get(gene, [])


def _load_extra(cursor: sqlite3.Cursor) -> pd.DataFrame:
    """Read the ``extra`` table, tolerating optional/missing columns."""
    cols = {r[1] for r in cursor.execute("PRAGMA table_info(extra)")}
    # column names use dots; quote with back-ticks
    select, names = [], []
    for sql_name, out_name in [
        ("gene", "gene"),
        ("genename", "gene_name"),
        ("`n.snps.in.model`", "n_snps_in_model"),
        ("`pred.perf.R2`", "pred_perf_r2"),
        ("`pred.perf.pval`", "pred_perf_pval"),
        ("`pred.perf.qval`", "pred_perf_qval"),
    ]:
        bare = sql_name.strip("`")
        if bare in cols:
            select.append(sql_name)
            names.append(out_name)
    rows = list(cursor.execute(f"SELECT {', '.join(select)} FROM extra ORDER BY gene"))
    df = pd.DataFrame(rows, columns=names)
    for missing in EXTRA_COLUMNS:
        if missing not in df.columns:
            df[missing] = np.nan
    return df[EXTRA_COLUMNS]


def load_model(path: str, snp_key: Optional[str] = None) -> PredictionModel:
    """Load a PrediXcan/MetaXcan prediction-model SQLite database.

    Parameters
    ----------
    path:
        Path to the ``.db`` file.
    snp_key:
        Optional alternative column in the ``weights`` table to use as the
        SNP id (mirrors MetaXcan's ``--model_db_snp_key``).  Defaults to
        ``rsid``.

    Returns
    -------
    PredictionModel
    """
    conn = sqlite3.connect(path)
    try:
        cur = conn.cursor()
        key = snp_key if snp_key else "rsid"
        wrows = list(
            cur.execute(
                f"SELECT {key}, gene, weight, ref_allele, eff_allele FROM weights"
            )
        )
        weights = pd.DataFrame(
            wrows,
            columns=["rsid", "gene", "weight", "non_effect_allele", "effect_allele"],
        )[WEIGHT_COLUMNS]
        weights["weight"] = weights["weight"].astype(np.float64)
        extra = _load_extra(cur)
    finally:
        conn.close()
    return PredictionModel(weights=weights, extra=extra)


# ----------------------------------------------------------------------
# SNP covariance
# ----------------------------------------------------------------------
class CovarianceDB:
    """In-memory store of per-gene SNP covariance matrices.

    Faithful port of ``metax.MatrixManager``.  Entries for a gene must be
    contiguous in the input file (this is validated).  Symmetric: a
    ``(RSID1, RSID2)`` row also defines ``(RSID2, RSID1)``.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        _validate_covariance(df)
        self._data: Dict[str, List[tuple]] = {}
        # NaN values are coded to the string "NA" (as in MetaXcan)
        df = df.fillna("NA")
        for gene, r1, r2, val in zip(
            df["GENE"].values, df["RSID1"].values, df["RSID2"].values, df["VALUE"].values
        ):
            self._data.setdefault(gene, []).append((gene, r1, r2, val))

    # -- public API ----------------------------------------------------
    def genes(self) -> set:
        """Return the set of genes with covariance data."""
        return set(self._data.keys())

    def n_snps(self, gene: str) -> float:
        """Number of distinct SNPs (with non-NA covariance) for ``gene``."""
        if gene not in self._data:
            return np.nan
        snps = {t[1] for t in self._data[gene] if t[3] != "NA"}
        return len(snps)

    def get(
        self, gene: str, whitelist: Optional[Sequence[str]] = None
    ) -> Tuple[Optional[List[str]], Optional[np.ndarray]]:
        """Return ``(snps, covariance_matrix)`` for a gene.

        ``snps`` preserves first-seen order.  Only SNPs in ``whitelist``
        (if given) and with non-NA covariance are kept.  Returns
        ``(None, None)`` if the gene is absent.
        """
        if gene not in self._data:
            return None, None
        wl = set(whitelist) if whitelist is not None else None
        snps, matrix = _rows_to_matrix(self._data[gene], wl)
        return snps, matrix

    def get_2(
        self, gene: str, snps_1: Sequence[str], snps_2: Sequence[str]
    ) -> Tuple[List[str], List[str], np.ndarray]:
        """Return the cross-covariance block between two SNP sets.

        Used by S-MultiXcan to build tissue-tissue correlations.
        """
        if gene not in self._data:
            return [], [], np.zeros((0, 0))
        whitelist = set(snps_1) | set(snps_2)
        entries: Dict[str, Dict[str, float]] = {}
        present: set = set()
        for _, id1, id2, value in self._data[gene]:
            if id1 not in whitelist or id2 not in whitelist:
                continue
            if value == "NA":
                continue
            v = float(value)
            entries.setdefault(id1, {})[id2] = v
            entries.setdefault(id2, {})[id1] = v
            present.add(id1)
            present.add(id2)
        is1 = sorted(s for s in set(snps_1) if s in present)
        is2 = sorted(s for s in set(snps_2) if s in present)
        matrix = _entries_to_matrix(entries, is1, is2)
        return is1, is2, matrix


def _rows_to_matrix(
    rows: List[tuple], whitelist: Optional[set]
) -> Tuple[List[str], np.ndarray]:
    """Build a symmetric covariance matrix from triplet rows."""
    entries: Dict[str, Dict[str, float]] = {}
    ids: List[str] = []
    seen: set = set()
    for _, id1, id2, value in rows:
        if whitelist is not None:
            if id1 not in whitelist or id2 not in whitelist:
                continue
        if value == "NA":
            continue
        v = float(value)
        entries.setdefault(id1, {})[id2] = v
        entries.setdefault(id2, {})[id1] = v
        if id1 not in seen:
            seen.add(id1)
            ids.append(id1)
    matrix = _entries_to_matrix(entries, ids, ids)
    return ids, matrix


def _entries_to_matrix(
    entries: Dict[str, Dict[str, float]],
    keys_i: Sequence[str],
    keys_j: Sequence[str],
) -> np.ndarray:
    """Materialise a dense matrix from the nested-dict covariance store."""
    rows = [[entries[ki][kj] for kj in keys_j] for ki in keys_i]
    return np.array(rows, dtype=np.float64).reshape(len(keys_i), len(keys_j))


def _validate_covariance(df: pd.DataFrame) -> None:
    """Validate gene contiguity and absence of duplicate rows."""
    seen: set = set()
    last = object()
    for k in df["GENE"]:
        if k != last:
            if k in seen:
                raise ValueError(
                    f"Covariance entries for gene {k!r} are not contiguous"
                )
            seen.add(k)
            last = k
    if df.duplicated().any():
        raise ValueError("Duplicated entries found in covariance file")


def load_covariance(path: str) -> CovarianceDB:
    """Load a SNP-covariance ``.txt`` / ``.txt.gz`` file.

    Parameters
    ----------
    path:
        Whitespace-delimited file with columns ``GENE RSID1 RSID2 VALUE``.
        gzip-compressed files (``.gz``) are read transparently.
    """
    df = pd.read_csv(path, sep=r"\s+", dtype={"GENE": str, "RSID1": str, "RSID2": str})
    return CovarianceDB(df)
