"""Numerical-parity tests of pytwas against the original MetaXcan.

The reference is the original MetaXcan software
(https://github.com/hakyimlab/MetaXcan).  If a clone is available under
``/tmp/metaxcan_ref`` (or ``$METAXCAN_REF``) the original ``M03_betas`` +
``M04_zscores`` (S-PrediXcan) and ``SMulTiXcan`` are executed live and
compared cell-by-cell with pytwas output.

When the original cannot be run, the tests fall back to **gold reference
CSVs** committed under ``tests/data/`` (produced once by the original
MetaXcan on the bundled sample data) so parity is still checked.

The S-PrediXcan z-score is a deterministic closed-form combination, so
parity is expected to machine precision (rel-diff < 1e-9).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

import pytwas

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# locate an optional MetaXcan checkout for live comparison
_METAXCAN_REF = os.environ.get("METAXCAN_REF", "/tmp/metaxcan_ref")
_METAXCAN_SW = os.path.join(_METAXCAN_REF, "software")
_HAS_METAXCAN = os.path.isfile(os.path.join(_METAXCAN_SW, "M04_zscores.py"))


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _run_original_spredixcan(model_db, covariance, gwas_file, gwas_format):
    """Run the original MetaXcan M03+M04 pipeline; return the result frame."""
    import copy
    import logging

    sys.path.insert(0, _METAXCAN_SW)
    import M03_betas  # noqa: E402
    import M04_zscores  # noqa: E402

    logging.disable(logging.CRITICAL)

    class _A:
        pass

    a = _A()
    defaults = dict(
        model_db_path=model_db, model_db_snp_key=None, covariance=covariance,
        gwas_file=gwas_file, gwas_folder=None, gwas_file_pattern=None,
        snp_column="SNP", effect_allele_column="A1", non_effect_allele_column="A2",
        chromosome_column=None, position_column=None, freq_column=None,
        beta_column=None, beta_sign_column=None, or_column=None, se_column=None,
        zscore_column=None, pvalue_column=None, separator=None,
        skip_until_header=None, handle_empty_columns=False, input_pvalue_fix=1e-50,
        keep_non_rsid=False, snp_map_file=None, split_column=None,
        input_gwas_format_json=None, output_folder=None, output=None,
        output_file=None, single_snp_model=False, stream_covariance=False,
        remove_ens_version=False, overwrite=True, additional_output=True,
        MAX_R=None, throw=True, verbosity=50, gwas_h2=None, gwas_N=None,
    )
    defaults.update(gwas_format)
    for k, v in defaults.items():
        setattr(a, k, v)
    g = M03_betas.run(copy.copy(a))
    return M04_zscores.run(a, g)


def _assert_frames_match(mine: pd.DataFrame, gold: pd.DataFrame,
                         numeric_cols, int_cols, tol=1e-6):
    """Assert two result frames agree cell-by-cell after aligning on gene."""
    mine = mine.set_index("gene")
    gold = gold.set_index("gene")
    assert set(mine.index) == set(gold.index), (
        f"gene sets differ: {set(mine.index) ^ set(gold.index)}"
    )
    max_abs = 0.0
    for g in gold.index:
        for col in numeric_cols:
            a, b = mine.loc[g, col], gold.loc[g, col]
            a = np.nan if (isinstance(a, str) or pd.isna(a)) else float(a)
            b = np.nan if (isinstance(b, str) or pd.isna(b)) else float(b)
            if np.isnan(a) and np.isnan(b):
                continue
            assert not (np.isnan(a) ^ np.isnan(b)), f"{g}.{col}: {a} vs {b}"
            d = abs(a - b)
            max_abs = max(max_abs, d)
            assert d <= tol * (abs(b) + 1.0), f"{g}.{col}: {a} vs {b} (d={d})"
        for col in int_cols:
            a, b = mine.loc[g, col], gold.loc[g, col]
            if pd.isna(a) and pd.isna(b):
                continue
            assert int(a) == int(b), f"{g}.{col}: {a} vs {b}"
    return max_abs


# ----------------------------------------------------------------------
# S-PrediXcan parity
# ----------------------------------------------------------------------
def test_spredixcan_parity_vs_gold():
    """pytwas S-PrediXcan matches the committed MetaXcan gold output."""
    gold_path = os.path.join(DATA, "spx_gold.csv")
    if not os.path.isfile(gold_path):
        pytest.skip("gold reference CSV not available")
    gold = pd.read_csv(gold_path)

    mine = pytwas.spredixcan(
        model_db_path=os.path.join(DATA, "sample_model.db"),
        covariance=os.path.join(DATA, "sample_cov.txt.gz"),
        gwas_file=os.path.join(DATA, "sample_gwas.txt.gz"),
        snp_column="SNP", effect_allele_column="A1", non_effect_allele_column="A2",
        zscore_column="ZSCORE", beta_column="BETA", additional_output=True,
    )
    max_abs = _assert_frames_match(
        mine, gold,
        numeric_cols=["zscore", "effect_size", "pvalue", "var_g",
                      "best_gwas_p", "largest_weight"],
        int_cols=["n_snps_used", "n_snps_in_model"],
    )
    # closed-form: parity is to machine precision
    assert max_abs < 1e-9


@pytest.mark.skipif(not _HAS_METAXCAN,
                    reason="MetaXcan reference checkout not available")
def test_spredixcan_parity_live():
    """pytwas matches a *live* run of the original MetaXcan M03+M04."""
    model_db = os.path.join(DATA, "sample_model.db")
    cov = os.path.join(DATA, "sample_cov.txt.gz")
    gwas = os.path.join(DATA, "sample_gwas.txt.gz")

    gold = _run_original_spredixcan(
        model_db, cov, gwas,
        gwas_format=dict(snp_column="SNP", effect_allele_column="A1",
                         non_effect_allele_column="A2",
                         zscore_column="ZSCORE", beta_column="BETA"),
    )
    mine = pytwas.spredixcan(
        model_db_path=model_db, covariance=cov, gwas_file=gwas,
        snp_column="SNP", effect_allele_column="A1", non_effect_allele_column="A2",
        zscore_column="ZSCORE", beta_column="BETA", additional_output=True,
    )
    max_abs = _assert_frames_match(
        mine, gold,
        numeric_cols=["zscore", "effect_size", "pvalue", "var_g",
                      "best_gwas_p", "largest_weight"],
        int_cols=["n_snps_used", "n_snps_in_model"],
    )
    assert max_abs < 1e-9

    # row order must match too (both sort by ascending p-value)
    assert list(mine.gene) == list(gold.gene)

    # Pearson correlation of z-scores
    m = mine.set_index("gene"); g = gold.set_index("gene")
    common = [x for x in g.index
              if np.isfinite(pd.to_numeric(g.loc[x, "zscore"], errors="coerce"))
              and np.isfinite(m.loc[x, "zscore"])]
    if len(common) > 1:
        zm = m.loc[common, "zscore"].astype(float).values
        zg = pd.to_numeric(g.loc[common, "zscore"]).astype(float).values
        r = np.corrcoef(zm, zg)[0, 1]
        assert r > 0.9999


@pytest.mark.skipif(not _HAS_METAXCAN,
                    reason="MetaXcan reference checkout not available")
def test_spredixcan_parity_pvalue_path():
    """The beta-from-pvalue harmonisation path matches MetaXcan exactly."""
    import scipy.stats as stats

    model_db = os.path.join(DATA, "sample_model.db")
    cov = os.path.join(DATA, "sample_cov.txt.gz")

    # build a pvalue+beta GWAS from the sample GWAS
    gwas = pd.read_csv(os.path.join(DATA, "sample_gwas.txt.gz"), sep="\t")
    gwas["P"] = 2 * stats.norm.sf(np.abs(gwas["ZSCORE"]))
    tmp = os.path.join(DATA, "_tmp_pval_gwas.txt.gz")
    gwas[["SNP", "A1", "A2", "BETA", "P"]].to_csv(
        tmp, sep="\t", index=False, compression="gzip"
    )
    try:
        gold = _run_original_spredixcan(
            model_db, cov, tmp,
            gwas_format=dict(snp_column="SNP", effect_allele_column="A1",
                             non_effect_allele_column="A2",
                             beta_column="BETA", pvalue_column="P"),
        )
        mine = pytwas.spredixcan(
            model_db_path=model_db, covariance=cov, gwas_file=tmp,
            snp_column="SNP", effect_allele_column="A1",
            non_effect_allele_column="A2", beta_column="BETA", pvalue_column="P",
        )
        max_abs = _assert_frames_match(
            mine, gold,
            numeric_cols=["zscore", "effect_size", "pvalue", "var_g"],
            int_cols=["n_snps_used", "n_snps_in_model"],
        )
        assert max_abs < 1e-9
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ----------------------------------------------------------------------
# S-MultiXcan parity
# ----------------------------------------------------------------------
def test_smultixcan_parity_vs_gold():
    """pytwas S-MultiXcan matches the committed MetaXcan gold output."""
    gold_path = os.path.join(DATA, "smx_gold.txt")
    smx_dir = os.path.join(DATA, "smx_data")
    if not (os.path.isfile(gold_path) and os.path.isdir(smx_dir)):
        pytest.skip("S-MultiXcan gold reference not available")

    import glob

    gold = pd.read_csv(gold_path, sep="\t")
    spx = {
        os.path.basename(f).split("__PM__")[1].replace(".csv", ""): pd.read_csv(f)
        for f in sorted(glob.glob(os.path.join(smx_dir, "spx", "*.csv")))
    }
    models = {
        os.path.basename(db).replace(".db", ""): pytwas.load_model(db)
        for db in sorted(glob.glob(os.path.join(smx_dir, "models", "*.db")))
    }
    cov = os.path.join(smx_dir, "snp_cov.txt.gz")

    mine = pytwas.smultixcan(spx, models, cov, cutoff_condition_number=30)
    max_abs = _assert_frames_match(
        mine, gold,
        numeric_cols=["pvalue", "p_i_best", "p_i_worst", "eigen_max",
                      "eigen_min", "eigen_min_kept", "z_min", "z_max",
                      "z_mean", "z_sd", "tmi"],
        int_cols=["n", "n_indep", "status"],
    )
    assert max_abs < 1e-6


@pytest.mark.skipif(not _HAS_METAXCAN,
                    reason="MetaXcan reference checkout not available")
def test_smultixcan_parity_live():
    """pytwas S-MultiXcan matches a *live* run of the original SMulTiXcan."""
    import glob
    import logging
    import shutil
    import tempfile

    smx_dir = os.path.join(DATA, "smx_data")
    if not os.path.isdir(smx_dir):
        pytest.skip("S-MultiXcan test data not available")

    sys.path.insert(0, _METAXCAN_SW)
    from metax.cross_model import Utilities as CMU  # noqa: E402
    from metax.cross_model import JointAnalysis  # noqa: E402

    logging.disable(logging.CRITICAL)

    workdir = tempfile.mkdtemp()
    try:
        # rename S-PrediXcan CSVs into pheno__PM__<tissue>.csv layout
        spx_dir = os.path.join(workdir, "spx")
        os.makedirs(spx_dir)
        for f in glob.glob(os.path.join(smx_dir, "spx", "*.csv")):
            shutil.copy(f, os.path.join(spx_dir, os.path.basename(f)))

        class _A:
            pass

        a = _A()
        for k, v in dict(
            models_folder=os.path.join(smx_dir, "models"),
            models_name_filter=None, models_name_pattern=None,
            model_db_snp_key=None,
            cleared_snps=os.path.join(smx_dir, "cleared_snps.txt"),
            snp_covariance=os.path.join(smx_dir, "snp_cov.txt.gz"),
            metaxcan_folder=spx_dir, metaxcan_filter=[".*csv"],
            metaxcan_file_name_parse_pattern="(.*)__PM__(.*).csv",
            model_product=None, regularization=None,
            cutoff_condition_number=30.0, cutoff_eigen_ratio=None,
            cutoff_threshold=None, cutoff_trace_ratio=None,
            trimmed_ensemble_id=False, MAX_M=None, throw=True,
        ).items():
            setattr(a, k, v)

        context = CMU.context_from_args(a)
        rows = [JointAnalysis.joint_analysis(context, g)
                for g in context.get_genes()]
        gold = JointAnalysis.format_results(rows)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    spx = {
        os.path.basename(f).split("__PM__")[1].replace(".csv", ""): pd.read_csv(f)
        for f in sorted(glob.glob(os.path.join(smx_dir, "spx", "*.csv")))
    }
    models = {
        os.path.basename(db).replace(".db", ""): pytwas.load_model(db)
        for db in sorted(glob.glob(os.path.join(smx_dir, "models", "*.db")))
    }
    mine = pytwas.smultixcan(
        spx, models, os.path.join(smx_dir, "snp_cov.txt.gz"),
        cutoff_condition_number=30,
    )
    max_abs = _assert_frames_match(
        mine, gold,
        numeric_cols=["pvalue", "p_i_best", "p_i_worst", "eigen_max",
                      "eigen_min", "eigen_min_kept", "z_min", "z_max",
                      "z_mean", "z_sd", "tmi"],
        int_cols=["n", "n_indep", "status"],
    )
    assert max_abs < 1e-6
