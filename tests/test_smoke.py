"""Algorithmic smoke tests for pytwas -- no MetaXcan reference required.

These build tiny synthetic prediction models, GWAS files and genotype
matrices and check the internal consistency of each ported routine
against hand-derived expectations.
"""
from __future__ import annotations

import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
import pytest

import pytwas

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
def _make_model_db(path: str) -> None:
    """Write a tiny two-gene prediction-model SQLite database."""
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(
        "CREATE TABLE weights (rsid TEXT, gene TEXT, weight DOUBLE, "
        "ref_allele CHARACTER, eff_allele CHARACTER)"
    )
    c.execute(
        "CREATE TABLE extra (gene TEXT, genename TEXT, `n.snps.in.model` INTEGER, "
        "`pred.perf.R2` DOUBLE, `pred.perf.pval` DOUBLE, `pred.perf.qval` DOUBLE)"
    )
    weights = [
        ("rs1", "GENE_A", 0.5, "G", "A"),
        ("rs2", "GENE_A", -0.3, "T", "C"),
        ("rs3", "GENE_A", 0.2, "C", "T"),
        ("rs4", "GENE_B", 0.4, "A", "G"),
        ("rs5", "GENE_B", 0.1, "G", "A"),
    ]
    for w in weights:
        c.execute("INSERT INTO weights VALUES (?,?,?,?,?)", w)
    c.execute("INSERT INTO extra VALUES (?,?,?,?,?,?)",
              ("GENE_A", "A", 3, 0.4, 0.04, 0.04))
    c.execute("INSERT INTO extra VALUES (?,?,?,?,?,?)",
              ("GENE_B", "B", 2, 0.2, 0.02, 0.02))
    conn.commit()
    conn.close()


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """A complete synthetic model + covariance + GWAS triple."""
    d = tmp_path_factory.mktemp("twas")
    db = str(d / "model.db")
    _make_model_db(db)

    # complete covariance over each gene's SNP set
    rng = np.random.default_rng(0)
    cov_rows = []
    for gene, snps in [("GENE_A", ["rs1", "rs2", "rs3"]),
                       ("GENE_B", ["rs4", "rs5"])]:
        n = len(snps)
        A = rng.normal(0, 1, (n, n))
        S = A @ A.T / n + np.eye(n) * 0.5
        for i in range(n):
            for j in range(i, n):
                cov_rows.append((gene, snps[i], snps[j], S[i, j]))
    cov_path = str(d / "cov.txt.gz")
    pd.DataFrame(cov_rows, columns=["GENE", "RSID1", "RSID2", "VALUE"]).to_csv(
        cov_path, sep=" ", index=False, compression="gzip"
    )

    # GWAS with z-scores, alleles matching the model
    gwas = pd.DataFrame({
        "SNP": ["rs1", "rs2", "rs3", "rs4", "rs5"],
        "A1": ["A", "C", "T", "G", "A"],
        "A2": ["G", "T", "C", "A", "G"],
        "ZSCORE": [1.5, -0.8, 2.2, 0.3, -1.1],
        "BETA": [0.05, -0.02, 0.08, 0.01, -0.03],
    })
    gwas_path = str(d / "gwas.txt.gz")
    gwas.to_csv(gwas_path, sep="\t", index=False, compression="gzip")
    return {"db": db, "cov": cov_path, "gwas": gwas_path, "dir": str(d)}


# ----------------------------------------------------------------------
# model / covariance readers
# ----------------------------------------------------------------------
def test_load_model(synthetic):
    """The model reader exposes weights, extra and the gene list."""
    model = pytwas.load_model(synthetic["db"])
    assert set(model.genes()) == {"GENE_A", "GENE_B"}
    assert model.snps() == {"rs1", "rs2", "rs3", "rs4", "rs5"}
    assert list(model.weights.columns) == [
        "rsid", "gene", "weight", "effect_allele", "non_effect_allele"
    ]
    assert model.weights_for_gene("GENE_A").shape[0] == 3
    entries = model.weight_entries("GENE_A")
    assert len(entries) == 3 and entries[0][0] == "rs1"


def test_load_covariance(synthetic):
    """The covariance reader returns symmetric, ordered matrices."""
    cov = pytwas.load_covariance(synthetic["cov"])
    assert cov.genes() == {"GENE_A", "GENE_B"}
    snps, m = cov.get("GENE_A")
    assert snps == ["rs1", "rs2", "rs3"]
    assert m.shape == (3, 3)
    np.testing.assert_allclose(m, m.T)  # symmetric
    # whitelist subsetting
    snps2, m2 = cov.get("GENE_A", ["rs1", "rs3"])
    assert snps2 == ["rs1", "rs3"]
    assert m2.shape == (2, 2)
    assert cov.n_snps("GENE_A") == 3


# ----------------------------------------------------------------------
# GWAS parsing / harmonisation
# ----------------------------------------------------------------------
def test_load_gwas_zscore(synthetic):
    """A GWAS with a z-score column is read directly."""
    g = pytwas.load_gwas(
        synthetic["gwas"], snp_column="SNP", effect_allele_column="A1",
        non_effect_allele_column="A2", zscore_column="ZSCORE", beta_column="BETA",
    )
    assert list(g.columns)[:4] == ["snp", "effect_allele", "non_effect_allele", "zscore"]
    assert g.shape[0] == 5
    np.testing.assert_allclose(sorted(g.zscore), sorted([1.5, -0.8, 2.2, 0.3, -1.1]))


def test_zscore_from_pvalue():
    """z = sign * -Phi^-1(p/2) inverts a known z-score round-trip."""
    import scipy.stats as stats

    z_true = np.array([1.5, -2.3, 0.4, -0.9])
    p = 2 * stats.norm.sf(np.abs(z_true))
    sign = np.sign(z_true)
    z = pytwas.zscore_from_pvalue(p, sign)
    np.testing.assert_allclose(z, z_true, rtol=1e-10)


def test_load_gwas_pvalue_path(synthetic):
    """A GWAS with pvalue+beta yields the same z as the direct z-score."""
    import scipy.stats as stats

    gwas = pd.read_csv(synthetic["gwas"], sep="\t")
    gwas["P"] = 2 * stats.norm.sf(np.abs(gwas["ZSCORE"]))
    p = os.path.join(synthetic["dir"], "gwas_p.txt.gz")
    gwas[["SNP", "A1", "A2", "BETA", "P"]].to_csv(
        p, sep="\t", index=False, compression="gzip"
    )
    g = pytwas.load_gwas(
        p, snp_column="SNP", effect_allele_column="A1",
        non_effect_allele_column="A2", beta_column="BETA", pvalue_column="P",
    )
    direct = pytwas.load_gwas(
        synthetic["gwas"], snp_column="SNP", effect_allele_column="A1",
        non_effect_allele_column="A2", zscore_column="ZSCORE",
    )
    merged = g.merge(direct, on="snp", suffixes=("_p", "_z"))
    np.testing.assert_allclose(
        merged.zscore_p.values, merged.zscore_z.values, rtol=1e-5
    )


def test_align_to_model_flips_swapped_alleles(synthetic):
    """A GWAS SNP with swapped effect allele has its z-score sign-flipped."""
    model = pytwas.load_model(synthetic["db"])
    # rs1 model coding is eff=A / non=G; provide swapped GWAS coding
    gwas = pd.DataFrame({
        "snp": ["rs1"],
        "effect_allele": ["G"],
        "non_effect_allele": ["A"],
        "zscore": np.array([2.0], dtype=np.float32),
        "beta": [0.1],
    })
    aligned = pytwas.align_to_model(gwas, model.weights)
    assert aligned.shape[0] == 1
    assert aligned.effect_allele.iloc[0] == "A"  # flipped to model coding
    assert aligned.zscore.iloc[0] == pytest.approx(-2.0)
    assert aligned.beta.iloc[0] == pytest.approx(-0.1)


# ----------------------------------------------------------------------
# S-PrediXcan
# ----------------------------------------------------------------------
def test_spredixcan_formula(synthetic):
    """The S-PrediXcan z-score equals the closed-form combination."""
    res = pytwas.spredixcan(
        model_db_path=synthetic["db"], covariance=synthetic["cov"],
        gwas_file=synthetic["gwas"], snp_column="SNP",
        effect_allele_column="A1", non_effect_allele_column="A2",
        zscore_column="ZSCORE", beta_column="BETA",
    )
    assert set(res.gene) == {"GENE_A", "GENE_B"}
    # all genes have all SNPs in model / covariance / GWAS
    a = res.set_index("gene").loc["GENE_A"]
    assert a.n_snps_used == 3
    assert a.n_snps_in_model == 3

    # recompute GENE_A z by hand from the closed-form formula
    model = pytwas.load_model(synthetic["db"])
    cov = pytwas.load_covariance(synthetic["cov"])
    snps, m = cov.get("GENE_A")
    w_map = {r: w for r, w, *_ in model.weight_entries("GENE_A")}
    w = np.array([w_map[s] for s in snps])
    gwas = pd.read_csv(synthetic["gwas"], sep="\t").set_index("SNP")
    # load_gwas casts z-scores to float32; match that for an exact check
    z = np.array([gwas.loc[s, "ZSCORE"] for s in snps], dtype=np.float32)
    z = z.astype(np.float64)
    sigma_l = np.sqrt(np.diag(m))
    var_g = w.dot(m).dot(w)
    z_expected = np.sum(w * z * sigma_l) / np.sqrt(var_g)
    assert a.zscore == pytest.approx(z_expected, rel=1e-9)
    assert a.var_g == pytest.approx(var_g, rel=1e-9)


def test_spredixcan_pvalue_is_two_sided_normal(synthetic):
    """The reported p-value is 2 * Phi(-|z|)."""
    import scipy.stats as stats

    res = pytwas.spredixcan(
        model_db_path=synthetic["db"], covariance=synthetic["cov"],
        gwas_file=synthetic["gwas"], snp_column="SNP",
        effect_allele_column="A1", non_effect_allele_column="A2",
        zscore_column="ZSCORE",
    )
    for _, row in res.iterrows():
        if np.isfinite(row.zscore):
            assert row.pvalue == pytest.approx(
                2 * stats.norm.sf(abs(row.zscore)), rel=1e-9
            )


def test_spredixcan_additional_output(synthetic):
    """additional_output adds best_gwas_p / largest_weight columns."""
    res = pytwas.spredixcan(
        model_db_path=synthetic["db"], covariance=synthetic["cov"],
        gwas_file=synthetic["gwas"], snp_column="SNP",
        effect_allele_column="A1", non_effect_allele_column="A2",
        zscore_column="ZSCORE", additional_output=True,
    )
    assert "best_gwas_p" in res.columns
    assert "largest_weight" in res.columns
    a = res.set_index("gene").loc["GENE_A"]
    assert a.largest_weight == pytest.approx(0.5)  # max |weight| of GENE_A


# ----------------------------------------------------------------------
# S-MultiXcan
# ----------------------------------------------------------------------
def test_capinv_truncates_singular_values():
    """capinv with a tight cutoff drops small singular values."""
    a = np.diag([4.0, 1.0, 1e-9])
    inv, n_indep, eigen = pytwas.capinv(a, rcond=1e-3)
    assert n_indep == 2  # the 1e-9 component is dropped
    # surviving block is the ordinary inverse
    np.testing.assert_allclose(np.diag(inv)[:2], [0.25, 1.0], rtol=1e-9)


def test_smultixcan_runs(synthetic):
    """S-MultiXcan combines two synthetic tissues into a chi-square test."""
    # two tissues = the same model used twice with different GWAS
    res1 = pytwas.spredixcan(
        model_db_path=synthetic["db"], covariance=synthetic["cov"],
        gwas_file=synthetic["gwas"], snp_column="SNP",
        effect_allele_column="A1", non_effect_allele_column="A2",
        zscore_column="ZSCORE",
    )
    res2 = res1.copy()
    res2["zscore"] = res2["zscore"] * 0.5
    model = pytwas.load_model(synthetic["db"])
    out = pytwas.smultixcan(
        {"T1": res1, "T2": res2},
        {"T1": model, "T2": model},
        synthetic["cov"],
        cutoff_condition_number=30,
    )
    assert set(out.gene) == {"GENE_A", "GENE_B"}
    assert (out.pvalue.dropna() >= 0).all()
    assert (out.pvalue.dropna() <= 1).all()
    assert (out.n == 2).all()


# ----------------------------------------------------------------------
# PrediXcan
# ----------------------------------------------------------------------
def test_predict_expression(synthetic):
    """Predicted expression is the weighted dosage sum."""
    model = pytwas.load_model(synthetic["db"])
    dosages = pd.DataFrame(
        {"rs1": [0.0, 2.0], "rs2": [1.0, 1.0], "rs3": [2.0, 0.0]},
        index=["S0", "S1"],
    )
    expr = pytwas.predict_expression(model, dosages, genes=["GENE_A"])
    # S0: 0.5*0 + (-0.3)*1 + 0.2*2 = 0.1
    assert expr.loc["S0", "GENE_A"] == pytest.approx(0.1)
    # S1: 0.5*2 + (-0.3)*1 + 0.2*0 = 0.7
    assert expr.loc["S1", "GENE_A"] == pytest.approx(0.7)


def test_predixcan_linear(synthetic):
    """PrediXcan linear association recovers a planted effect."""
    model = pytwas.load_model(synthetic["db"])
    rng = np.random.default_rng(1)
    n = 300
    dosages = pd.DataFrame(
        {s: rng.integers(0, 3, n).astype(float) for s in model.snps()},
        index=[f"S{i}" for i in range(n)],
    )
    expr = pytwas.predict_expression(model, dosages, genes=["GENE_A"])
    pheno = 2.0 * expr["GENE_A"].values + rng.normal(0, 1, n)
    res = pytwas.predixcan(model, dosages, pheno, mode="linear", genes=["GENE_A"])
    row = res.iloc[0]
    assert row.gene == "GENE_A"
    assert row.n_samples == n
    assert row.effect == pytest.approx(2.0, abs=0.3)
    assert row.pvalue < 1e-6


def test_predixcan_logistic_runs(synthetic):
    """PrediXcan logistic association produces a finite z-score."""
    model = pytwas.load_model(synthetic["db"])
    rng = np.random.default_rng(2)
    n = 300
    dosages = pd.DataFrame(
        {s: rng.integers(0, 3, n).astype(float) for s in model.snps()},
        index=[f"S{i}" for i in range(n)],
    )
    expr = pytwas.predict_expression(model, dosages, genes=["GENE_A"])
    prob = 1 / (1 + np.exp(-(1.5 * expr["GENE_A"].values)))
    pheno = (rng.random(n) < prob).astype(float)
    res = pytwas.predixcan(model, dosages, pheno, mode="logistic", genes=["GENE_A"])
    row = res.iloc[0]
    assert np.isfinite(row.zscore)
    assert row.status is None or row.status == "MLE_did_not_converge"
