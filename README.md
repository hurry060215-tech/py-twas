# py-twas

**py-twas** (`pytwas`) is a clean, modern-Python reimplementation of
**TWAS** — transcriptome-wide association study — faithful to the original
**MetaXcan / PrediXcan** software
([`hakyimlab/MetaXcan`](https://github.com/hakyimlab/MetaXcan); Barbeira
*et al.*, *Nature Communications* 2018).

It reproduces the three TWAS workhorses — **S-PrediXcan**, **S-MultiXcan**
and **PrediXcan** — with a clean importable API and a thin CLI, and is
**numerically identical** to the original MetaXcan (verified to machine
precision against a live MetaXcan run).

* Pure Python: only `numpy`, `scipy`, `pandas`, and stdlib `sqlite3`.
* No `rpy2`, no R, no compiled extensions.

---

## What is TWAS?

TWAS asks, gene by gene, whether *genetically predicted* gene expression
is associated with a trait.  An elastic-net **prediction model** maps SNPs
to expression; that model is combined with **GWAS** association statistics
to produce a gene-level association — pinpointing genes whose
cis-regulated expression mediates a GWAS signal.

---

## Installation

```bash
pip install pytwas
# or, from a checkout
pip install -e .
```

---

## The three engines

### 1. S-PrediXcan — summary-statistics TWAS (the workhorse)

Combines GWAS **summary statistics** (z-scores / betas) with the
prediction-model weights and the reference **SNP covariance** into a
gene-level association.  The closed-form S-PrediXcan formula
(Barbeira 2018) is, for a gene with weight vector `w`, GWAS z-scores `z`,
SNP standard deviations `sigma_l` and covariance `Sigma`:

```
sigma_g^2 = wᵀ Σ w
Z_g       = Σ_l ( w_l · z_l · sigma_l ) / sqrt(sigma_g^2)
```

```python
import pytwas

res = pytwas.spredixcan(
    model_db_path="model.db",          # PrediXcan/MetaXcan elastic-net model
    covariance="cov.txt.gz",           # reference SNP covariance
    gwas_file="gwas.txt.gz",
    snp_column="SNP",
    effect_allele_column="A1",
    non_effect_allele_column="A2",
    beta_column="BETA",
    pvalue_column="P",                 # z derived from beta + pvalue
)
res[["gene", "gene_name", "zscore", "pvalue", "effect_size",
     "n_snps_used", "n_snps_in_model", "pred_perf_r2"]].head()
```

Full GWAS harmonisation is supported: a `zscore` column directly, or a
`pvalue` column with `beta` / `or` / `beta_sign`, or an `se` column with
`beta` / `or`; odds-ratio → beta conversion; allele-flip alignment to the
model; the divergent-z-score `input_pvalue_fix`; `keep_non_rsid` and
`additional_output` flags.

### 2. S-MultiXcan — multi-tissue joint TWAS

Aggregates per-tissue S-PrediXcan z-scores into a single joint test
through an SVD-regularised chi-square statistic on the tissue-tissue
correlation matrix:

```python
res = pytwas.smultixcan(
    spredixcan_results={"Whole_Blood": spx_blood, "Liver": spx_liver, ...},
    models={"Whole_Blood": "blood.db", "Liver": "liver.db", ...},
    snp_covariance="snp_covariance.txt.gz",   # one merged covariance
    cutoff_condition_number=30,               # SVD truncation
)
```

The truncation strategy mirrors MetaXcan: `cutoff_condition_number`,
`cutoff_eigen_ratio`, `cutoff_threshold` or `cutoff_trace_ratio`.

### 3. PrediXcan — individual-level TWAS

Predicts expression from individual genotype dosages, then regresses it
against the phenotype (linear or logistic):

```python
res = pytwas.predixcan(
    model_db_path="model.db",
    dosages=dosage_df,        # samples × SNPs effect-allele dosages
    pheno=phenotype_vector,
    mode="linear",            # or "logistic"
)
```

---

## Command-line interface

`pytwas` ships a CLI mirroring `SPrediXcan.py` / `SMulTiXcan.py`:

```bash
pytwas spredixcan \
    --model_db_path model.db --covariance cov.txt.gz \
    --gwas_file gwas.txt.gz \
    --snp_column SNP --effect_allele_column A1 --non_effect_allele_column A2 \
    --beta_column BETA --pvalue_column P \
    --output_file results.csv

pytwas smultixcan \
    --models_folder models/ --covariances_folder covs/ \
    --spredixcan_folder spx/ \
    --cutoff_condition_number 30 --output joint.txt
```

---

## Input formats

| Input | Format |
|-------|--------|
| Prediction model | SQLite `.db` — `weights(rsid, gene, weight, ref_allele, eff_allele)` + `extra(gene, genename, n.snps.in.model, pred.perf.R2, pred.perf.pval, pred.perf.qval)` |
| SNP covariance | whitespace `.txt[.gz]` — `GENE RSID1 RSID2 VALUE` |
| GWAS | whitespace / tab table; columns configurable |

These are exactly the PrediXcan/MetaXcan/GTEx model and covariance
formats, so existing GTEx v7/v8 model databases work unchanged.

---

## Public API

```
spredixcan   smultixcan   predixcan          # the three TWAS engines
associate    gene_association  GeneAssociation
capinv       tissue_correlation_matrix  MultiXcanResult
predict_expression  PredixcanResult
load_model   PredictionModel               # model .db reader
load_covariance  CovarianceDB              # covariance reader
load_gwas    align_to_model                # GWAS parsing / harmonisation
zscore_from_pvalue  beta_from_pvalue
```

---

## Numerical parity

`tests/test_parity.py` drives the bundled MetaXcan sample data through
**both** `pytwas` and a live run of the original MetaXcan
(`M03_betas` + `M04_zscores` for S-PrediXcan; `cross_model` for
S-MultiXcan) and asserts cell-by-cell agreement:

* **S-PrediXcan**: per-gene `zscore`, `effect_size`, `pvalue`, `var_g`,
  `best_gwas_p`, `largest_weight`, `n_snps_*` — bit-identical
  (max abs diff ≈ 4e-16, z-score Pearson r = 1.0).
* **S-MultiXcan**: per-gene `pvalue`, eigen-spectrum, `n_indep`, `tmi` —
  agreement to ≈ 1e-15.
* **PrediXcan**: the OLS / logistic association matches `statsmodels`
  (the engine the original uses) to ≈ 1e-9.

If no MetaXcan checkout is present the tests fall back to committed gold
reference CSVs, so parity is always checked.

```bash
python -m pytest tests/ -q
python examples/benchmark.py        # head-to-head vs MetaXcan
```

See `examples/compare_reference.ipynb` for a worked comparison with a
z-score scatter and a TWAS Manhattan plot.

---

## License & credit

MIT, the same license as the original MetaXcan.  All credit for the TWAS
methodology and the reference implementation goes to the **Hae Kyung Im
lab** and collaborators — see Barbeira *et al.*, *Nat. Commun.* 9, 1825
(2018).  py-twas is an independent, faithful reimplementation built for
the [omicverse](https://github.com/Starlitnightly/omicverse) ecosystem.
