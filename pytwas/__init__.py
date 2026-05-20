"""pytwas: a clean modern-Python reimplementation of TWAS (MetaXcan).

``py-twas`` is a standalone, dependency-light port of the transcriptome-wide
association study (TWAS) software **MetaXcan / PrediXcan** (Barbeira *et al.*,
*Nature Communications* 2018; ``hakyimlab/MetaXcan``).  It is built for
**numerical parity** with the original software while exposing a clean,
importable Python API and a thin CLI.

What is TWAS?
-------------
TWAS tests, gene by gene, whether *genetically predicted* expression is
associated with a trait.  An elastic-net prediction model maps SNPs to
expression; that model is combined with GWAS association statistics (and a
reference LD covariance) to obtain a gene-level association.

Three engines
-------------
* :func:`spredixcan` -- **S-PrediXcan**: summary-statistics TWAS.  The
  workhorse.  Closed-form combination of GWAS z-scores, prediction-model
  weights and SNP covariance into a gene-level z-score, p-value and
  effect size.
* :func:`smultixcan` -- **S-MultiXcan**: multi-tissue joint TWAS.
  Aggregates per-tissue S-PrediXcan z-scores through an SVD-regularised
  chi-square test on the tissue-tissue correlation matrix.
* :func:`predixcan` -- **PrediXcan**: individual-level TWAS.  Predicts
  expression from genotype dosages, then regresses it on the phenotype
  (linear or logistic).

Readers and helpers
-------------------
* :func:`load_model` / :class:`PredictionModel` -- PrediXcan/MetaXcan
  prediction-model ``.db`` reader.
* :func:`load_covariance` / :class:`CovarianceDB` -- SNP-covariance
  ``.txt[.gz]`` reader.
* :func:`load_gwas` -- GWAS summary-statistics parser with full
  harmonisation (zscore / pvalue+beta / pvalue+or / se+beta paths).
* :func:`align_to_model` -- allele-flip harmonisation of a GWAS to a
  model.
* :func:`associate` / :func:`gene_association` -- the per-gene
  S-PrediXcan association primitives.
* :func:`predict_expression` -- genotype -> expression prediction.
* :func:`capinv` -- the truncated-SVD pseudo-inverse used by S-MultiXcan.

Quick-start
-----------
>>> import pytwas
>>> res = pytwas.spredixcan(
...     model_db_path="model.db",
...     covariance="cov.txt.gz",
...     gwas_file="gwas.txt.gz",
...     snp_column="SNP", effect_allele_column="A1",
...     non_effect_allele_column="A2",
...     beta_column="BETA", pvalue_column="P",
... )
>>> res[["gene", "gene_name", "zscore", "pvalue", "effect_size"]].head()
"""
from __future__ import annotations

from .association import GeneAssociation, associate, gene_association
from .gwas import align_to_model, beta_from_pvalue, load_gwas, zscore_from_pvalue
from .model import CovarianceDB, PredictionModel, load_covariance, load_model
from .predixcan import PredixcanResult, predict_expression, predixcan
from .smultixcan import (
    MultiXcanResult,
    capinv,
    smultixcan,
    tissue_correlation_matrix,
)
from .spredixcan import spredixcan

__version__ = "0.1.0"

__all__ = [
    # main TWAS engines
    "spredixcan",
    "smultixcan",
    "predixcan",
    # S-PrediXcan primitives
    "associate",
    "gene_association",
    "GeneAssociation",
    # S-MultiXcan primitives
    "capinv",
    "tissue_correlation_matrix",
    "MultiXcanResult",
    # PrediXcan primitives
    "predict_expression",
    "PredixcanResult",
    # model / covariance readers
    "load_model",
    "PredictionModel",
    "load_covariance",
    "CovarianceDB",
    # GWAS parsing / harmonisation
    "load_gwas",
    "align_to_model",
    "zscore_from_pvalue",
    "beta_from_pvalue",
]
