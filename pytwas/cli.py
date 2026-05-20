"""Command-line interface for py-twas.

Thin wrappers mirroring the original MetaXcan ``SPrediXcan.py`` and
``SMulTiXcan.py`` argument layout.  Invoked as::

    pytwas spredixcan --model_db_path MODEL.db --covariance COV.txt.gz \\
        --gwas_file GWAS.txt.gz --snp_column SNP --effect_allele_column A1 \\
        --non_effect_allele_column A2 --beta_column BETA --pvalue_column P \\
        --output_file results.csv

    pytwas smultixcan --models_folder MODELS/ --covariances_folder COVS/ \\
        --spredixcan_folder SPX/ --cutoff_condition_number 30 \\
        --output OUT.txt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List

import pandas as pd

from .spredixcan import spredixcan
from .smultixcan import smultixcan


def _add_gwas_args(p: argparse.ArgumentParser) -> None:
    """Attach the GWAS-format arguments common to S-PrediXcan."""
    p.add_argument("--snp_column", default="SNP")
    p.add_argument("--effect_allele_column", default="A1")
    p.add_argument("--non_effect_allele_column", default="A2")
    p.add_argument("--chromosome_column", default=None)
    p.add_argument("--position_column", default=None)
    p.add_argument("--beta_column", default=None)
    p.add_argument("--beta_sign_column", default=None)
    p.add_argument("--or_column", default=None)
    p.add_argument("--se_column", default=None)
    p.add_argument("--zscore_column", default=None)
    p.add_argument("--pvalue_column", default=None)
    p.add_argument("--separator", default=None)
    p.add_argument("--keep_non_rsid", action="store_true", default=False)
    p.add_argument("--input_pvalue_fix", type=float, default=1e-50)


def _spredixcan_parser(sub: argparse._SubParsersAction) -> None:
    """Define the ``spredixcan`` sub-command."""
    p = sub.add_parser("spredixcan", help="Summary-statistics TWAS (S-PrediXcan)")
    p.add_argument("--model_db_path", required=True)
    p.add_argument("--model_db_snp_key", default=None)
    p.add_argument("--covariance", required=True)
    p.add_argument("--gwas_file", required=True)
    _add_gwas_args(p)
    p.add_argument("--additional_output", action="store_true", default=False)
    p.add_argument("--remove_ens_version", action="store_true", default=False)
    p.add_argument("--output_file", required=True)


def _smultixcan_parser(sub: argparse._SubParsersAction) -> None:
    """Define the ``smultixcan`` sub-command."""
    p = sub.add_parser("smultixcan", help="Multi-tissue TWAS (S-MultiXcan)")
    p.add_argument("--models_folder", required=True,
                   help="Folder of prediction-model .db files")
    p.add_argument("--snp_covariance", required=True,
                   help="Single merged SNP-covariance file (covering every "
                        "SNP of every tissue's model)")
    p.add_argument("--spredixcan_folder", required=True,
                   help="Folder of per-tissue S-PrediXcan result CSVs "
                        "(file stem = tissue name)")
    p.add_argument("--cutoff_condition_number", type=float, default=None)
    p.add_argument("--cutoff_eigen_ratio", type=float, default=None)
    p.add_argument("--cutoff_threshold", type=float, default=None)
    p.add_argument("--cutoff_trace_ratio", type=float, default=None)
    p.add_argument("--regularization", type=float, default=None)
    p.add_argument("--trimmed_ensemble_id", action="store_true", default=False)
    p.add_argument("--output", required=True)


def _run_spredixcan(args: argparse.Namespace) -> None:
    """Execute the ``spredixcan`` sub-command."""
    df = spredixcan(
        model_db_path=args.model_db_path,
        covariance=args.covariance,
        gwas_file=args.gwas_file,
        snp_column=args.snp_column,
        effect_allele_column=args.effect_allele_column,
        non_effect_allele_column=args.non_effect_allele_column,
        zscore_column=args.zscore_column,
        beta_column=args.beta_column,
        se_column=args.se_column,
        or_column=args.or_column,
        beta_sign_column=args.beta_sign_column,
        pvalue_column=args.pvalue_column,
        chromosome_column=args.chromosome_column,
        position_column=args.position_column,
        separator=args.separator,
        keep_non_rsid=args.keep_non_rsid,
        input_pvalue_fix=args.input_pvalue_fix,
        model_db_snp_key=args.model_db_snp_key,
        additional_output=args.additional_output,
        remove_ens_version=args.remove_ens_version,
        output_file=args.output_file,
    )
    print(f"S-PrediXcan: {df.shape[0]} genes -> {args.output_file}")


def _match_by_stem(folder: str, exts: List[str]) -> dict:
    """Return ``stem -> path`` for files in ``folder`` with given extensions."""
    out = {}
    for ext in exts:
        for path in glob.glob(os.path.join(folder, f"*{ext}")):
            stem = os.path.basename(path)
            for e in exts:
                if stem.endswith(e):
                    stem = stem[: -len(e)]
                    break
            out[stem] = path
    return out


def _run_smultixcan(args: argparse.Namespace) -> None:
    """Execute the ``smultixcan`` sub-command."""
    models = _match_by_stem(args.models_folder, [".db"])
    spx = _match_by_stem(args.spredixcan_folder, [".csv"])

    common = set(models) & set(spx)
    if not common:
        sys.exit("No tissues with matching model and S-PrediXcan files")

    spredixcan_results = {t: pd.read_csv(spx[t]) for t in common}
    model_paths = {t: models[t] for t in common}

    df = smultixcan(
        spredixcan_results,
        model_paths,
        args.snp_covariance,
        cutoff_condition_number=args.cutoff_condition_number,
        cutoff_eigen_ratio=args.cutoff_eigen_ratio,
        cutoff_threshold=args.cutoff_threshold,
        cutoff_trace_ratio=args.cutoff_trace_ratio,
        regularization=args.regularization,
        trimmed_ensemble_id=args.trimmed_ensemble_id,
        output_file=args.output,
    )
    print(f"S-MultiXcan: {df.shape[0]} genes -> {args.output}")


def main(argv: List[str] | None = None) -> None:
    """py-twas command-line entry point."""
    parser = argparse.ArgumentParser(
        prog="pytwas",
        description="py-twas: transcriptome-wide association study "
        "(faithful S-PrediXcan / S-MultiXcan reimplementation)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    _spredixcan_parser(sub)
    _smultixcan_parser(sub)

    args = parser.parse_args(argv)
    if args.command == "spredixcan":
        _run_spredixcan(args)
    elif args.command == "smultixcan":
        _run_smultixcan(args)


if __name__ == "__main__":  # pragma: no cover
    main()
