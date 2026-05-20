"""Head-to-head accuracy + speed benchmark: original MetaXcan vs pytwas.

Runs S-PrediXcan on the bundled MetaXcan sample data (a small prediction
model ``.db``, a SNP-covariance file and a GWAS), with both:

* the **original MetaXcan** software (``M03_betas`` + ``M04_zscores``), and
* **pytwas** (:func:`pytwas.spredixcan`),

then reports, per gene, the agreement of z-score / effect-size / p-value
and the wall-clock time of each implementation.

Usage::

    python examples/benchmark.py [--runs N] [--metaxcan-ref DIR]

The MetaXcan reference clone is expected at ``/tmp/metaxcan_ref`` (or set
``--metaxcan-ref`` / ``$METAXCAN_REF``).  If it is unavailable the script
still benchmarks pytwas against the committed gold CSV.
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "tests", "data")
sys.path.insert(0, ROOT)

import pytwas  # noqa: E402


def _run_metaxcan(sw_dir, model_db, cov, gwas):
    """Run the original MetaXcan M03+M04 pipeline; return (frame, seconds)."""
    sys.path.insert(0, sw_dir)
    import M03_betas  # noqa: E402
    import M04_zscores  # noqa: E402

    logging.disable(logging.CRITICAL)

    class _A:
        pass

    a = _A()
    for k, v in dict(
        model_db_path=model_db, model_db_snp_key=None, covariance=cov,
        gwas_file=gwas, gwas_folder=None, gwas_file_pattern=None,
        snp_column="SNP", effect_allele_column="A1", non_effect_allele_column="A2",
        chromosome_column=None, position_column=None, freq_column=None,
        beta_column="BETA", beta_sign_column=None, or_column=None, se_column=None,
        zscore_column="ZSCORE", pvalue_column=None, separator=None,
        skip_until_header=None, handle_empty_columns=False, input_pvalue_fix=1e-50,
        keep_non_rsid=False, snp_map_file=None, split_column=None,
        input_gwas_format_json=None, output_folder=None, output=None,
        output_file=None, single_snp_model=False, stream_covariance=False,
        remove_ens_version=False, overwrite=True, additional_output=True,
        MAX_R=None, throw=True, verbosity=50, gwas_h2=None, gwas_N=None,
    ).items():
        setattr(a, k, v)

    t0 = time.perf_counter()
    g = M03_betas.run(copy.copy(a))
    res = M04_zscores.run(a, g)
    return res, time.perf_counter() - t0


def _run_pytwas(model_db, cov, gwas):
    """Run pytwas S-PrediXcan; return (frame, seconds)."""
    t0 = time.perf_counter()
    res = pytwas.spredixcan(
        model_db_path=model_db, covariance=cov, gwas_file=gwas,
        snp_column="SNP", effect_allele_column="A1", non_effect_allele_column="A2",
        zscore_column="ZSCORE", beta_column="BETA", additional_output=True,
    )
    return res, time.perf_counter() - t0


def main() -> None:
    """Run the benchmark and print a summary table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, help="timing repetitions")
    parser.add_argument(
        "--metaxcan-ref",
        default=os.environ.get("METAXCAN_REF", "/tmp/metaxcan_ref"),
    )
    args = parser.parse_args()

    model_db = os.path.join(DATA, "sample_model.db")
    cov = os.path.join(DATA, "sample_cov.txt.gz")
    gwas = os.path.join(DATA, "sample_gwas.txt.gz")
    sw_dir = os.path.join(args.metaxcan_ref, "software")
    has_ref = os.path.isfile(os.path.join(sw_dir, "M04_zscores.py"))

    print("=" * 64)
    print("py-twas S-PrediXcan benchmark")
    print("=" * 64)

    # pytwas timing
    py_times = []
    for _ in range(args.runs):
        mine, dt = _run_pytwas(model_db, cov, gwas)
        py_times.append(dt)
    print(f"pytwas       : {np.mean(py_times)*1e3:8.2f} ms  "
          f"({mine.shape[0]} genes)")

    if has_ref:
        ref_times = []
        for _ in range(args.runs):
            gold, dt = _run_metaxcan(sw_dir, model_db, cov, gwas)
            ref_times.append(dt)
        print(f"MetaXcan     : {np.mean(ref_times)*1e3:8.2f} ms  "
              f"({gold.shape[0]} genes)")
        speedup = np.mean(ref_times) / np.mean(py_times)
        print(f"speed-up     : {speedup:8.2f}x")
    else:
        print(f"MetaXcan     : (reference not found at {sw_dir}; "
              "comparing to committed gold CSV)")
        gp = os.path.join(DATA, "spx_gold.csv")
        gold = pd.read_csv(gp) if os.path.isfile(gp) else None

    # accuracy
    if gold is not None:
        print("-" * 64)
        m = mine.set_index("gene")
        g = gold.set_index("gene")
        common = sorted(set(m.index) & set(g.index))
        max_abs = 0.0
        print(f"{'gene':>8} {'z(pytwas)':>12} {'z(MetaXcan)':>13} {'|diff|':>10}")
        for gene in common:
            zm = float(m.loc[gene, "zscore"]) if not pd.isna(m.loc[gene, "zscore"]) \
                else np.nan
            zg = pd.to_numeric(g.loc[gene, "zscore"], errors="coerce")
            d = abs(zm - zg) if (np.isfinite(zm) and np.isfinite(zg)) else 0.0
            max_abs = max(max_abs, d)
            print(f"{gene:>8} {zm:12.6f} {float(zg) if np.isfinite(zg) else np.nan:13.6f} {d:10.2e}")
        print("-" * 64)
        zmv = pd.to_numeric(m.loc[common, "zscore"], errors="coerce")
        zgv = pd.to_numeric(g.loc[common, "zscore"], errors="coerce")
        ok = np.isfinite(zmv) & np.isfinite(zgv)
        if ok.sum() > 1:
            r = np.corrcoef(zmv[ok], zgv[ok])[0, 1]
            print(f"z-score Pearson r : {r:.12f}")
        print(f"max |z diff|      : {max_abs:.2e}")
        print("PARITY: " + ("OK (machine precision)" if max_abs < 1e-9
                            else "CHECK"))


if __name__ == "__main__":
    main()
