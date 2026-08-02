"""Command line: edgeaudit <export.csv> [-o report.html] [--subject "Account 12345"]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from . import audit, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="edgeaudit", description="Statistical trade audit.")
    ap.add_argument("csv", help="Raw broker export. No template, no reformatting.")
    ap.add_argument("-o", "--out", default=None, help="HTML report path")
    ap.add_argument("--markdown", action="store_true", help="Print markdown instead")
    ap.add_argument("--subject", default="Account", help="Name shown on the report")
    ap.add_argument("--min-n", type=int, default=30, help="Minimum trades to test a slice")
    ap.add_argument("--q", type=float, default=0.10, help="FDR level")
    ap.add_argument("--resamples", type=int, default=4000)
    ap.add_argument("--json", default=None, help="Also write machine-readable summary")
    a = ap.parse_args(argv)

    try:
        res = audit.run(a.csv, min_bucket_n=a.min_n, fdr_q=a.q, resamples=a.resamples)
    except Exception as exc:
        print(f"Could not audit this file: {exc}", file=sys.stderr)
        return 2

    print(f"{res.verdict}\n{res.verdict_detail}\n")
    print(f"Read as: {res.format_name} | {res.n_trades:,} trades | "
          f"{res.n_naive_significant} of {res.family_size} slices look significant, "
          f"{res.n_survived} survive correction.")

    if a.markdown:
        print()
        print(report.to_markdown(res))
    out = Path(a.out) if a.out else Path(a.csv).with_suffix(".audit.html")
    out.write_text(report.to_html(res, subject=a.subject), encoding="utf-8")
    print(f"\nReport written to {out}")
    if a.json:
        Path(a.json).write_text(json.dumps({
            "verdict": res.verdict, "n_trades": res.n_trades,
            "global": {k: (None if v != v else v) for k, v in res.global_stats.items()},
            "family_size": res.family_size, "naive_significant": res.n_naive_significant,
            "survived": res.n_survived,
            "verified_slices": res.buckets.loc[res.buckets.get("verified", False) == True,
                                               ["dimension", "label", "n", "mean_r",
                                                "shrunk_mean_r", "q_vs_self"]].to_dict("records")
            if len(res.buckets) else [],
        }, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
