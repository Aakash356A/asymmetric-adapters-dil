"""
CLAP Mismatch Confidence Analysis
===================================
Filters the raw CLAP results to surface only high-confidence mismatches —
these are much more likely to be genuine label errors than borderline cases.

Also shows score distributions so you can pick the right threshold.

Usage
-----
python clap_analyze_mismatches.py
python clap_analyze_mismatches.py --top_score_thresh 0.5 --gt_score_thresh 0.15
python clap_analyze_mismatches.py --export_suspicious suspicious_files.csv
"""

import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv",       default="clap_label_check_results.csv")
    parser.add_argument("--top_score_thresh",  type=float, default=0.40,
                        help="Min CLAP top-prediction score to consider confident.")
    parser.add_argument("--gt_score_thresh",   type=float, default=0.20,
                        help="Max ground-truth class score to consider clearly wrong.")
    parser.add_argument("--export_suspicious", default=None,
                        help="If set, export high-confidence mismatches to this CSV.")
    args = parser.parse_args()

    df = pd.read_csv(args.results_csv)
    df["top_score"] = df["top_score"].astype(float)
    df["gt_score"]  = df["gt_score"].astype(float)
    df["gt_rank"]   = df["gt_rank"].astype(int)

    total = len(df)
    all_mm = df[df["mismatch"] == True].copy()

    print(f"\n{'='*70}")
    print(f"CLAP Results Summary")
    print(f"{'='*70}")
    print(f"Total files processed : {total}")
    print(f"Total mismatches      : {len(all_mm)}  ({len(all_mm)/total*100:.1f}%)")
    print(f"Agreement rate        : {(total - len(all_mm))/total*100:.1f}%")

    # ── Score distributions ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Score distribution for CORRECT predictions (top_score):")
    correct = df[df["mismatch"] == False]
    print(correct["top_score"].describe().round(3).to_string())

    print(f"\nScore distribution for MISMATCH predictions (top_score):")
    print(all_mm["top_score"].describe().round(3).to_string())

    print(f"\nGround-truth score distribution in MISMATCHES:")
    print(all_mm["gt_score"].describe().round(3).to_string())

    # ── Confidence-filtered mismatches ────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"HIGH-CONFIDENCE mismatches")
    print(f"  top_score >= {args.top_score_thresh}  AND  gt_score <= {args.gt_score_thresh}")
    print(f"  (These are the most credible label-error candidates)")
    print(f"{'='*70}")

    suspicious = all_mm[
        (all_mm["top_score"] >= args.top_score_thresh) &
        (all_mm["gt_score"]  <= args.gt_score_thresh)
    ].copy()
    suspicious = suspicious.sort_values("top_score", ascending=False)

    print(f"Count: {len(suspicious)} out of {len(all_mm)} total mismatches\n")

    if len(suspicious):
        print(f"{'File':<45} {'Dom':<4} {'GT':<22} {'Predicted':<22} {'Top score':>9} {'GT score':>8} {'GT rank':>7}")
        print("-" * 115)
        for _, r in suspicious.iterrows():
            print(
                f"{r['filename']:<45} {r['domain']:<4} {r['ground_truth']:<22} "
                f"{r['predicted_label']:<22} {r['top_score']:>9.3f} {r['gt_score']:>8.3f} {r['gt_rank']:>7}"
            )

    # ── Breakdown by class ────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("High-confidence mismatch count by ground-truth class:")
    if len(suspicious):
        print(suspicious.groupby(["domain", "ground_truth"])["mismatch"].count().to_string())

    print(f"\nWhat CLAP predicted for high-confidence mismatches:")
    if len(suspicious):
        print(suspicious.groupby(["ground_truth", "predicted_label"])["mismatch"].count().to_string())

    # ── Borderline cases (mid-confidence) ────────────────────────────────────
    borderline = all_mm[
        ~(
            (all_mm["top_score"] >= args.top_score_thresh) &
            (all_mm["gt_score"]  <= args.gt_score_thresh)
        )
    ]
    print(f"\n{'─'*70}")
    print(f"Borderline mismatches (lower confidence): {len(borderline)}")
    print(f"  These are likely CLAP classification errors, not real label noise.")

    # ── Per-threshold sweep ───────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Sensitivity to confidence threshold (top_score >= T, gt_score <= 0.20):")
    print(f"{'Threshold':>10}  {'Suspicious':>10}  {'% of mismatches':>16}")
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        n = len(all_mm[(all_mm["top_score"] >= t) & (all_mm["gt_score"] <= 0.20)])
        pct = n / len(all_mm) * 100 if len(all_mm) else 0
        print(f"{t:>10.2f}  {n:>10}  {pct:>15.1f}%")

    # ── Export ────────────────────────────────────────────────────────────────
    if args.export_suspicious and len(suspicious):
        suspicious.to_csv(args.export_suspicious, index=False)
        print(f"\nHigh-confidence mismatches exported to: {args.export_suspicious}")

    print()

if __name__ == "__main__":
    main()
