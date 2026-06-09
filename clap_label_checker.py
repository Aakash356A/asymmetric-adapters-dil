"""
CLAP Zero-Shot Label Noise Checker
====================================
Uses laion/clap-htsat-unfused to classify every audio in the D2/D3 datasets
against the domain-specific class list.  Mismatches (where CLAP disagrees with
the ground-truth label) are printed as potential mislabelings.

Results are saved incrementally to a CSV so the run can be interrupted and
resumed safely.

Usage
-----
python clap_label_checker.py                          # all splits, all domains
python clap_label_checker.py --split test --domain D3
python clap_label_checker.py --end_idx 200            # first 200 files only
"""

import argparse
import csv
import os
import pandas as pd

# ── Domain class lists (raw dataset labels → human-readable for CLAP) ────────

# CLAP works best with natural-language descriptions, so we map each class to
# a short phrase that the model can compare against the audio embedding.

D2_LABEL_MAP = {
    "alarm":      "alarm sound or siren beeping",
    "dog_bark":   "dog barking",
    "engine":     "engine or motor running",
    "fire":       "fire crackling or burning",
    "footsteps":  "footsteps or walking sounds",
    "knocking":   "knocking on a door",
    "piano":      "piano music",
    "speech":     "human speech or talking",
}

D3_LABEL_MAP = {
    "alarm":              "alarm sound or siren beeping",
    "baby_cry":           "baby crying",
    "dog_bark":           "dog barking",
    "engine":             "engine or motor running",
    "fire":               "fire crackling or burning",
    "footsteps":          "footsteps or walking sounds",
    "piano":              "piano music",
    "speech":             "human speech or talking",
    "telephone_ringing":  "telephone ringing",
}

DOMAIN_LABEL_MAP = {"D2": D2_LABEL_MAP, "D3": D3_LABEL_MAP}


def phrase_to_class(phrase: str, label_map: dict) -> str:
    """Reverse-map a CLAP phrase back to the original dataset class name."""
    for cls, p in label_map.items():
        if p == phrase:
            return cls
    return "unknown"


# ── Metadata loader ───────────────────────────────────────────────────────────

def load_metadata(data_root: str, split: str, domain_filter: str) -> pd.DataFrame:
    split_files = []
    if split in ("train", "both"):
        split_files.append(os.path.join(data_root, "evaluation_setup", "development_train.txt"))
    if split in ("test", "both"):
        split_files.append(os.path.join(data_root, "evaluation_setup", "development_test.txt"))

    dfs = []
    for f in split_files:
        df = pd.read_csv(f, sep="\t", header=None,
                         names=["filename", "label", "domain", "class_idx"])
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    if domain_filter != "both":
        df = df[df["domain"] == domain_filter]

    return df.reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CLAP zero-shot label noise checker")
    parser.add_argument("--split",       choices=["train", "test", "both"], default="both")
    parser.add_argument("--domain",      choices=["D2", "D3", "both"],      default="both")
    parser.add_argument("--data_root",   default="task7_data")
    parser.add_argument("--results_csv", default="clap_label_check_results.csv")
    parser.add_argument("--model",       default="laion/clap-htsat-unfused")
    parser.add_argument("--device",      default="cuda:0")
    parser.add_argument("--start_idx",   type=int, default=0,
                        help="Row index to start from (0-based).")
    parser.add_argument("--end_idx",     type=int, default=None,
                        help="Row index to stop at (exclusive). Omit to process all.")
    args = parser.parse_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root  = args.data_root  if os.path.isabs(args.data_root)  else os.path.join(script_dir, args.data_root)
    results_csv = args.results_csv if os.path.isabs(args.results_csv) else os.path.join(script_dir, args.results_csv)

    # ── Load metadata ─────────────────────────────────────────────────────────
    print(f"Loading metadata  split={args.split}  domain={args.domain}")
    df = load_metadata(data_root, args.split, args.domain)
    total   = len(df)
    end_idx = args.end_idx if args.end_idx is not None else total
    print(f"Total samples: {total}  |  Processing rows {args.start_idx}–{end_idx - 1}")

    # ── Resume: skip already-processed files ─────────────────────────────────
    already_done: set = set()
    if os.path.exists(results_csv):
        prev = pd.read_csv(results_csv)
        already_done = set(prev["filename"].tolist())
        print(f"Resuming — {len(already_done)} files already saved in {results_csv}")

    # ── Load CLAP pipeline ────────────────────────────────────────────────────
    from transformers import pipeline as hf_pipeline

    print(f"\nLoading CLAP model: {args.model}  (device={args.device})")
    classifier = hf_pipeline(
        "zero-shot-audio-classification",
        model=args.model,
        device=args.device,
    )
    print("Model ready.\n")

    # ── CSV output ────────────────────────────────────────────────────────────
    csv_exists = os.path.exists(results_csv)
    csv_fh = open(results_csv, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_fh)
    if not csv_exists:
        writer.writerow([
            "filename", "ground_truth", "domain",
            "predicted_label", "top_phrase", "top_score",
            "gt_score", "gt_rank", "mismatch",
        ])

    # ── Inference loop ────────────────────────────────────────────────────────
    new_mismatches = []

    for idx in range(args.start_idx, end_idx):
        row      = df.iloc[idx]
        filename = row["filename"]
        gt_label = row["label"]
        domain   = row["domain"]

        if filename in already_done:
            continue

        audio_path = os.path.join(data_root, filename)
        if not os.path.exists(audio_path):
            print(f"[{idx+1}/{total}] MISSING: {audio_path}")
            continue

        label_map         = DOMAIN_LABEL_MAP[domain]
        candidate_phrases = list(label_map.values())

        try:
            results = classifier(audio_path, candidate_labels=candidate_phrases)
        except Exception as e:
            print(f"[{idx+1}/{total}] ERROR on {filename}: {e}")
            continue

        # results is sorted by score descending
        top_phrase = results[0]["label"]
        top_score  = results[0]["score"]
        predicted  = phrase_to_class(top_phrase, label_map)

        # Find rank and score of the ground-truth class
        gt_phrase = label_map.get(gt_label, "")
        gt_entry  = next((r for r in results if r["label"] == gt_phrase), None)
        gt_score  = gt_entry["score"] if gt_entry else 0.0
        gt_rank   = next((i + 1 for i, r in enumerate(results) if r["label"] == gt_phrase), -1)

        mismatch = predicted != gt_label

        status = "*** MISMATCH ***" if mismatch else "OK"
        print(
            f"[{idx+1}/{total}] {filename}  "
            f"gt={gt_label}  pred={predicted}  "
            f"score={top_score:.3f}  gt_rank={gt_rank}  {status}"
        )

        writer.writerow([
            filename, gt_label, domain,
            predicted, top_phrase, f"{top_score:.4f}",
            f"{gt_score:.4f}", gt_rank, mismatch,
        ])
        csv_fh.flush()

        if mismatch:
            new_mismatches.append({
                "filename": filename,
                "ground_truth": gt_label,
                "domain": domain,
                "predicted": predicted,
                "top_phrase": top_phrase,
                "top_score": top_score,
                "gt_score": gt_score,
                "gt_rank": gt_rank,
            })

    csv_fh.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("POTENTIAL MISLABELED SAMPLES  (CLAP disagrees with ground truth)")
    print("=" * 70)

    if os.path.exists(results_csv):
        all_results  = pd.read_csv(results_csv)
        all_mm       = all_results[all_results["mismatch"] == True]
        total_done   = len(all_results)
        total_mm     = len(all_mm)
        accuracy     = (total_done - total_mm) / total_done * 100 if total_done else 0

        print(f"\nProcessed : {total_done} files")
        print(f"Mismatches: {total_mm}  ({100 - accuracy:.1f}%)")
        print(f"Agreement : {accuracy:.1f}%\n")

        if total_mm:
            print(f"{'File':<45} {'Domain':<6} {'GT':<20} {'CLAP pred':<20} {'Top score':>9} {'GT rank':>7}")
            print("-" * 110)
            for _, r in all_mm.sort_values(["domain", "ground_truth"]).iterrows():
                print(
                    f"{r['filename']:<45} {r['domain']:<6} {r['ground_truth']:<20} "
                    f"{r['predicted_label']:<20} {float(r['top_score']):>9.3f} {int(r['gt_rank']):>7}"
                )

            print("\n--- Mismatch count by ground-truth class ---")
            print(all_mm.groupby(["domain", "ground_truth"])["mismatch"].count().to_string())

            print("\n--- What CLAP predicted instead ---")
            print(all_mm.groupby(["ground_truth", "predicted_label"])["mismatch"].count().to_string())

    print("\nFull results saved to:", results_csv)


if __name__ == "__main__":
    main()
