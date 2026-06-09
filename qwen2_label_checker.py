"""
Qwen2-Audio Label Noise Checker
================================
For each audio in the D2/D3 datasets, asks Qwen2-Audio-7B-Instruct to classify
the sound from the domain-specific class list.  Results are saved to a CSV so
progress survives restarts, and mismatches (potential mislabelings) are printed
at the end.

Usage
-----
python qwen2_label_checker.py \
    --split both          # train | test | both
    --domain both         # D2 | D3 | both
    --data_root task7_data
    --results_csv qwen2_label_check_results.csv
    --batch_size 1        # Qwen2-Audio does not support true batching yet
    --max_new_tokens 20
    --start_idx 0         # resume from this row index if restarting
"""

import argparse
import os
import csv
import re
import librosa
import torch
import pandas as pd
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# ── Domain class lists ────────────────────────────────────────────────────────

D2_CLASSES = [
    "alarm", "dog_bark", "engine", "fire",
    "footsteps", "knocking", "piano", "speech",
]

D3_CLASSES = [
    "alarm", "baby_cry", "dog_bark", "engine",
    "fire", "footsteps", "piano", "speech",
    "telephone_ringing",
]

DOMAIN_CLASSES = {"D2": D2_CLASSES, "D3": D3_CLASSES}

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(domain: str) -> str:
    classes = DOMAIN_CLASSES[domain]
    class_list_str = ", ".join(f'"{c}"' for c in classes)
    return (
        f"Listen carefully to the audio clip. "
        f"Identify the primary sound event. "
        f"You MUST choose exactly one label from this list: [{class_list_str}]. "
        f"Reply with ONLY the label word, nothing else."
    )


def extract_label(response: str, domain: str) -> str:
    """Parse the model's reply and map it to the closest known class."""
    classes = DOMAIN_CLASSES[domain]
    resp_lower = response.strip().lower()

    # Exact match first
    if resp_lower in classes:
        return resp_lower

    # Check if any class name appears in the response
    for cls in classes:
        if cls in resp_lower:
            return cls

    # Fuzzy fallbacks for common paraphrases
    paraphrase_map = {
        "bark": "dog_bark", "dog": "dog_bark", "barking": "dog_bark",
        "cry": "baby_cry", "crying": "baby_cry", "infant": "baby_cry", "baby": "baby_cry",
        "step": "footsteps", "walking": "footsteps", "footstep": "footsteps",
        "knock": "knocking", "door knock": "knocking",
        "ring": "telephone_ringing", "phone": "telephone_ringing", "ringing": "telephone_ringing",
        "music": "piano", "instrument": "piano", "melody": "piano",
        "talk": "speech", "speaking": "speech", "voice": "speech", "human": "speech",
        "fire": "fire", "burning": "fire", "flame": "fire",
        "alarm": "alarm", "siren": "alarm", "beep": "alarm",
        "engine": "engine", "motor": "engine", "machine": "engine",
    }
    for key, mapped in paraphrase_map.items():
        if key in resp_lower and mapped in classes:
            return mapped

    return "unknown"


# ── Main ──────────────────────────────────────────────────────────────────────

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

    df = df.reset_index(drop=True)
    return df


def run_inference(model, processor, audio_path: str, domain: str, max_new_tokens: int, device: str) -> str:
    sr = processor.feature_extractor.sampling_rate
    try:
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        return f"LOAD_ERROR: {e}"

    prompt_text = build_prompt(domain)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=text,
        audios=[audio],
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Strip the input tokens
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Qwen2-Audio label noise checker")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--domain", choices=["D2", "D3", "both"], default="both")
    parser.add_argument("--data_root", default="task7_data")
    parser.add_argument("--results_csv", default="qwen2_label_check_results.csv")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Row index to resume from (skip already-processed rows).")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Process up to this row index (exclusive). Useful for partial runs.")
    parser.add_argument("--model_name", default="Qwen/Qwen2-Audio-7B-Instruct")
    args = parser.parse_args()

    # ── Resolve paths relative to this script's directory ────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = args.data_root if os.path.isabs(args.data_root) else os.path.join(script_dir, args.data_root)
    results_csv = args.results_csv if os.path.isabs(args.results_csv) else os.path.join(script_dir, args.results_csv)

    # ── Load metadata ─────────────────────────────────────────────────────────
    print(f"Loading metadata  split={args.split}  domain={args.domain}")
    df = load_metadata(data_root, args.split, args.domain)
    total = len(df)
    print(f"Total samples: {total}")

    end_idx = args.end_idx if args.end_idx is not None else total

    # ── Load already-processed rows so we can skip them ──────────────────────
    already_done: set[str] = set()
    if os.path.exists(results_csv):
        prev = pd.read_csv(results_csv)
        already_done = set(prev["filename"].tolist())
        print(f"Resuming — {len(already_done)} files already in {results_csv}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"Model on device: {device}")

    # ── CSV output setup ──────────────────────────────────────────────────────
    csv_exists = os.path.exists(results_csv)
    csv_file = open(results_csv, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not csv_exists:
        writer.writerow(["filename", "ground_truth", "domain", "model_response", "predicted_label", "mismatch"])

    # ── Inference loop ────────────────────────────────────────────────────────
    mismatches = []

    for idx in range(args.start_idx, end_idx):
        row = df.iloc[idx]
        filename = row["filename"]
        gt_label = row["label"]
        domain = row["domain"]

        if filename in already_done:
            continue

        audio_path = os.path.join(data_root, filename)
        if not os.path.exists(audio_path):
            print(f"[{idx+1}/{total}] MISSING: {audio_path}")
            continue

        print(f"[{idx+1}/{total}] {filename}  gt={gt_label}", end=" ... ", flush=True)

        response = run_inference(model, processor, audio_path, domain, args.max_new_tokens, str(device))
        predicted = extract_label(response, domain)
        mismatch = (predicted != gt_label) and not response.startswith("LOAD_ERROR")

        print(f"pred={predicted}  raw='{response}'  {'*** MISMATCH ***' if mismatch else 'OK'}")

        writer.writerow([filename, gt_label, domain, response, predicted, mismatch])
        csv_file.flush()

        if mismatch:
            mismatches.append({
                "filename": filename,
                "ground_truth": gt_label,
                "domain": domain,
                "predicted": predicted,
                "model_response": response,
            })

    csv_file.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("POTENTIAL MISLABELED SAMPLES (Qwen2 disagrees with ground truth)")
    print("=" * 70)

    # Also include mismatches from previous runs
    if os.path.exists(results_csv):
        all_results = pd.read_csv(results_csv)
        all_mismatches = all_results[all_results["mismatch"] == True]
        print(f"\nTotal mismatches across all processed files: {len(all_mismatches)} / {len(all_results)}\n")
        for _, r in all_mismatches.iterrows():
            print(f"  File: {r['filename']}")
            print(f"    Domain:       {r['domain']}")
            print(f"    Ground truth: {r['ground_truth']}")
            print(f"    Qwen2 pred:   {r['predicted_label']}")
            print(f"    Raw response: {r['model_response']}")
            print()

    # Mismatch counts by class
    if os.path.exists(results_csv):
        all_results = pd.read_csv(results_csv)
        mm = all_results[all_results["mismatch"] == True]
        if len(mm):
            print("\n--- Mismatch count by ground-truth class ---")
            print(mm.groupby(["domain", "ground_truth"])["mismatch"].count().to_string())

            print("\n--- What Qwen2 predicted instead ---")
            print(mm.groupby(["ground_truth", "predicted_label"])["mismatch"].count().to_string())

    print("\nDone. Full results saved to:", results_csv)


if __name__ == "__main__":
    main()
