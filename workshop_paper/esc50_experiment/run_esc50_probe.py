"""
Out-of-challenge domain generalization / rejection probe.

Loads the submitted ResAdapt model (Divakar_IND_task7_1, D3 checkpoint,
which contains D1+D2+D3 components in one state dict) and evaluates it,
with no further training, on:

  (a) a 7-class subset of ESC-50 (a genuinely unseen domain: different
      microphones, environments, and source distribution than D1/D2/D3),
      restricted to the classes that overlap the Task 7 label space
      (alarm, baby_cry, bark, engine, fire, footsteps, knock), and

  (b) the real D2/D3 dev-test split, for a same-protocol in-domain
      reference.

For each sample we run the domain-agnostic ADIL forward pass (task=0,1,2),
compute per-task softmax entropy, and route to the minimum-entropy task.
We report classification accuracy (on the label subset present) and the
entropy distribution per condition (in-domain D2, in-domain D3, OOD ESC-50).
"""
import csv
import json
import os
import random
import sys

import librosa
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SUB_DIR = os.path.join(REPO_ROOT, "Divakar_IND_task7", "task7", "Divakar_IND_task7_1")
sys.path.insert(0, SUB_DIR)
from Divakar_IND_task7_1_model import build_model, ADAPTER_DEPTHS  # noqa: E402

SAMPLE_RATE = 32000
CLIP_SAMPLES = SAMPLE_RATE * 4

DICT_CLASS_LABELS = {
    'alarm': 0, 'baby_cry': 1, 'dog_bark': 2, 'engine': 3, 'fire': 4,
    'footsteps': 5, 'knocking': 6, 'telephone_ringing': 7, 'piano': 8, 'speech': 9,
}
IDX_TO_CLASS = {v: k for k, v in DICT_CLASS_LABELS.items()}

ESC50_TO_TASK7 = {
    'clock_alarm': 'alarm',
    'crying_baby': 'baby_cry',
    'dog': 'dog_bark',
    'engine': 'engine',
    'crackling_fire': 'fire',
    'footsteps': 'footsteps',
    'door_wood_knock': 'knocking',
}

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                       else 'mps' if torch.backends.mps.is_available()
                       else 'cpu')


def load_audio_fixed(path, rng=None):
    audio, _ = librosa.core.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < CLIP_SAMPLES:
        audio = np.concatenate([audio, np.zeros(CLIP_SAMPLES - len(audio))])
    elif len(audio) > CLIP_SAMPLES:
        # center crop (fixed 5s ESC-50 clips -> centered 4s window)
        start = (len(audio) - CLIP_SAMPLES) // 2
        audio = audio[start:start + CLIP_SAMPLES]
    return audio.astype(np.float32)


def load_model():
    model = build_model(nb_tasks=3, adapter_depths=ADAPTER_DEPTHS)
    ckpt = os.path.join(SUB_DIR, "Divakar_IND_task7_1_D3_dictionary.pth")
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    return model


@torch.no_grad()
def adil_predict(model, audio_batch):
    """audio_batch: (B, T) float32 tensor. Returns per-task probs, entropies,
    routed task index and routed class prediction for each sample."""
    probs_per_task = []
    for task in (0, 1, 2):
        logits = model(audio_batch, task=task)
        probs = F.softmax(logits, dim=1)
        probs_per_task.append(probs)
    probs_stack = torch.stack(probs_per_task, dim=0)  # (3, B, C)
    ent = -(probs_stack * torch.log(probs_stack.clamp_min(1e-12))).sum(-1)  # (3, B)
    routed_task = ent.argmin(dim=0)  # (B,)
    B = audio_batch.shape[0]
    routed_probs = probs_stack[routed_task, torch.arange(B)]
    routed_pred = routed_probs.argmax(dim=1)
    routed_entropy = ent[routed_task, torch.arange(B)]
    return {
        'routed_task': routed_task.cpu().numpy(),
        'routed_pred': routed_pred.cpu().numpy(),
        'routed_entropy': routed_entropy.cpu().numpy(),
        'entropy_per_task': ent.cpu().numpy(),  # (3, B)
    }


def batched(items, bs=16):
    for i in range(0, len(items), bs):
        yield items[i:i + bs]


def eval_esc50(model, esc50_root, n_per_class=None, seed=0):
    meta_path = os.path.join(esc50_root, "meta", "esc50.csv")
    audio_dir = os.path.join(esc50_root, "audio")
    rows_by_class = {c: [] for c in ESC50_TO_TASK7}
    with open(meta_path) as f:
        for row in csv.DictReader(f):
            cat = row['category']
            if cat in ESC50_TO_TASK7:
                rows_by_class[cat].append(row['filename'])

    rng = random.Random(seed)
    samples = []  # (path, task7_label)
    for cat, files in rows_by_class.items():
        files = sorted(files)
        if n_per_class is not None:
            rng.shuffle(files)
            files = files[:n_per_class]
        for fn in files:
            samples.append((os.path.join(audio_dir, fn), ESC50_TO_TASK7[cat]))

    print(f"[ESC-50] evaluating {len(samples)} clips across {len(rows_by_class)} classes")

    results = []
    for chunk in batched(samples, bs=16):
        audios = np.stack([load_audio_fixed(p) for p, _ in chunk])
        audio_t = torch.from_numpy(audios).to(DEVICE)
        out = adil_predict(model, audio_t)
        for i, (path, label) in enumerate(chunk):
            results.append({
                'file': os.path.basename(path),
                'true_label': label,
                'true_idx': DICT_CLASS_LABELS[label],
                'routed_task': int(out['routed_task'][i]),
                'pred_idx': int(out['routed_pred'][i]),
                'pred_label': IDX_TO_CLASS[int(out['routed_pred'][i])],
                'routed_entropy': float(out['routed_entropy'][i]),
                'entropy_task0': float(out['entropy_per_task'][0, i]),
                'entropy_task1': float(out['entropy_per_task'][1, i]),
                'entropy_task2': float(out['entropy_per_task'][2, i]),
            })
    return results


def eval_in_domain(model, task7_data_root, domain, n_per_class=None, seed=0):
    """domain: 'D2' or 'D3'. Uses the real dev-test split."""
    eval_file = os.path.join(task7_data_root, "evaluation_setup", "development_test.txt")
    rows_by_class = {}
    with open(eval_file) as f:
        for line in f:
            rel_path, label, dom, _ = line.strip().split('\t')
            if dom != domain:
                continue
            rows_by_class.setdefault(label, []).append(rel_path)

    rng = random.Random(seed)
    samples = []
    for label, files in rows_by_class.items():
        files = sorted(files)
        if n_per_class is not None:
            rng.shuffle(files)
            files = files[:n_per_class]
        for rel_path in files:
            samples.append((os.path.join(task7_data_root, rel_path), label))

    print(f"[{domain} dev-test] evaluating {len(samples)} clips across {len(rows_by_class)} classes")

    results = []
    for chunk in batched(samples, bs=16):
        audios = np.stack([load_audio_fixed(p) for p, _ in chunk])
        audio_t = torch.from_numpy(audios).to(DEVICE)
        out = adil_predict(model, audio_t)
        for i, (path, label) in enumerate(chunk):
            results.append({
                'file': os.path.basename(path),
                'true_label': label,
                'true_idx': DICT_CLASS_LABELS[label],
                'routed_task': int(out['routed_task'][i]),
                'pred_idx': int(out['routed_pred'][i]),
                'pred_label': IDX_TO_CLASS[int(out['routed_pred'][i])],
                'routed_entropy': float(out['routed_entropy'][i]),
                'entropy_task0': float(out['entropy_per_task'][0, i]),
                'entropy_task1': float(out['entropy_per_task'][1, i]),
                'entropy_task2': float(out['entropy_per_task'][2, i]),
            })
    return results


def summarize(results, tag):
    n = len(results)
    acc = sum(r['pred_idx'] == r['true_idx'] for r in results) / n
    route_counts = {0: 0, 1: 0, 2: 0}
    for r in results:
        route_counts[r['routed_task']] += 1
    mean_entropy = float(np.mean([r['routed_entropy'] for r in results]))
    print(f"\n=== {tag} ===")
    print(f"  n={n}  accuracy={acc*100:.1f}%  mean routed entropy={mean_entropy:.3f}")
    print(f"  routed-to: D1={route_counts[0]} D2={route_counts[1]} D3={route_counts[2]} "
          f"({route_counts[0]/n*100:.1f}% / {route_counts[1]/n*100:.1f}% / {route_counts[2]/n*100:.1f}%)")
    per_class = {}
    for r in results:
        per_class.setdefault(r['true_label'], []).append(r['pred_idx'] == r['true_idx'])
    for cls, vals in sorted(per_class.items()):
        print(f"    {cls:20s} n={len(vals):3d}  acc={100*sum(vals)/len(vals):5.1f}%")
    return {
        'tag': tag, 'n': n, 'accuracy': acc, 'mean_entropy': mean_entropy,
        'route_pct': {k: v / n for k, v in route_counts.items()},
        'per_class_acc': {c: sum(v) / len(v) for c, v in per_class.items()},
    }


def main():
    esc50_root = sys.argv[1] if len(sys.argv) > 1 else \
        "/private/tmp/claude-501/-Users-aakashdivakar-Downloads-Idea-dcase-task7/0cb50178-df08-4dec-89d7-264becba8294/scratchpad/ESC-50-master"
    task7_root = os.path.join(REPO_ROOT, "task7_data")

    print(f"device: {DEVICE}")
    model = load_model()

    esc_results = eval_esc50(model, esc50_root)
    d2_results = eval_in_domain(model, task7_root, "D2", n_per_class=40, seed=0)
    d3_results = eval_in_domain(model, task7_root, "D3", n_per_class=40, seed=0)

    summary = {}
    summary['esc50_ood'] = summarize(esc_results, "ESC-50 (unseen domain, 7-class subset)")
    summary['d2_indomain'] = summarize(d2_results, "D2 dev-test (in-domain reference, 40/class)")
    summary['d3_indomain'] = summarize(d3_results, "D3 dev-test (in-domain reference, 40/class)")

    out_path = os.path.join(os.path.dirname(__file__), "esc50_probe_results.json")
    with open(out_path, "w") as f:
        json.dump({
            'summary': summary,
            'esc50_raw': esc_results,
            'd2_raw': d2_results,
            'd3_raw': d3_results,
        }, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
