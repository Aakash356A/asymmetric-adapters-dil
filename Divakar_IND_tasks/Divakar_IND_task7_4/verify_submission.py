"""
Submission self-check + eval inference for DCASE 2026 Task 7.
Uses Mahalanobis distance-based domain routing instead of entropy routing.

Usage
-----
    python submission/Divakar_IND_task7_4/verify_submission.py \\
        --checkpoint_dir /path/to/BN_RES_DEEP_CLF_lr1e-3_bs32_ep180_deep_clf_d3_ep180 \\
        --router_pkl    /path/to/mahal_router.pkl \\
        --threshold     60 \\
        [--eval_data_dir task7_data/eval_data] \\
        [--output_tsv   Divakar_IND_task7_4.output.tsv] \\
        [--cuda] [--batch_size 16] [--num_workers 4]

Steps
-----
  1. Import the standalone model module from this folder.
  2. Build the model with ADAPTER_DEPTHS = {1: 1, 2: 3}.
  3. Load checkpoint_D3.pth (strict=False, reports mismatches).
  4. Load Mahalanobis router statistics from pkl.
  5. Sanity-check forward pass on a dummy waveform.
  6. Run inference: extract D1-BN embedding, route via Mahalanobis distance,
     classify with the routed domain head.
  7. Write predictions as:
       - <output_tsv>             (TSV: filename, predicted_class, domain_used, confidence)
       - <output_tsv_stem>.csv    (CSV: filename, class  — submission simple format)
"""

import os
import sys
import argparse
import pickle
from collections import defaultdict

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# Also add utils for config + dataset
_UTILS = os.path.join(HERE, '..', '..', 'code', 'utils')
if os.path.isdir(_UTILS):
    sys.path.insert(0, os.path.abspath(_UTILS))

import Divakar_IND_task7_4 as M

NB_TASKS = 3

CLASSES = [
    'alarm', 'baby_cry', 'bark', 'engine', 'fire',
    'footsteps', 'knock', 'telephone_ringing', 'piano', 'speech',
]
TASK_TO_DOMAIN = {0: 'D1', 1: 'D2', 2: 'D3'}
_ROUTE_TASK_OF_CLASS = {0: 1, 1: 2}   # router class -> model task (D2=1, D3=2)


# ---------------------------------------------------------------------------
# Mahalanobis Router (minimal, loads from pkl saved by mahalanobis_router.py)
# ---------------------------------------------------------------------------
class MahalanobisRouter:
    def __init__(self):
        self.means = {}
        self.inv_covs = {}
        self.num_domains = 2

    def load(self, path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.means     = state['means']
        self.inv_covs  = state['inv_covs']
        self.num_domains = state['num_domains']
        print(f'  [router] loaded from {path}  '
              f'(num_domains={self.num_domains})')

    def distance(self, emb, domain_id):
        diff = emb - self.means[domain_id]
        inv  = self.inv_covs.get(domain_id)
        if inv is None:
            return float('inf')
        return float(np.sqrt(diff @ inv @ diff))

    def route(self, emb, threshold=None):
        """Returns (routed_task, domain_str, min_dist)."""
        dists = {i: self.distance(emb, i) for i in range(self.num_domains)}
        min_dist  = min(dists.values())
        best_cls  = min(dists, key=dists.get)
        if threshold is not None and min_dist > threshold:
            return 0, 'D1', min_dist          # rejected -> D1
        return _ROUTE_TASK_OF_CLASS[best_cls], TASK_TO_DOMAIN[_ROUTE_TASK_OF_CLASS[best_cls]], min_dist

# ---------------------------------------------------------------------------
# Dataset for unlabeled eval audio
# ---------------------------------------------------------------------------
class EvalDataset(Dataset):
    """Loads WAV files from a directory; returns (waveform [T], filename)."""

    def __init__(self, audio_dir: str):
        self.audio_dir = audio_dir
        self.files = sorted(
            f for f in os.listdir(audio_dir) if f.lower().endswith('.wav')
        )
        if not self.files:
            raise FileNotFoundError(f'No WAV files found in {audio_dir}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        fpath = os.path.join(self.audio_dir, fname)
        clip = M.SAMPLE_RATE * 4   # 4-second window = 128 000 samples
        audio, _ = librosa.load(fpath, sr=M.SAMPLE_RATE, mono=True, dtype=np.float32)
        if audio.ndim != 1:
            audio = np.ravel(audio)
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) < clip:
            audio = np.pad(audio, (0, clip - len(audio)), mode='constant')
        else:
            audio = audio[:clip]
        return torch.from_numpy(audio).float(), fname


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_and_check(model, ckpt_path):
    print(f"\nLoading  {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        print("  ERROR — file not found")
        return ["<file missing>"], []
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state and all(
        not k.startswith(("bn0", "conv_block", "fc", "adapters"))
        for k in list(state.keys())[:5]
    ):
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  MISSING keys ({len(missing)}): "
              f"{missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"  UNEXPECTED keys ({len(unexpected)}): "
              f"{unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
    if not missing and not unexpected:
        print("  OK — exact match")
    return missing, unexpected


# ---------------------------------------------------------------------------
# Dev-test evaluation (labeled split, per-domain and per-class accuracy)
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_devtest_eval(model, router, threshold, device):
    try:
        import config_task7 as config
        from datasetfactory_task7 import DILDatasetInc as DILDataset
    except ImportError:
        print('  [devtest] Could not import config_task7 / datasetfactory_task7 — skipping.')
        return

    _ROUTE_DOMAINS     = [['D2'], ['D3']]
    _ROUTE_TASK_OF_CLASS = {0: 1, 1: 2}

    df = config.df_DIL_dev_test
    audio_folder = config.audio_folder_DIL

    # accumulators
    route_correct = route_total = d1_rejections = 0
    per_domain_cls = defaultdict(lambda: [0, 0])           # dom -> [correct, total]
    per_domain_class = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # dom -> class -> [c, t]
    class_names    = {v: k for k, v in config.dict_class_labels.items()}

    print('\n' + '=' * 56)
    print('  Dev-test evaluation (Mahalanobis routing)')
    print('=' * 56)

    for true_router_cls, dom_list in enumerate(_ROUTE_DOMAINS):
        sub = df[df['domain'].isin(dom_list)]
        ds  = DILDataset(sub, audio_folder, training=False)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        dom_name = dom_list[0]

        for audio, label, _ in tqdm(loader, desc=f'  {dom_name}', leave=False):
            audio = audio.float().to(device)
            gt    = int(torch.argmax(label, dim=-1).item())

            emb = model.get_embedding(audio, task=0)
            emb_np = emb[0].detach().cpu().numpy().astype(np.float64)
            routed_task, domain_str, _ = router.route(emb_np, threshold=threshold)

            if domain_str == 'D1':
                d1_rejections += 1
            else:
                route_correct += int(_ROUTE_TASK_OF_CLASS[true_router_cls] == routed_task)
            route_total += 1

            logits    = model(audio, task=routed_task)
            pred      = int(torch.argmax(logits, dim=1).item())
            correct   = int(pred == gt)
            cls_name  = class_names.get(gt, str(gt))

            per_domain_cls[dom_name][0] += correct
            per_domain_cls[dom_name][1] += 1
            per_domain_class[dom_name][cls_name][0] += correct
            per_domain_class[dom_name][cls_name][1] += 1

    print(f'\n  Routing accuracy (D2 vs D3, excl. D1 rejections):')
    non_rejected = route_total - d1_rejections
    print(f'    {route_correct}/{non_rejected} = '
          f'{100*route_correct/max(non_rejected,1):.2f}%')
    print(f'  D1 rejections: {d1_rejections}/{route_total} '
          f'({100*d1_rejections/max(route_total,1):.2f}%)')

    print()
    for dom in ['D2', 'D3']:
        c, t = per_domain_cls[dom]
        avg  = 100.0 * c / max(t, 1)
        print(f'      {dom}:')
        print(f'        average: {avg:.2f}')
        for cls in sorted(per_domain_class[dom]):
            cc, ct = per_domain_class[dom][cls]
            print(f'        {cls}: {100.0*cc/max(ct,1):.1f}')
        print()
    print('=' * 56)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Verify submission and run eval inference (Mahalanobis routing).')
    parser.add_argument('--checkpoint_dir', type=str, default=HERE,
                        help='Checkpoint directory containing checkpoint_D3.pth.')
    parser.add_argument('--router_pkl', type=str, required=True,
                        help='Path to saved Mahalanobis router pkl file.')
    parser.add_argument('--threshold', type=float, default=60.0,
                        help='Mahalanobis distance threshold for D1 rejection (default: 60).')
    parser.add_argument('--eval_data_dir', type=str,
                        default=os.path.join(HERE, 'eval_data'),
                        help='Directory with unlabeled WAV files.')
    parser.add_argument('--output_tsv', type=str,
                        default=os.path.join(HERE, 'Divakar_IND_task7_4.output.tsv'),
                        help='Where to write the full DCASE prediction TSV.')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--devtest', action='store_true',
                        help='Evaluate on dev-test split (with labels) and print per-domain/per-class accuracy.')
    parser.add_argument('--devtest-only', dest='devtest_only', action='store_true',
                        help='Same as --devtest but skip eval-folder inference entirely (no TSV/CSV written).')
    args = parser.parse_args()

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    print("=" * 70)
    print("DCASE 2026 Task 7 — Mahalanobis routing submission")
    print("=" * 70)
    print(f"adapter depths : {M.ADAPTER_DEPTHS}")
    print(f"threshold      : {args.threshold}")
    print(f"checkpoint_dir : {args.checkpoint_dir}")
    print(f"router_pkl     : {args.router_pkl}")
    print(f"eval_data_dir  : {os.path.abspath(args.eval_data_dir)}")
    print(f"device         : {device}")

    # ── 1. Build model and load checkpoint ───────────────────────────────────
    model = M.build_model(nb_tasks=NB_TASKS, adapter_depths=M.ADAPTER_DEPTHS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params   : {n_params / 1e6:.2f} M")

    ckpt_d3 = os.path.join(args.checkpoint_dir, 'checkpoint_D3.pth')
    missing, unexpected = _load_and_check(model, ckpt_d3)
    problems = len(missing) + len(unexpected)
    model.eval()
    model.to(device)

    # ── 2. Load Mahalanobis router ───────────────────────────────────────────
    router = MahalanobisRouter()
    router.load(args.router_pkl)

    # ── 3. Sanity-check forward pass ─────────────────────────────────────────
    clip = M.SAMPLE_RATE * 4
    wav  = torch.randn(2, clip, device=device)
    print(f"\nForward-pass check on dummy input {tuple(wav.shape)}:")
    with torch.no_grad():
        for t in range(NB_TASKS):
            out = model(wav, task=t)
            ok  = out.shape == (2, M.CLASSES_NUM)
            print(f"  task {t}: logits {tuple(out.shape)}  {'OK' if ok else 'SHAPE MISMATCH'}")
            if not ok:
                problems += 1
        emb_test = model.get_embedding(wav, task=0)
        print(f"  D1 embedding: {tuple(emb_test.shape)}  "
              f"{'OK' if emb_test.shape == (2, 2048) else 'SHAPE MISMATCH'}")

    if problems:
        print(f"\nWARNING — {problems} issue(s) found. Continuing anyway.\n")
    else:
        print("\nModel checks passed.\n")

    # ── 4a. Optional: dev-test evaluation ────────────────────────────────────
    if args.devtest or args.devtest_only:
        run_devtest_eval(model, router, args.threshold, device)

    if args.devtest_only:
        print("\nDev-test only mode — skipping eval-folder inference.")
        sys.exit(1 if problems else 0)

    # ── 4. Run inference on eval_data ────────────────────────────────────────
    eval_dir = os.path.abspath(args.eval_data_dir)
    dataset  = EvalDataset(eval_dir)
    loader   = DataLoader(dataset, batch_size=1, shuffle=False,
                          num_workers=max(0, args.num_workers),
                          pin_memory=(device == 'cuda'))
    print(f"Found {len(dataset)} eval files in {eval_dir}")

    rows = []   # (filename, predicted_class, domain, confidence)
    with torch.inference_mode():
        for waveforms, filenames in tqdm(loader, desc='Mahalanobis routing inference'):
            waveforms = waveforms.to(device)

            # Extract D1-BN embedding for routing
            emb = model.get_embedding(waveforms, task=0)   # (1, 2048)
            emb_np = emb[0].cpu().numpy().astype(np.float64)

            routed_task, domain_str, _ = router.route(emb_np, threshold=args.threshold)

            logits = model(waveforms, task=routed_task)
            probs  = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

            rows.append((
                filenames[0],
                CLASSES[pred_idx[0].item()],
                domain_str,
                conf[0].item(),
            ))

    # ── 5. Write outputs ─────────────────────────────────────────────────────
    total         = len(rows)
    domain_counts = {}
    class_counts  = {}
    for _, cls, dom, _ in rows:
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        class_counts[cls]  = class_counts.get(cls, 0)  + 1

    ckpt_name = os.path.basename(os.path.normpath(args.checkpoint_dir))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_tsv)), exist_ok=True)

    # Full TSV (with metadata headers)
    with open(args.output_tsv, 'w') as f:
        f.write(f'# checkpoint : {ckpt_name}\n')
        f.write(f'# router_pkl : {os.path.basename(args.router_pkl)}\n')
        f.write(f'# threshold  : {args.threshold}\n')
        f.write(f'# model_type : bn_res_deep_clf\n')
        f.write(f'# adapter_depths : {M.ADAPTER_DEPTHS}\n')
        f.write(f'# eval_dir   : {eval_dir}\n')
        f.write(f'# total_files: {total}\n')
        f.write('#\n# domain_distribution:\n')
        for d in ['D1', 'D2', 'D3']:
            n = domain_counts.get(d, 0)
            f.write(f'#   {d}: {n} ({100 * n / total:.1f}%)\n')
        f.write('#\n# class_distribution:\n')
        for cls in sorted(class_counts, key=class_counts.get, reverse=True):
            f.write(f'#   {cls}: {class_counts[cls]}\n')
        f.write('#\n')
        f.write('filename\tpredicted_class\tdomain_used\tconfidence\n')
        for fname, cls, dom, conf in rows:
            f.write(f'{fname}\t{cls}\t{dom}\t{conf:.4f}\n')

    # Simple CSV (filename\tclass only — DCASE submission format)
    simple_csv = os.path.splitext(args.output_tsv)[0] + '_simple.csv'
    with open(simple_csv, 'w') as f:
        for fname, cls, _, _ in rows:
            f.write(f'{fname}\t{cls}\n')

    print(f'\nWrote {total} predictions')
    print(f'  Full TSV : {args.output_tsv}')
    print(f'  Simple CSV: {simple_csv}')
    print('Domain usage:')
    for d in ['D1', 'D2', 'D3']:
        n = domain_counts.get(d, 0)
        print(f'  {d}: {n} ({100 * n / total:.1f}%)')
    print('Predicted class distribution:')
    for cls in sorted(class_counts, key=class_counts.get, reverse=True):
        print(f'  {cls}: {class_counts[cls]}')

    print("\n" + "=" * 70)
    if problems == 0:
        print("ALL CHECKS PASSED — model is self-consistent.")
    else:
        print(f"DONE with {problems} model-key issue(s) — see above.")
    print("=" * 70)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
