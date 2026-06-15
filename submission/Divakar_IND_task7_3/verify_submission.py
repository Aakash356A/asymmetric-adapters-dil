"""
Submission self-check + eval inference for DCASE 2026 Task 7.

Verifies that the standalone model loads cleanly from a trained checkpoint,
then runs entropy-based domain-agnostic inference over the unlabeled eval_data
and writes the DCASE submission TSV + a simple filename/class CSV.

Usage
-----
    python submission/Divakar_IND_task7_3/verify_submission.py \\
        --checkpoint_dirs /data/dcase_2026/audio_classification/dcase_task7/checkpoints/BN_RES_DEEP_CLF_lr1e-3_bs32_ep180_deep_clf_d3_ep180,/data/dcase_2026/audio_classification/dcase_task7/checkpoints/BN_RES_DEEP_CLF_lr1e-3_bs32_ep240_deep_clf_d3_sup02_aug_5  \\
        [--checkpoint_weights 1.0,1.0] \\
        [--eval_data_dir task7_data/eval_data] \\
        [--output_tsv Divakar_IND_task7_3.output.tsv] \\
        [--cuda] [--batch_size 16] [--num_workers 4] [--tta_n 1]
        [--devtest]       # print per-domain/per-class accuracy on labeled dev-test
        [--devtest-only]  # same as --devtest but skip eval-folder inference
"""

import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_UTILS = os.path.join(HERE, '..', '..', 'code', 'utils')
if os.path.isdir(_UTILS):
    sys.path.insert(0, os.path.abspath(_UTILS))

import Divakar_IND_task7_3 as M

NB_TASKS = 3

CLASSES = [
    'alarm', 'baby_cry', 'bark', 'engine', 'fire',
    'footsteps', 'knock', 'telephone_ringing', 'piano', 'speech',
]
TASK_TO_DOMAIN = {0: 'D1', 1: 'D2', 2: 'D3'}


# ---------------------------------------------------------------------------
# Dataset for unlabeled eval audio
# ---------------------------------------------------------------------------
class EvalDataset(Dataset):
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
        fname  = self.files[idx]
        fpath  = os.path.join(self.audio_dir, fname)
        clip   = M.SAMPLE_RATE * 4
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
def _tta_predict(model, audio: torch.Tensor, task: int, tta_n: int = 1) -> torch.Tensor:
    if tta_n <= 1:
        return torch.softmax(model(audio, task), dim=1)
    T    = audio.shape[-1]
    step = max(1, T // (tta_n * 4))
    p    = torch.softmax(model(audio, task), dim=1)
    for i in range(1, tta_n):
        p = p + torch.softmax(model(torch.roll(audio, step * i, dims=-1), task), dim=1)
    return p / tta_n


def _load_and_check(model, ckpt_path: str):
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
def run_devtest_eval(models, weights, device, tta_n=1):
    try:
        import config_task7 as config
        from datasetfactory_task7 import DILDatasetInc as DILDataset
    except ImportError:
        print('  [devtest] Could not import config_task7 / datasetfactory_task7 — skipping.')
        return

    _ROUTE_DOMAINS = [['D2'], ['D3']]
    weight_sum     = float(sum(weights))
    class_names    = {v: k for k, v in config.dict_class_labels.items()}

    per_domain_cls   = defaultdict(lambda: [0, 0])
    per_domain_class = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    print('\n' + '=' * 56)
    print('  Dev-test evaluation (entropy routing)')
    print('=' * 56)

    for dom_list in _ROUTE_DOMAINS:
        sub    = config.df_DIL_dev_test[config.df_DIL_dev_test['domain'].isin(dom_list)]
        ds     = DILDataset(sub, config.audio_folder_DIL, training=False)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
        dom_name = dom_list[0]

        for audio, label, _ in tqdm(loader, desc=f'  {dom_name}', leave=False):
            audio = audio.float().to(device)
            gt    = int(torch.argmax(label, dim=-1).item())

            task_probs = []
            for task in range(NB_TASKS):
                task_prob = None
                for idx, model in enumerate(models):
                    p = _tta_predict(model, audio, task, tta_n) * weights[idx]
                    task_prob = p if task_prob is None else task_prob + p
                task_probs.append(task_prob / weight_sum)

            stacked  = torch.stack(task_probs, dim=0)
            eps      = torch.finfo(torch.float32).tiny
            entropy  = -(stacked * torch.log(stacked + eps)).sum(-1)
            best     = entropy.argmin(dim=0)[0].item()
            logits   = stacked[best, 0, :]
            pred     = int(logits.argmax().item())
            correct  = int(pred == gt)
            cls_name = class_names.get(gt, str(gt))

            per_domain_cls[dom_name][0]          += correct
            per_domain_cls[dom_name][1]          += 1
            per_domain_class[dom_name][cls_name][0] += correct
            per_domain_class[dom_name][cls_name][1] += 1

    print()
    for dom in ['D2', 'D3']:
        c, t = per_domain_cls[dom]
        print(f'      {dom}:')
        print(f'        average: {100.0*c/max(t,1):.2f}')
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
        description='Verify submission and run eval inference (entropy routing).')
    parser.add_argument('--checkpoint_dir', type=str, default=HERE)
    parser.add_argument('--checkpoint_dirs', type=str, default=None,
                        help='Comma-separated checkpoint directories for ensembling.')
    parser.add_argument('--checkpoint_weights', type=str, default=None)
    parser.add_argument('--eval_data_dir', type=str,
                        default=os.path.join(HERE, 'eval_data'))
    parser.add_argument('--output_tsv', type=str,
                        default=os.path.join(HERE, 'Divakar_IND_task7_3.output.tsv'))
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--tta_n', type=int, default=1)
    parser.add_argument('--devtest', action='store_true',
                        help='Print per-domain/per-class accuracy on labeled dev-test.')
    parser.add_argument('--devtest-only', dest='devtest_only', action='store_true',
                        help='Same as --devtest but skip eval-folder inference entirely.')
    args = parser.parse_args()

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    print("=" * 70)
    print("DCASE 2026 Task 7 — submission verify + eval inference")
    print("=" * 70)
    print(f"adapter depths : {M.ADAPTER_DEPTHS}")

    ckpt_dirs = ([d.strip() for d in args.checkpoint_dirs.split(',') if d.strip()]
                 if args.checkpoint_dirs else [args.checkpoint_dir])
    weights   = ([float(w.strip()) for w in args.checkpoint_weights.split(',') if w.strip()]
                 if args.checkpoint_weights else [1.0] * len(ckpt_dirs))
    if len(weights) != len(ckpt_dirs):
        raise ValueError('--checkpoint_weights length must match --checkpoint_dirs')

    print(f"checkpoint_dirs   : {ckpt_dirs}")
    print(f"checkpoint_weights: {weights}")
    print(f"eval_data_dir     : {os.path.abspath(args.eval_data_dir)}")
    print(f"device            : {device}")

    # ── 1. Build and load models ─────────────────────────────────────────────
    models   = []
    problems = 0
    for ckpt_dir in ckpt_dirs:
        model = M.build_model(nb_tasks=NB_TASKS, adapter_depths=M.ADAPTER_DEPTHS)
        print(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M  ({ckpt_dir})")
        missing, unexpected = _load_and_check(model, os.path.join(ckpt_dir, 'checkpoint_D3.pth'))
        problems += len(missing) + len(unexpected)
        model.eval().to(device)
        models.append(model)

    # ── 2. Sanity-check forward pass ─────────────────────────────────────────
    clip = M.SAMPLE_RATE * 4
    wav  = torch.randn(2, clip, device=device)
    print(f"\nForward-pass check on dummy input {tuple(wav.shape)}:")
    with torch.no_grad():
        for mi, model in enumerate(models):
            for t in range(NB_TASKS):
                out = model(wav, task=t)
                ok  = out.shape == (2, M.CLASSES_NUM)
                print(f"  model {mi} task {t}: {tuple(out.shape)}  {'OK' if ok else 'SHAPE MISMATCH'}")
                if not ok:
                    problems += 1

    if problems:
        print(f"\nWARNING — {problems} issue(s) found. Continuing anyway.\n")
    else:
        print("\nModel checks passed.\n")

    # ── 3. Optional dev-test evaluation ──────────────────────────────────────
    if args.devtest or args.devtest_only:
        run_devtest_eval(models, weights, device, tta_n=args.tta_n)

    if args.devtest_only:
        print("\nDev-test only mode — skipping eval-folder inference.")
        sys.exit(1 if problems else 0)

    # ── 4. Run inference on eval_data ────────────────────────────────────────
    eval_dir = os.path.abspath(args.eval_data_dir)
    dataset  = EvalDataset(eval_dir)
    loader   = DataLoader(dataset, batch_size=max(1, args.batch_size),
                          shuffle=False, num_workers=max(0, args.num_workers),
                          pin_memory=(device == 'cuda'))
    print(f"Found {len(dataset)} eval files in {eval_dir}")

    rows       = []
    weight_sum = float(sum(weights))
    with torch.inference_mode():
        for waveforms, filenames in tqdm(loader, desc='Evaluating'):
            waveforms = waveforms.to(device, non_blocking=(device == 'cuda'))
            B = waveforms.shape[0]

            task_probs = []
            for task in range(NB_TASKS):
                task_prob = None
                for idx, model in enumerate(models):
                    p = _tta_predict(model, waveforms, task, args.tta_n) * weights[idx]
                    task_prob = p if task_prob is None else task_prob + p
                task_probs.append(task_prob / weight_sum)

            stacked    = torch.stack(task_probs, dim=0)
            eps        = torch.finfo(torch.float32).tiny
            entropy    = -(stacked * torch.log(stacked + eps)).sum(-1)
            _, best    = torch.min(entropy, dim=0)
            sample_idx = torch.arange(B, device=device)
            final      = stacked[best, sample_idx, :]
            pred_idx   = torch.argmax(final, dim=1)
            conf       = final[sample_idx, pred_idx]

            for b in range(B):
                rows.append((
                    filenames[b],
                    CLASSES[pred_idx[b].item()],
                    TASK_TO_DOMAIN[best[b].item()],
                    conf[b].item(),
                ))

    # ── 5. Write TSV + simple CSV ─────────────────────────────────────────────
    total         = len(rows)
    domain_counts = {}
    class_counts  = {}
    for _, cls, dom, _ in rows:
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        class_counts[cls]  = class_counts.get(cls, 0)  + 1

    ckpt_name = os.path.basename(os.path.normpath(ckpt_dirs[0]))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_tsv)), exist_ok=True)

    with open(args.output_tsv, 'w') as f:
        f.write(f'# checkpoint : {ckpt_name}\n')
        f.write(f'# checkpoint_dirs : {",".join(ckpt_dirs)}\n')
        f.write(f'# checkpoint_weights : {",".join(map(str, weights))}\n')
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

    simple_csv = os.path.splitext(args.output_tsv)[0] + '_simple.csv'
    with open(simple_csv, 'w') as f:
        for fname, cls, _, _ in rows:
            f.write(f'{fname}\t{cls}\n')

    print(f'\nWrote {total} predictions')
    print(f'  Full TSV  : {args.output_tsv}')
    print(f'  Simple CSV: {simple_csv}')
    print('Domain usage:')
    for d in ['D1', 'D2', 'D3']:
        n = domain_counts.get(d, 0)
        print(f'  {d}: {n} ({100 * n / total:.1f}%)')
    print('Predicted class distribution:')
    for cls in sorted(class_counts, key=class_counts.get, reverse=True):
        print(f'  {cls}: {class_counts[cls]}')

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED." if problems == 0 else f"DONE with {problems} issue(s).")
    print("=" * 70)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()


import os
import sys
import argparse

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import Divakar_IND_task7_1 as M  # standalone submitted model (same directory)

NB_TASKS = 3

CLASSES = [
    'alarm', 'baby_cry', 'bark', 'engine', 'fire',
    'footsteps', 'knock', 'telephone_ringing', 'piano', 'speech',
]
TASK_TO_DOMAIN = {0: 'D1', 1: 'D2', 2: 'D3'}

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
def _tta_predict(model, audio: torch.Tensor, task: int, tta_n: int = 1) -> torch.Tensor:
    if tta_n <= 1:
        return torch.softmax(model(audio, task), dim=1)
    T    = audio.shape[-1]
    step = max(1, T // (tta_n * 4))
    p    = torch.softmax(model(audio, task), dim=1)
    for i in range(1, tta_n):
        p = p + torch.softmax(model(torch.roll(audio, step * i, dims=-1), task), dim=1)
    return p / tta_n


def load_and_check(model, ckpt_path: str):
    print(f"\nLoading  {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        print("  ERROR — file not found")
        return ["<file missing>"], []
    state = torch.load(ckpt_path, map_location="cpu")
    # Unwrap if the training pipeline wrapped it in {"model": state_dict}
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
# Main
# ---------------------------------------------------------------------------
def _resolve_device(use_cuda: bool) -> str:
    if use_cuda and torch.cuda.is_available():
        return 'cuda'
    if use_cuda:
        print('CUDA requested but unavailable; falling back to CPU for safety.')
    return 'cpu'


def main():
    parser = argparse.ArgumentParser(
        description='Verify submission and run eval inference.')
    parser.add_argument(
        '--checkpoint_dir', type=str, default=HERE,
        help='Single checkpoint directory containing checkpoint_D3.pth (default: same folder as this script).')
    parser.add_argument(
        '--checkpoint_dirs', type=str, default=None,
        help='Comma-separated list of checkpoint directories for ensembling.')
    parser.add_argument(
        '--checkpoint_weights', type=str, default=None,
        help='Comma-separated weights matching --checkpoint_dirs; defaults to equal weighting.')
    parser.add_argument(
        '--eval_data_dir', type=str,
        default=os.path.join(HERE, 'eval_data'),
        help='Directory with unlabeled WAV files (default: eval_data/ next to this script).')
    parser.add_argument(
        '--output_tsv', type=str,
        default=os.path.join(HERE, 'Divakar_IND_task7_1.output.tsv'),
        help='Where to write the DCASE prediction TSV.')
    parser.add_argument('--cuda', action='store_true',
                        help='Use GPU if available (disabled by default for safety).')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for eval inference (default: 1 for lower memory use).')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (default: 0 for lower memory pressure).')
    parser.add_argument('--tta_n', type=int, default=1,
                        help='Test-time augmentation shifts (1 = off).')
    args = parser.parse_args()

    device = _resolve_device(args.cuda)
    print("=" * 70)
    print("DCASE 2026 Task 7 — submission verify + eval inference")
    print("=" * 70)
    print(f"adapter depths : {M.ADAPTER_DEPTHS}")
    print(f"sample rate    : {M.SAMPLE_RATE} Hz   mel bins: {M.MEL_BINS}   "
          f"classes: {M.CLASSES_NUM}")
    if args.checkpoint_dirs:
        ckpt_dirs = [d.strip() for d in args.checkpoint_dirs.split(',') if d.strip()]
    else:
        ckpt_dirs = [args.checkpoint_dir]
    print(f"checkpoint_dirs: {ckpt_dirs}")
    if args.checkpoint_weights:
        weights = [float(w.strip()) for w in args.checkpoint_weights.split(',') if w.strip()]
        if len(weights) != len(ckpt_dirs):
            raise ValueError('--checkpoint_weights must have the same number of entries as --checkpoint_dirs')
    else:
        weights = [1.0] * len(ckpt_dirs)
    print(f"checkpoint_weights: {weights}")
    print(f"eval_data_dir  : {os.path.abspath(args.eval_data_dir)}")
    print(f"device         : {device}")

    # ── 1. Build models ─────────────────────────────────────────────────────
    models = []
    problems = 0
    for ckpt_dir in ckpt_dirs:
        model = M.build_model(nb_tasks=NB_TASKS, adapter_depths=M.ADAPTER_DEPTHS)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model params   : {n_params / 1e6:.2f} M for {ckpt_dir}\n")

        # D3.pth already contains all domain parameters (D1, D2, D3). Do NOT
        # load D2.pth afterwards — it would overwrite shared backbone weights
        # with D2's values, corrupting D3's adapters.
        ckpt_d3 = os.path.join(ckpt_dir, 'checkpoint_D3.pth')
        missing, unexpected = load_and_check(model, ckpt_d3)
        problems += len(missing) + len(unexpected)
        model.eval()
        model.to(device)
        models.append(model)

    # ── 2. Sanity-check forward pass ─────────────────────────────────────────
    clip = M.SAMPLE_RATE * 4
    wav  = torch.randn(2, clip, device=device)
    print(f"\nForward-pass check on dummy input {tuple(wav.shape)}:")
    with torch.no_grad():
        for model_idx, model in enumerate(models):
            for t in range(NB_TASKS):
                out = model(wav, task=t)
                ok  = out.shape == (2, M.CLASSES_NUM)
                print(f"  model {model_idx} task {t}: logits {tuple(out.shape)}  {'OK' if ok else 'SHAPE MISMATCH'}")
                if not ok:
                    problems += 1
            pred = M.predict_domain_agnostic(model, wav, nb_tasks=NB_TASKS)
            print(f"  model {model_idx} ADIL prediction: {pred.tolist()}")
            if pred.shape != (2,):
                problems += 1

    if problems:
        print(f"\nWARNING — {problems} issue(s) found above. "
              f"Continuing with eval inference anyway.\n")
    else:
        print("\nModel checks passed.\n")

    # ── 3. Run inference on eval_data ────────────────────────────────────────
    eval_dir = os.path.abspath(args.eval_data_dir)
    dataset  = EvalDataset(eval_dir)
    loader   = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=(device == 'cuda'))
    print(f"Found {len(dataset)} eval files in {eval_dir}")

    rows = []   # (filename, predicted_class, domain, confidence)
    weight_sum = float(sum(weights))
    with torch.inference_mode():
        for waveforms, filenames in tqdm(loader, desc='Evaluating'):
            waveforms = waveforms.to(device, non_blocking=(device == 'cuda'))
            B = waveforms.shape[0]

            # Average class probabilities across the ensemble members first,
            # then route via entropy over the averaged task-specific distributions.
            task_probs = []
            for task in range(NB_TASKS):
                task_prob = None
                for model_idx, model in enumerate(models):
                    p = _tta_predict(model, waveforms, task, args.tta_n)
                    p = p * weights[model_idx]
                    task_prob = p if task_prob is None else task_prob + p
                task_probs.append(task_prob / weight_sum)

            stacked = torch.stack(task_probs, dim=0)  # (NB_TASKS, B, C)
            eps = torch.finfo(torch.float32).tiny
            entropy   = -(stacked * torch.log(stacked + eps)).sum(-1)  # (T, B)
            _, best   = torch.min(entropy, dim=0)                       # (B,)
            sample_idx = torch.arange(B, device=device)
            final     = stacked[best, sample_idx, :]                   # (B, C)

            pred_idx  = torch.argmax(final, dim=1)
            conf      = final[sample_idx, pred_idx]

            for b in range(B):
                rows.append((
                    filenames[b],
                    CLASSES[pred_idx[b].item()],
                    TASK_TO_DOMAIN[best[b].item()],
                    conf[b].item(),
                ))

    # ── 4. Write DCASE submission TSV ────────────────────────────────────────
    total         = len(rows)
    domain_counts = {}
    class_counts  = {}
    for _, cls, dom, _ in rows:
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        class_counts[cls]  = class_counts.get(cls, 0) + 1

    ckpt_name = os.path.basename(os.path.normpath(ckpt_dirs[0]))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_tsv)), exist_ok=True)
    with open(args.output_tsv, 'w') as f:
        f.write(f'# checkpoint : {ckpt_name}\n')
        f.write(f'# checkpoint_dirs : {",".join(ckpt_dirs)}\n')
        f.write(f'# checkpoint_weights : {",".join(map(str, weights))}\n')
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

    print(f'\nWrote {total} predictions → {args.output_tsv}')
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
