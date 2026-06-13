"""
Evaluate unlabeled audio files in task7_data/eval_data/ using a trained
domain-incremental model.  For each file, the domain (D1/D2/D3) is chosen
automatically via entropy minimisation across all task heads.

Output (TSV):
    filename  predicted_class  domain_used  confidence

Usage:
    python evaluate_eval_data.py \
        --checkpoint_dir /home/aakash/dcase_task7/checkpoints/BN_RES_CLF_lr1e-3_bs32_ep120_clf_supcon_0.2 \
        --model_type bn_res_clf \
        [--output_file results/eval_predictions.tsv] \
        [--eval_data_dir task7_data/eval_data] \
        [--cuda] [--num_workers 4] [--tta_n 1]
"""

import os
import sys
import argparse

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ── Add code paths so model and config modules are importable ──────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_BASELINE_DIR = os.path.join(_SCRIPT_DIR, 'code', 'baseline')
_UTILS_DIR    = os.path.join(_SCRIPT_DIR, 'code', 'utils')
sys.path.insert(0, _BASELINE_DIR)
sys.path.insert(0, _UTILS_DIR)

from domain_net import MCnn14
from domain_net_dtn import MCnn14DTN, MCnn14Mixed, MCnn14DTN2, MCnn14GRN
from domain_net_arch import (
    MCnn14CLF, MCnn14Cos, MCnn14FiLM,
    MCnn14SE, MCnn14Att,
    MCnn14SEDeep, MCnn14SEClf, MCnn14Res,
    MCnn14GN, MCnn14GNSE,
    MCnn14ResAdil, MCnn14ResClf,
    MCnn14ResDeep, MCnn14ResDeepClf,
    set_adapter_depths,
)

# Audio constants (from config_task7)
sample_rate  = 32000
clip_samples = sample_rate * 4   # 4-second clips = 128 000 samples
mel_bins     = 64
fmin         = 50
fmax         = 14000
window_size  = 1024
hop_size     = 320
classes_num  = 10

# ── Constants ────────────────────────────────────────────────────────────────
CLASSES = [
    'alarm', 'baby_cry', 'bark', 'engine', 'fire',
    'footsteps', 'knock', 'telephone_ringing', 'piano', 'speech',
]
TASK_TO_DOMAIN = {0: 'D1', 1: 'D2', 2: 'D3'}

MODEL_ZOO = {
    'bn':               MCnn14,
    'dtn':              MCnn14DTN,
    'dtn2':             MCnn14DTN2,
    'mixed':            MCnn14Mixed,
    'bn_clf':           MCnn14CLF,
    'bn_cos':           MCnn14Cos,
    'bn_film':          MCnn14FiLM,
    'bn_se':            MCnn14SE,
    'bn_att':           MCnn14Att,
    'bn_se_deep':       MCnn14SEDeep,
    'bn_se_clf':        MCnn14SEClf,
    'bn_res':           MCnn14Res,
    'bn_gn':            MCnn14GN,
    'bn_gn_se':         MCnn14GNSE,
    'bn_res_adil':      MCnn14ResAdil,
    'bn_res_clf':       MCnn14ResClf,
    'bn_res_deep':      MCnn14ResDeep,
    'bn_res_deep_clf':  MCnn14ResDeepClf,
    'bn_grn':           MCnn14GRN,
}


# ── Dataset for unlabeled evaluation audio ──────────────────────────────────
class EvalDataset(Dataset):
    """Loads raw WAV files from a directory; returns (waveform, filename)."""

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
        audio, _ = librosa.core.load(fpath, sr=sample_rate, mono=True)
        # Pad to clip_samples (4 s); truncate if longer
        if len(audio) < clip_samples:
            audio = np.concatenate([audio, np.zeros(clip_samples - len(audio))])
        else:
            audio = audio[:clip_samples]
        return torch.from_numpy(audio).float(), fname


# ── Helpers ──────────────────────────────────────────────────────────────────
def _tta_predict(model, audio: torch.Tensor, task: int, tta_n: int = 1) -> torch.Tensor:
    """Return softmax probabilities, optionally averaged over time-shifted copies."""
    if tta_n <= 1:
        return torch.softmax(model(audio, task), dim=1)
    T = audio.shape[-1]
    step = max(1, T // (tta_n * 4))
    probs = torch.softmax(model(audio, task), dim=1)
    for i in range(1, tta_n):
        shifted = torch.roll(audio, step * i, dims=-1)
        probs = probs + torch.softmax(model(shifted, task), dim=1)
    return probs / tta_n


def build_model(model_type: str, num_tasks: int, adapter_depths: str = None):
    """Instantiate the model and optionally configure adapter depths."""
    if adapter_depths:
        depth_list = [int(d) for d in adapter_depths.split(',') if d.strip()]
        set_adapter_depths({t + 1: depth_list[t] for t in range(len(depth_list))})

    Model = MODEL_ZOO.get(model_type)
    if Model is None:
        raise ValueError(f'Unknown model_type "{model_type}". '
                         f'Valid choices: {sorted(MODEL_ZOO)}')
    return Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                 classes_num, num_tasks)


def load_checkpoint(model, checkpoint_dir: str, device: str):
    """Load the D3 checkpoint (final model after all three domains)."""
    ckpt_path = os.path.join(checkpoint_dir, 'checkpoint_D3.pth')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f'Checkpoint not found: {ckpt_path}\n'
            f'Make sure the model was trained on all three domains (D1/D2/D3).'
        )
    state = torch.load(ckpt_path, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        # Print the full mismatch details and abort — do not silently ignore keys.
        print('\n[ERROR] Checkpoint does not match the model architecture.')
        print('  This usually means --model_type or --adapter_depths is wrong.')
        print(f'  Checkpoint : {ckpt_path}')
        print(f'  Model keys : {len(model.state_dict())}')
        print(f'  Ckpt  keys : {len(state)}')
        # Show missing and unexpected keys explicitly
        model_keys = set(model.state_dict().keys())
        ckpt_keys  = set(state.keys())
        missing    = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)
        if missing:
            print(f'\n  Missing keys ({len(missing)}) — in model but not in checkpoint:')
            for k in missing:
                print(f'    {k}')
        if unexpected:
            print(f'\n  Unexpected keys ({len(unexpected)}) — in checkpoint but not in model:')
            for k in unexpected:
                print(f'    {k}')
        raise SystemExit(1) from None
    print(f'Loaded checkpoint from {ckpt_path}')


# ── Main evaluation ──────────────────────────────────────────────────────────
def evaluate(args):
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    print(f'Device: {device}')

    # Build and load model
    num_tasks = 3  # D1, D2, D3
    model = build_model(args.model_type, num_tasks, args.adapter_depths)
    load_checkpoint(model, args.checkpoint_dir, device)
    model.to(device)
    model.eval()

    # Dataset
    eval_dir = os.path.join(_SCRIPT_DIR, 'task7_data', 'eval_data') \
        if args.eval_data_dir is None else args.eval_data_dir
    dataset = EvalDataset(eval_dir)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == 'cuda'),
    )
    print(f'Found {len(dataset)} eval files in {eval_dir}')

    # Output path — default name is <checkpoint_folder_name>_eval_pred.tsv
    ckpt_name = os.path.basename(os.path.normpath(args.checkpoint_dir))
    output_file = args.output_file
    if output_file is None:
        results_dir = os.path.join(_SCRIPT_DIR, 'task7_data', 'results')
        output_file = os.path.join(results_dir, f'{ckpt_name}_eval_pred.tsv')
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # ── Inference with entropy-based domain routing ──────────────────────────
    rows = []  # (filename, predicted_class, domain_used, confidence)

    with torch.no_grad():
        for waveforms, filenames in tqdm(loader, desc='Evaluating'):
            waveforms = waveforms.to(device)
            B = waveforms.shape[0]

            # Forward through every task head
            all_probs = []  # list of (B, C) tensors, one per task
            for task in range(num_tasks):
                p = _tta_predict(model, waveforms, task, args.tta_n)
                all_probs.append(p)

            # Stack → (num_tasks, B, C)
            stacked = torch.stack(all_probs, dim=0)

            # Entropy per task per sample → (num_tasks, B)
            eps = torch.finfo(torch.float32).tiny
            entropy = -torch.sum(stacked * torch.log(stacked + eps), dim=-1)

            # Best task per sample (minimum entropy)
            _, best_tasks = torch.min(entropy, dim=0)  # (B,)
            sample_idx   = torch.arange(B, device=device)
            final_probs  = stacked[best_tasks, sample_idx, :]  # (B, C)

            pred_indices = torch.argmax(final_probs, dim=1)     # (B,)
            confidences  = final_probs[sample_idx, pred_indices] # (B,)

            for b in range(B):
                fname      = filenames[b]
                pred_cls   = CLASSES[pred_indices[b].item()]
                domain     = TASK_TO_DOMAIN[best_tasks[b].item()]
                confidence = confidences[b].item()
                rows.append((fname, pred_cls, domain, confidence))

    # Compute summaries
    domain_counts = {}
    class_counts  = {}
    for _, cls, domain, _ in rows:
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        class_counts[cls]     = class_counts.get(cls, 0) + 1
    total = len(rows)

    # Write results
    with open(output_file, 'w') as fout:
        # ── Metadata header ──────────────────────────────────────────────
        fout.write(f'# checkpoint : {ckpt_name}\n')
        fout.write(f'# model_type : {args.model_type}\n')
        fout.write(f'# eval_dir   : {eval_dir}\n')
        fout.write(f'# total_files: {total}\n')
        fout.write('#\n')
        # Domain distribution
        fout.write('# domain_distribution:\n')
        for d in ['D1', 'D2', 'D3']:
            n = domain_counts.get(d, 0)
            fout.write(f'#   {d}: {n} ({100 * n / total:.1f}%)\n')
        fout.write('#\n')
        # Class distribution
        fout.write('# class_distribution:\n')
        for cls in sorted(class_counts, key=class_counts.get, reverse=True):
            fout.write(f'#   {cls}: {class_counts[cls]}\n')
        fout.write('#\n')
        # ── Data rows ────────────────────────────────────────────────────
        fout.write('filename\tpredicted_class\tdomain_used\tconfidence\n')
        for fname, pred_cls, domain, confidence in rows:
            fout.write(f'{fname}\t{pred_cls}\t{domain}\t{confidence:.4f}\n')

    print(f'\nWrote {total} predictions → {output_file}')

    # Print domain distribution summary
    print('Domain usage:')
    for d in ['D1', 'D2', 'D3']:
        n = domain_counts.get(d, 0)
        print(f'  {d}: {n} ({100 * n / total:.1f}%)')

    # Print class distribution summary
    print('Predicted class distribution:')
    for cls in sorted(class_counts, key=class_counts.get, reverse=True):
        print(f'  {cls}: {class_counts[cls]}')


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run inference on unlabeled eval_data and write predictions to a file.'
    )
    parser.add_argument(
        '--checkpoint_dir', type=str,
        default='/home/aakash/dcase_task7/checkpoints/'
                'BN_RES_CLF_lr1e-3_bs32_ep120_clf_supcon_0.2',
        help='Directory containing checkpoint_D3.pth',
    )
    parser.add_argument(
        '--model_type', type=str, default='bn_res_clf',
        choices=sorted(MODEL_ZOO),
        help='Model architecture (must match the checkpoint)',
    )
    parser.add_argument(
        '--adapter_depths', type=str, default=None,
        help='Per-task adapter depths for bn_res_deep/bn_res_deep_clf, '
             'comma-separated e.g. "1,3"',
    )
    parser.add_argument(
        '--eval_data_dir', type=str, default=None,
        help='Directory with unlabeled WAV files (default: task7_data/eval_data)',
    )
    parser.add_argument(
        '--output_file', type=str, default=None,
        # When None, defaults to task7_data/results/<checkpoint_name>_eval_pred.tsv
        help='Output TSV path (default: task7_data/results/eval_predictions_<timestamp>.tsv)',
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch size for inference',
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='DataLoader worker processes',
    )
    parser.add_argument(
        '--tta_n', type=int, default=1,
        help='Test-time augmentation: number of temporal shifts to average (1=off)',
    )
    parser.add_argument(
        '--cuda', action='store_true', default=False,
        help='Use GPU if available',
    )

    args = parser.parse_args()
    evaluate(args)
