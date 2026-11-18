"""
Submission self-check + eval inference for DCASE 2026 Task 7.

Verifies that the standalone model (Divakar_IND_task7_1.py) loads cleanly
from a trained checkpoint, then runs domain-agnostic inference over the
unlabeled eval_data and writes the DCASE submission TSV.

Usage
-----
    # From the workspace root or from inside this folder:
    python submission/Divakar_IND_task7_1/verify_submission.py \\
        --checkpoint_dir checkpoints/BN_RES_DEEP_CLF_lr1e-3_bs32_ep240_deep_clf_d3_sup02 \\
        [--eval_data_dir task7_data/eval_data] \\
        [--output_tsv submission/Divakar_IND_task7_1/Divakar_IND_task7_1.output.tsv] \\
        [--cuda] [--batch_size 16] [--num_workers 4] [--tta_n 1]

Steps
-----
  1. Import the standalone model module from this folder.
  2. Build the model with ADAPTER_DEPTHS = {1: 1, 2: 3}.
  3. Load Divakar_IND_task7_1_D3_dictionary.pth (strict=False, reports any mismatches).
  4. Sanity-check forward passes for every task on a dummy waveform.
  5. Run inference over all WAV files in eval_data_dir.
  6. Write predictions to output_tsv in DCASE submission format.
"""

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

import Divakar_IND_task7_1_model as M  # standalone submitted model (same directory)

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
        clip  = M.SAMPLE_RATE * 4   # 4-second window = 128 000 samples
        audio, _ = librosa.core.load(fpath, sr=M.SAMPLE_RATE, mono=True)
        # Pad if shorter than 4 s; pass full audio unchanged if longer
        # (matches training-val behaviour: pad_sequence returns x as-is when len >= max_len,
        #  and CNN14's global avg+max pooling is length-agnostic)
        if len(audio) < clip:
            audio = np.concatenate([audio, np.zeros(clip - len(audio))])
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
def main():
    parser = argparse.ArgumentParser(
        description='Verify submission and run eval inference.')
    parser.add_argument(
        '--checkpoint_dir', type=str, default=HERE,
        help='Directory containing the _D2/_D3 dictionary .pth files (default: same folder as this script).')
    parser.add_argument(
        '--eval_data_dir', type=str,
        default=os.path.join(HERE, 'eval_data'),
        help='Directory with unlabeled WAV files (default: eval_data/ next to this script).')
    parser.add_argument(
        '--output_tsv', type=str,
        default=os.path.join(HERE, 'Divakar_IND_task7_1.output.tsv'),
        help='Where to write the DCASE prediction TSV.')
    # parser.add_argument('--cuda', action='store_true', default=)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tta_n', type=int, default=1,
                        help='Test-time augmentation shifts (1 = off).')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 70)
    print("DCASE 2026 Task 7 — submission verify + eval inference")
    print("=" * 70)
    print(f"adapter depths : {M.ADAPTER_DEPTHS}")
    print(f"sample rate    : {M.SAMPLE_RATE} Hz   mel bins: {M.MEL_BINS}   "
          f"classes: {M.CLASSES_NUM}")
    print(f"checkpoint_dir : {args.checkpoint_dir}")
    print(f"eval_data_dir  : {os.path.abspath(args.eval_data_dir)}")
    print(f"device         : {device}")

    # ── 1. Build model ───────────────────────────────────────────────────────
    model = M.build_model(nb_tasks=NB_TASKS, adapter_depths=M.ADAPTER_DEPTHS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params   : {n_params / 1e6:.2f} M\n")

    # ── 2. Load checkpoint (D3 is the final model holding all domain params) ─
    # D3.pth already contains all domain parameters (D1, D2, D3). Do NOT
    # load D2.pth afterwards — it would overwrite shared backbone weights
    # with D2's values, corrupting D3's adapters.
    problems = 0
    ckpt_d3 = os.path.join(args.checkpoint_dir, 'Divakar_IND_task7_1_D3_dictionary.pth')
    missing, unexpected = load_and_check(model, ckpt_d3)
    problems += len(missing) + len(unexpected)

    # ── 3. Sanity-check forward pass ─────────────────────────────────────────
    model.eval()
    model.to(device)
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
    if problems:
        print(f"\nWARNING — {problems} issue(s) found above. "
              f"Continuing with eval inference anyway.\n")
    else:
        print("\nModel checks passed.\n")

    # ── 4. Run inference on eval_data ────────────────────────────────────────
    eval_dir = os.path.abspath(args.eval_data_dir)
    dataset  = EvalDataset(eval_dir)
    loader   = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == 'cuda'))
    print(f"Found {len(dataset)} eval files in {eval_dir}")

    rows = []   # (filename, predicted_class, domain, confidence)
    with torch.no_grad():
        for waveforms, filenames in tqdm(loader, desc='Evaluating'):
            waveforms = waveforms.to(device)
            B = waveforms.shape[0]

            # All task heads
            all_probs = [_tta_predict(model, waveforms, t, args.tta_n)
                         for t in range(NB_TASKS)]
            stacked = torch.stack(all_probs, dim=0)  # (NB_TASKS, B, C)

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

    # ── 5. Write DCASE submission TSV ────────────────────────────────────────
    total         = len(rows)
    domain_counts = {}
    class_counts  = {}
    for _, cls, dom, _ in rows:
        domain_counts[dom] = domain_counts.get(dom, 0) + 1
        class_counts[cls]  = class_counts.get(cls, 0) + 1

    ckpt_name = os.path.basename(os.path.normpath(args.checkpoint_dir))
    os.makedirs(os.path.dirname(os.path.abspath(args.output_tsv)), exist_ok=True)
    with open(args.output_tsv, 'w') as f:
        f.write(f'# checkpoint : {ckpt_name}\n')
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
