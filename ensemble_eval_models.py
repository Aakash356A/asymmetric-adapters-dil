#!/usr/bin/env python3
"""
Ensemble two DCASE Task 7 evaluation models by averaging their class probabilities
before the entropy-based domain routing step.

Example:
    python ensemble_eval_models.py \
        --checkpoint_dir_a checkpoints/BN_RES_DEEP_CLF_lr1e-3_bs32_ep180_deep_clf_d3_ep180 \
        --checkpoint_dir_b checkpoints/BN_RES_DEEP_CLF_lr1e-3_bs32_ep240_deep_clf_d3_sup02_aug_5 \
        --model_type bn_res_deep_clf \
        --adapter_depths 1,3 \
        --eval_data_dir /data/dcase_2026/audio_classification/dcase2026_task7_data/dil-dcase26-eval \
        --output_file checkpoints/ensemble_eval_predictions.tsv \
        --batch_size 16 --num_workers 4 --cuda
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add code paths so model and config modules are importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASELINE_DIR = os.path.join(_SCRIPT_DIR, 'code', 'baseline')
_UTILS_DIR = os.path.join(_SCRIPT_DIR, 'code', 'utils')
sys.path.insert(0, _BASELINE_DIR)
sys.path.insert(0, _UTILS_DIR)

from evaluate_eval_data import (
    CLASSES,
    TASK_TO_DOMAIN,
    EvalDataset,
    build_model,
    load_checkpoint,
    _tta_predict,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble two Task 7 evaluation models')
    parser.add_argument('--checkpoint_dir_a', required=True)
    parser.add_argument('--checkpoint_dir_b', required=True)
    parser.add_argument('--model_type', required=True)
    parser.add_argument('--adapter_depths', default=None)
    parser.add_argument('--eval_data_dir', default=None)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tta_n', type=int, default=1)
    parser.add_argument('--weights', default='1.0,1.0',
                        help='Comma-separated weights for the two models')
    parser.add_argument('--cuda', action='store_true')
    return parser.parse_args()


def load_model(checkpoint_dir, model_type, adapter_depths, device):
    model = build_model(model_type, 3, adapter_depths)
    load_checkpoint(model, checkpoint_dir, device)
    model.to(device)
    model.eval()
    return model


def evaluate_ensemble(args):
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    print(f'Device: {device}')

    weights = [float(w.strip()) for w in args.weights.split(',') if w.strip()]
    if len(weights) != 2:
        raise ValueError('--weights must contain exactly two comma-separated values')
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError('--weights must sum to a positive value')

    model_a = load_model(args.checkpoint_dir_a, args.model_type, args.adapter_depths, device)
    model_b = load_model(args.checkpoint_dir_b, args.model_type, args.adapter_depths, device)
    print(f'Loaded model A from {args.checkpoint_dir_a}')
    print(f'Loaded model B from {args.checkpoint_dir_b}')

    eval_dir = args.eval_data_dir
    if eval_dir is None:
        eval_dir = os.path.join(_SCRIPT_DIR, 'task7_data', 'eval_data')
    dataset = EvalDataset(eval_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == 'cuda'),
    )
    print(f'Found {len(dataset)} eval files in {eval_dir}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    simple_output_file = args.output_file.replace('.tsv', '_simple.tsv')

    rows = []
    with torch.no_grad():
        for waveforms, filenames in tqdm(loader, desc='Ensembling predictions'):
            waveforms = waveforms.to(device)
            B = waveforms.shape[0]

            all_probs = []
            for task in range(3):
                p_a = _tta_predict(model_a, waveforms, task, args.tta_n)
                p_b = _tta_predict(model_b, waveforms, task, args.tta_n)
                p_avg = (p_a * weights[0] + p_b * weights[1]) / total_weight
                all_probs.append(p_avg)

            stacked = torch.stack(all_probs, dim=0)
            eps = torch.finfo(torch.float32).tiny
            entropy = -torch.sum(stacked * torch.log(stacked + eps), dim=-1)
            _, best_tasks = torch.min(entropy, dim=0)
            sample_idx = torch.arange(B, device=device)
            final_probs = stacked[best_tasks, sample_idx, :]

            pred_indices = torch.argmax(final_probs, dim=1)
            confidences = final_probs[sample_idx, pred_indices]

            for b in range(B):
                fname = filenames[b]
                pred_cls = CLASSES[pred_indices[b].item()]
                domain = TASK_TO_DOMAIN[best_tasks[b].item()]
                confidence = confidences[b].item()
                rows.append((fname, pred_cls, domain, confidence))

    with open(args.output_file, 'w') as fout:
        fout.write(f'# checkpoint_a : {os.path.basename(os.path.normpath(args.checkpoint_dir_a))}\n')
        fout.write(f'# checkpoint_b : {os.path.basename(os.path.normpath(args.checkpoint_dir_b))}\n')
        fout.write(f'# model_type : {args.model_type}\n')
        fout.write(f'# eval_dir   : {eval_dir}\n')
        fout.write(f'# total_files: {len(rows)}\n')
        fout.write('#\n')
        fout.write('# domain_distribution:\n')
        domain_counts = {}
        for _, _, domain, _ in rows:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        for domain in ['D1', 'D2', 'D3']:
            count = domain_counts.get(domain, 0)
            fout.write(f'#   {domain}: {count} ({count / max(len(rows), 1):.1%})\n')
        fout.write('#\n')
        fout.write('# class_distribution:\n')
        class_counts = {}
        for _, cls, _, _ in rows:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        for cls in CLASSES:
            count = class_counts.get(cls, 0)
            if count > 0:
                fout.write(f'#   {cls}: {count}\n')
        fout.write('#\n')
        fout.write('filename\tpredicted_class\tdomain_used\tconfidence\n')
        for fname, cls, domain, conf in rows:
            fout.write(f'{fname}\t{cls}\t{domain}\t{conf:.4f}\n')

    with open(simple_output_file, 'w') as fout:
        for fname, cls, _, _ in rows:
            fout.write(f'{fname}\t{cls}\n')

    print(f'Wrote {len(rows)} ensemble predictions to {args.output_file}')
    print(f'Wrote simple predictions to {simple_output_file}')


if __name__ == '__main__':
    evaluate_ensemble(parse_args())
