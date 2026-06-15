#!/usr/bin/env python3
"""
Domain router trained on features from the model's global-average-pooled
convolutional representation.

This script loads the D1 / D2 / D3 checkpoints, extracts GAP features from the
final convolution block of each domain-specific model, trains a small MLP on the
labeled development set, and then uses the trained MLP to route each evaluation
sample to a domain model before classifying it.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASELINE_DIR = os.path.join(_SCRIPT_DIR, 'code', 'baseline')
_UTILS_DIR = os.path.join(_SCRIPT_DIR, 'code', 'utils')
sys.path.insert(0, _BASELINE_DIR)
sys.path.insert(0, _UTILS_DIR)

import config_task7 as config
from evaluate_eval_data import CLASSES, EvalDataset, build_model


TASK_TO_DOMAIN = {0: 'D1', 1: 'D2', 2: 'D3'}


def resolve_data_root() -> str:
    candidates = [
        os.path.join(_SCRIPT_DIR, 'task7_data'),
        os.path.join(os.path.dirname(_SCRIPT_DIR), 'dcase2026_task7_data'),
        '/data/dcase_2026/audio_classification/dcase2026_task7_data',
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, 'evaluation_setup', 'development_train.txt')):
            return candidate
    raise FileNotFoundError('Could not locate the task-7 data root')


DATA_ROOT = resolve_data_root()
AUDIO_ROOT = os.path.join(DATA_ROOT, 'audio')
DEV_TRAIN_PATH = os.path.join(DATA_ROOT, 'evaluation_setup', 'development_train.txt')
DEV_TEST_PATH = os.path.join(DATA_ROOT, 'evaluation_setup', 'development_test.txt')


class GAPDomainDataset(Dataset):
    def __init__(self, df, audio_folder, max_samples=None, return_class_label=False):
        self.samples = []
        self.return_class_label = return_class_label
        for _, row in df.iterrows():
            if max_samples is not None and len(self.samples) >= max_samples:
                break
            domain = str(row['domain']).upper()
            if domain not in {'D1', 'D2', 'D3'}:
                continue
            domain_label = {'D1': 0, 'D2': 1, 'D3': 2}[domain]
            class_label = int(row['new_target']) if 'new_target' in row else int(row['target'])
            path = os.path.join(audio_folder, row['filename'])
            self.samples.append((path, int(domain_label), int(class_label), row['filename']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, domain_label, class_label, fname = self.samples[idx]
        audio, _ = librosa.core.load(path, sr=config.sample_rate, mono=True)
        if len(audio) < config.clip_samples:
            audio = np.concatenate([audio, np.zeros(config.clip_samples - len(audio))])
        else:
            audio = audio[:config.clip_samples]
        if self.return_class_label:
            return torch.from_numpy(audio).float(), torch.tensor(domain_label, dtype=torch.long), torch.tensor(class_label, dtype=torch.long)
        return torch.from_numpy(audio).float(), torch.tensor(domain_label, dtype=torch.long)


class GAPMLPRouter(nn.Module):
    def __init__(self, input_dim=6144, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),  # [p(D2), p(D3)]
        )

    def forward(self, x):
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser(description='GAP-based MLP router using trained D1/D2/D3 checkpoints')
    parser.add_argument('--checkpoint_dir_d1', default=None)
    parser.add_argument('--checkpoint_dir_d2', default=None)
    parser.add_argument('--checkpoint_dir_d3', default=None)
    parser.add_argument('--model_type', default='bn_res_deep_clf')
    parser.add_argument('--adapter_depths', default='1,3')
    parser.add_argument('--eval_data_dir', default=None)
    parser.add_argument('--eval_on_dev_test', action='store_true', help='Evaluate on the development-test split instead of the unlabeled eval set')
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit on eval files; leave unset to use all samples')
    parser.add_argument('--train_limit', type=int, default=None, help='Optional limit on training samples; leave unset to use all samples')
    parser.add_argument('--epochs', type=int, default=4, help='Router training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for router training')
    parser.add_argument('--routing_threshold', type=float, default=0.5, help='Probability threshold for routing to D2/D3; below it routes to D1')
    parser.add_argument('--router_checkpoint', default=None, help='Optional router checkpoint path')
    return parser.parse_args()


def load_checkpoint(model, checkpoint_dir: str, checkpoint_name: str, device: str):
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    try:
        state = torch.load(ckpt_path, map_location=device)
    except Exception as exc:
        raise RuntimeError(f'Failed to load checkpoint {ckpt_path}: {exc}') from exc
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Loaded {checkpoint_name} into model with missing={len(missing)}, unexpected={len(unexpected)}')


def build_domain_model(model_type: str, num_tasks: int, checkpoint_dir: str, checkpoint_name: str, adapter_depths: str, device: str):
    model = build_model(model_type, num_tasks, adapter_depths)
    try:
        load_checkpoint(model, checkpoint_dir, checkpoint_name, device)
    except RuntimeError as exc:
        fallback_dir = '/data/dcase_2026/audio_classification/dcase2026_task7_baseline/pretrained_models'
        if os.path.abspath(checkpoint_dir) != os.path.abspath(fallback_dir):
            print(f'Checkpoint {checkpoint_name} failed in {checkpoint_dir}; retrying with fallback {fallback_dir}')
            load_checkpoint(model, fallback_dir, checkpoint_name, device)
        else:
            raise
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_gap_features(model, waveforms, device, task_idx: int):
    model.eval()
    captured = []

    def _hook(module, inputs, output):
        captured.append(output)

    handle = model.conv_block6.register_forward_hook(_hook)
    try:
        _ = model(waveforms, task_idx)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError('Failed to capture GAP features from the final conv block')

    feat = captured[-1]
    gap = feat.mean(dim=(2, 3)).float()
    return gap.detach().cpu()


@torch.no_grad()
def extract_combined_gap_features(models: Dict[int, torch.nn.Module], waveforms, device):
    feats = []
    for task_idx in [0, 1, 2]:
        feat = extract_gap_features(models[task_idx], waveforms.to(device), device, task_idx)
        feats.append(feat)
    return torch.cat(feats, dim=1)


@torch.no_grad()
def build_router_feature_tensors(models, dataset, device, batch_size, num_workers, desc):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
    )

    all_features = []
    all_domains = []
    all_classes = []
    for batch in tqdm(loader, desc=desc):
        if len(batch) == 3:
            waveforms, domain_labels, class_labels = batch
            all_classes.append(class_labels.long())
        else:
            waveforms, domain_labels = batch
        features = extract_combined_gap_features(models, waveforms.to(device), device)
        all_features.append(features.float())
        all_domains.append(domain_labels.long())

    feature_tensor = torch.cat(all_features, dim=0)
    domain_tensor = torch.cat(all_domains, dim=0)
    class_tensor = torch.cat(all_classes, dim=0) if all_classes else None
    return feature_tensor, domain_tensor, class_tensor


def train_router(router, train_features, train_labels, device, epochs, lr, batch_size):
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f'Training router on {len(train_dataset)} labeled domain samples')

    optimizer = torch.optim.Adam(router.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        router.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for features, labels in tqdm(train_loader, desc=f'Router epoch {epoch + 1}/{epochs}', leave=False):
            features = features.to(device)
            labels = labels.to(device)
            logits = router(features)

            targets = torch.zeros((labels.size(0), 2), device=device)
            targets[labels == 1, 0] = 1.0
            targets[labels == 2, 1] = 1.0
            loss = criterion(logits, targets.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits)
            pred_domains = []
            for b in range(labels.size(0)):
                p2, p3 = probs[b, 0].item(), probs[b, 1].item()
                if p2 <= 0.5 and p3 <= 0.5:
                    pred_domains.append(0)
                elif p2 > p3:
                    pred_domains.append(1)
                else:
                    pred_domains.append(2)
            pred_domains = torch.tensor(pred_domains, device=device)
            correct += (pred_domains == labels).sum().item()
            total += len(labels)

        print(f'  epoch {epoch + 1:02d} | loss={total_loss / max(total, 1):.4f} | acc={100 * correct / max(total, 1):.2f}%')

    router.eval()
    return router


def save_router(router, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(router.state_dict(), path)


def load_router(router, path, device):
    state = torch.load(path, map_location=device)
    router.load_state_dict(state)
    router.eval()
    return router


def build_eval_dataset(args, data_root: str):
    if args.eval_on_dev_test:
        df = pd.read_csv(DEV_TEST_PATH, sep='\t', names=['filename', 'target', 'domain', 'new_target'])
        return df, os.path.join(data_root, 'audio')

    eval_dir = args.eval_data_dir if args.eval_data_dir is not None else os.path.join(data_root, 'dil-dcase26-eval')
    return None, eval_dir


def evaluate(args):
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    print(f'Device: {device}')

    checkpoint_dir_d1 = args.checkpoint_dir_d1 or '/data/dcase_2026/audio_classification/dcase2026_task7_baseline/pretrained_models'
    checkpoint_dir_d2 = args.checkpoint_dir_d2 or '/data/dcase_2026/audio_classification/dcase2026_task7_baseline/pretrained_models'
    checkpoint_dir_d3 = args.checkpoint_dir_d3 or '/data/dcase_2026/audio_classification/dcase2026_task7_baseline/pretrained_models'

    model_d1 = build_domain_model(args.model_type, 1, checkpoint_dir_d1, 'checkpoint_D1.pth', args.adapter_depths, device)
    model_d2 = build_domain_model(args.model_type, 2, checkpoint_dir_d2, 'checkpoint_D2.pth', args.adapter_depths, device)
    model_d3 = build_domain_model(args.model_type, 3, checkpoint_dir_d3, 'checkpoint_D3.pth', args.adapter_depths, device)
    models = {0: model_d1, 1: model_d2, 2: model_d3}

    eval_df, eval_source = build_eval_dataset(args, DATA_ROOT)
    if eval_df is not None:
        eval_dataset = GAPDomainDataset(eval_df, AUDIO_ROOT, max_samples=args.limit, return_class_label=True)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == 'cuda'))
        print(f'Found {len(eval_dataset)} development-test files in {DEV_TEST_PATH}')
    else:
        eval_dir = eval_source
        eval_dataset = EvalDataset(eval_dir)
        if args.limit is not None:
            eval_dataset.files = eval_dataset.files[:args.limit]
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == 'cuda'))
        print(f'Found {len(eval_dataset)} eval files in {eval_dir}')

    router = GAPMLPRouter(input_dim=6144, hidden_dim=256).to(device)
    if args.router_checkpoint and os.path.isfile(args.router_checkpoint):
        router = load_router(router, args.router_checkpoint, device)
        print(f'Loaded router from {args.router_checkpoint}')
    else:
        train_df = pd.read_csv(DEV_TRAIN_PATH, sep='\t', names=['filename', 'target', 'domain', 'new_target'])
        train_dataset = GAPDomainDataset(train_df, AUDIO_ROOT, max_samples=args.train_limit)
        train_features, train_domains, _ = build_router_feature_tensors(
            models,
            train_dataset,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            desc='Extracting train GAP features',
        )
        router = train_router(
            router,
            train_features,
            train_domains,
            device,
            args.epochs,
            args.lr,
            args.batch_size,
        )
        if args.router_checkpoint:
            save_router(router, args.router_checkpoint)
            print(f'Saved router to {args.router_checkpoint}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    rows = []
    true_domains = []
    true_classes = []
    pred_domains = []
    pred_classes = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc='Routing with GAP MLP'):
            if eval_df is not None:
                waveforms, domain_labels, class_labels = batch
                waveforms = waveforms.to(device)
                domain_labels = domain_labels.to(device)
                class_labels = class_labels.to(device)
                features = extract_combined_gap_features(models, waveforms, device).to(device)
                router_logits = router(features)
                router_probs = torch.sigmoid(router_logits)
                domain_idx = []
                for b in range(waveforms.shape[0]):
                    p2 = router_probs[b, 0].item()
                    p3 = router_probs[b, 1].item()
                    if p2 <= args.routing_threshold and p3 <= args.routing_threshold:
                        task = 0
                    elif p2 > p3:
                        task = 1
                    else:
                        task = 2
                    domain_idx.append(task)

                preds = [0] * waveforms.shape[0]
                confidences = [0.0] * waveforms.shape[0]
                for task in [0, 1, 2]:
                    selected = [i for i, d in enumerate(domain_idx) if d == task]
                    if not selected:
                        continue
                    selected_tensor = torch.tensor(selected, device=waveforms.device, dtype=torch.long)
                    logits = models[task](waveforms.index_select(0, selected_tensor), task)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred = probs.max(dim=1)
                    for j, sample_idx in enumerate(selected):
                        preds[sample_idx] = int(pred[j].item())
                        confidences[sample_idx] = float(conf[j].item())

                true_domains.extend(int(x.item()) for x in domain_labels)
                true_classes.extend(int(x.item()) for x in class_labels)
                pred_domains.extend(int(domain_idx[b]) for b in range(len(domain_idx)))
                pred_classes.extend(int(preds[b]) for b in range(len(preds)))
                for b in range(len(preds)):
                    rows.append((f'devtest_{b}', CLASSES[preds[b]], TASK_TO_DOMAIN[int(domain_idx[b])], confidences[b]))
            else:
                waveforms, filenames = batch
                waveforms = waveforms.to(device)
                features = extract_combined_gap_features(models, waveforms, device).to(device)
                router_logits = router(features)
                router_probs = torch.sigmoid(router_logits)
                domain_idx = []
                for b in range(waveforms.shape[0]):
                    p2 = router_probs[b, 0].item()
                    p3 = router_probs[b, 1].item()
                    if p2 <= args.routing_threshold and p3 <= args.routing_threshold:
                        task = 0
                    elif p2 > p3:
                        task = 1
                    else:
                        task = 2
                    domain_idx.append(task)

                preds = [0] * waveforms.shape[0]
                confidences = [0.0] * waveforms.shape[0]
                for task in [0, 1, 2]:
                    selected = [i for i, d in enumerate(domain_idx) if d == task]
                    if not selected:
                        continue
                    selected_tensor = torch.tensor(selected, device=waveforms.device, dtype=torch.long)
                    logits = models[task](waveforms.index_select(0, selected_tensor), task)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred = probs.max(dim=1)
                    for j, sample_idx in enumerate(selected):
                        preds[sample_idx] = int(pred[j].item())
                        confidences[sample_idx] = float(conf[j].item())

                for b, fname in enumerate(filenames):
                    rows.append((fname, CLASSES[preds[b]], TASK_TO_DOMAIN[int(domain_idx[b])], confidences[b]))

    with open(args.output_file, 'w') as fout:
        fout.write('filename\tpredicted_class\tdomain_used\tconfidence\n')
        for fname, cls, domain, conf in rows:
            fout.write(f'{fname}\t{cls}\t{domain}\t{conf:.4f}\n')

    domain_counts = {}
    class_counts = {}
    domain_class_counts = {}
    for _, cls, domain, _ in rows:
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        class_counts[cls] = class_counts.get(cls, 0) + 1
        domain_class_counts.setdefault(domain, {})
        domain_class_counts[domain][cls] = domain_class_counts[domain].get(cls, 0) + 1

    if eval_df is not None:
        true_domains = torch.tensor(true_domains, dtype=torch.long)
        pred_domains = torch.tensor(pred_domains, dtype=torch.long)
        true_classes = torch.tensor(true_classes, dtype=torch.long)
        pred_classes = torch.tensor(pred_classes, dtype=torch.long)

        print('Development-test performance (baseline-style):')
        domain_accuracies = []
        for domain_idx, domain_name in [(1, 'D2'), (2, 'D3')]:
            mask = true_domains == domain_idx
            if not mask.any():
                continue
            domain_true_classes = true_classes[mask]
            domain_pred_classes = pred_classes[mask]
            domain_acc = 100 * (domain_pred_classes == domain_true_classes).float().mean().item()
            domain_accuracies.append(domain_acc)
            print(f'  seen domain: [{domain_name}] and its accuracy: {domain_acc:.2f}')

            class_labels = domain_true_classes.cpu().numpy()
            pred_labels = domain_pred_classes.cpu().numpy()
            if len(np.unique(class_labels)) > 0:
                cm = metrics.confusion_matrix(class_labels, pred_labels, labels=list(range(len(CLASSES))))
                row_sums = cm.sum(axis=1)
                class_acc = np.divide(
                    cm.diagonal(),
                    row_sums,
                    out=np.zeros(len(CLASSES), dtype=float),
                    where=row_sums > 0,
                )
                active = [i for i in range(len(class_acc)) if row_sums[i] > 0]
                for i in active:
                    print(f'    {CLASSES[i]}: {np.around(class_acc[i] * 100, decimals=1)}%')

        if domain_accuracies:
            print(f'Average Accuracy: {np.mean(domain_accuracies):.2f}')

    print(f'Wrote {len(rows)} predictions -> {args.output_file}')
    print('Domain summary:')
    for domain in ['D1', 'D2', 'D3']:
        count = domain_counts.get(domain, 0)
        print(f'  {domain}: {count} ({100 * count / max(len(rows), 1):.1f}%)')

    print('Class summary:')
    for cls in sorted(class_counts, key=class_counts.get, reverse=True):
        print(f'  {cls}: {class_counts[cls]}')

    print('Domain x class summary:')
    for domain in ['D1', 'D2', 'D3']:
        print(f'  {domain}:')
        domain_counts_for_domain = domain_class_counts.get(domain, {})
        for cls in sorted(domain_counts_for_domain, key=domain_counts_for_domain.get, reverse=True):
            print(f'    {cls}: {domain_counts_for_domain[cls]}')


if __name__ == '__main__':
    evaluate(parse_args())
