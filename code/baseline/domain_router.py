"""
D1-BN Embedding + Sigmoid Rejection Router   (Angad's idea)
===========================================================

Implements a domain-routing system for the DCASE-2026 Task-7 DIL setup.

Core idea
---------
The released D1 checkpoint defines a *fixed* normalisation coordinate system
(BN_D1).  When D2 / D3 audio is pushed through the FROZEN D1 backbone + BN_D1,
each domain's features shift in a characteristic direction.  A tiny MLP router
learns those directions and predicts the domain.  D1 is handled implicitly as a
rejection class (low router confidence).

Why this is "with our best iteration"
--------------------------------------
Our best model (bn_res) keeps the original D1 path untouched at task index 0
(adapters[0] = Identity, bn0/bnF/bnS[0] = D1 BN).  So ONE model gives us both:
  * the D1-BN embedding for routing:        model.get_embedding(x, task=0)
  * the per-domain classifier for a domain: model(x, task=t)
At inference we route with the router, then classify with the routed domain's
head -- replacing the entropy-min routing in _compute_uncertainity.

Usage
-----
Train the router (only needs the D1 checkpoint):
    python3 domain_router.py train \
        --model_type bn_res --learning_rate 1e-3 --batch_size 32 \
        --epoch 30 --num_workers 8 --cuda \
        --d1_checkpoint /path/checkpoint_D1.pth \
        --router_out /path/router.pth

Evaluate routed classification on the dev set (needs the final D3 checkpoint
of the trained best model + the router):
    python3 domain_router.py eval \
        --model_type bn_res --batch_size 1 --num_workers 8 --cuda \
        --model_checkpoint /path/checkpoint_D3.pth \
        --router_in /path/router.pth --threshold 0.5
"""

import os
import sys
import argparse

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, DataLoader

import config_task7 as config
from config_task7 import (sample_rate, mel_bins, fmin, fmax,
                          window_size, hop_size)

from domain_net import MCnn14
from domain_net_arch import (MCnn14CLF, MCnn14Cos, MCnn14FiLM, MCnn14SE,
                             MCnn14Att, MCnn14SEDeep, MCnn14SEClf, MCnn14Res,
                             MCnn14GN, MCnn14GNSE, MCnn14ResAdil, MCnn14ResClf,
                             MCnn14ResDeep)
from domain_net_dtn import MCnn14DTN, MCnn14Mixed, MCnn14DTN2, MCnn14GRN
from datasetfactory_task7 import DILDatasetInc as DILDataset


_MODEL_ZOO = {
    'bn':          MCnn14,    'dtn':        MCnn14DTN,   'dtn2': MCnn14DTN2,
    'mixed':       MCnn14Mixed,
    'bn_clf':      MCnn14CLF, 'bn_cos':     MCnn14Cos,   'bn_film': MCnn14FiLM,
    'bn_se':       MCnn14SE,  'bn_att':     MCnn14Att,
    'bn_se_deep':  MCnn14SEDeep, 'bn_se_clf': MCnn14SEClf, 'bn_res': MCnn14Res,
    'bn_gn':       MCnn14GN,  'bn_gn_se':   MCnn14GNSE,
    'bn_res_adil': MCnn14ResAdil, 'bn_res_clf': MCnn14ResClf,
    'bn_res_deep': MCnn14ResDeep, 'bn_grn':   MCnn14GRN,
}

# Domains used for routing.  Task index in the model: D1=0, D2=1, D3=2.
_ROUTE_DOMAINS = [['D2'], ['D3']]      # -> router classes [0, 1]
_ROUTE_TASK_OF_CLASS = {0: 1, 1: 2}    # router class -> model task index
_NUM_TASKS = 3                         # D1 + D2 + D3


# ---------------------------------------------------------------------------
# Router head
# ---------------------------------------------------------------------------

class DomainRouter(nn.Module):
    """Linear(2048->128) -> ReLU -> Linear(128->2). Outputs logits (use BCE)."""

    def __init__(self, in_dim=2048, hidden=128, num_domains=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, z):
        return self.net(z)


# ---------------------------------------------------------------------------
# Dataset that tags each waveform with a fixed domain target vector
# ---------------------------------------------------------------------------

class _DomainTagged(Dataset):
    def __init__(self, base, target_vec):
        self.base = base
        self.target = torch.tensor(target_vec, dtype=torch.float32)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        wav, _, fname = self.base[i]
        return torch.as_tensor(wav, dtype=torch.float32), self.target, fname


def _build_router_dataset(df, training):
    """Concatenate D2 and D3 with router targets [1,0] / [0,1]."""
    parts = []
    for cls, dom in enumerate(_ROUTE_DOMAINS):
        sub = df[df['domain'].isin(dom)]
        base = DILDataset(sub, config.audio_folder_DIL, training=training)
        tgt = [0.0] * len(_ROUTE_DOMAINS)
        tgt[cls] = 1.0
        parts.append(_DomainTagged(base, tgt))
        print(f'  [router-data] {dom} -> target {tgt}  ({len(base)} samples)')
    return ConcatDataset(parts)


# ---------------------------------------------------------------------------
# Backbone helpers
# ---------------------------------------------------------------------------

def _build_model(model_type, device):
    Model = _MODEL_ZOO.get(model_type, MCnn14)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                  config.classes_num_DIL, _NUM_TASKS)
    return model.to(device)


def _load_ckpt(model, path, device):
    inc = model.load_state_dict(
        torch.load(path, map_location=torch.device(device)), strict=False)
    print(f'  [ckpt] loaded {os.path.basename(path)}  '
          f'(missing={len(inc.missing_keys)}, unexpected={len(inc.unexpected_keys)})')


@torch.no_grad()
def _d1_embedding(model, audio):
    """Frozen D1-BN embedding (task index 0).  model must be in eval mode."""
    return model.get_embedding(audio, task=0)      # (B, 2048)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_router(args):
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    model = _build_model(args.model_type, device)
    _load_ckpt(model, args.d1_checkpoint, device)
    # Freeze everything; BN_D1 must use fixed running stats.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    router = DomainRouter(in_dim=2048, hidden=128,
                          num_domains=len(_ROUTE_DOMAINS)).to(device)

    ds = _build_router_dataset(config.df_DIL_dev_train, training=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers,
                        pin_memory=(device == 'cuda'))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(router.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epoch + 1):
        router.train()
        sum_loss, correct, total = 0.0, 0, 0
        for audio, target, _ in loader:
            audio = audio.float().to(device)
            target = target.float().to(device)

            z = _d1_embedding(model, audio)        # no grad, frozen backbone
            logits = router(z)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.item()
            pred = logits.argmax(dim=1)
            gt = target.argmax(dim=1)
            correct += (pred == gt).sum().item()
            total += gt.numel()

        print(f'==> router epoch {epoch:3d} | loss {sum_loss/len(loader):.4f} '
              f'| D2/D3 train acc {100*correct/max(total,1):.2f}%')

    os.makedirs(os.path.dirname(os.path.abspath(args.router_out)), exist_ok=True)
    torch.save(router.state_dict(), args.router_out)
    print(f'  [router] saved -> {args.router_out}')


# ---------------------------------------------------------------------------
# Eval: route, then classify with the routed domain head
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_router(args):
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    model = _build_model(args.model_type, device)
    _load_ckpt(model, args.model_checkpoint, device)
    model.eval()

    router = DomainRouter(in_dim=2048, hidden=128,
                          num_domains=len(_ROUTE_DOMAINS)).to(device)
    router.load_state_dict(torch.load(args.router_in, map_location=device))
    router.eval()

    df = config.df_DIL_dev_test
    # Per-domain accuracy + routing accuracy
    cls_correct, cls_total = 0, 0
    route_correct, route_total = 0, 0
    per_dom = {}

    for true_cls, dom in enumerate(_ROUTE_DOMAINS):
        sub = df[df['domain'].isin(dom)]
        ds = DILDataset(sub, config.audio_folder_DIL, training=False)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device == 'cuda'))
        dc, dt = 0, 0
        for audio, label, _ in loader:
            audio = audio.float().to(device)
            gt_class = int(torch.argmax(label, dim=-1).item())

            # --- route ---
            z = _d1_embedding(model, audio)
            probs = torch.sigmoid(router(z))[0]        # (num_domains,)
            conf, route_pred = torch.max(probs, dim=0)
            if conf.item() < args.threshold:
                # Rejection -> D1 path (task 0). In dev eval there is no D1,
                # so this is effectively "unsure"; fall back to argmax domain.
                routed_task = _ROUTE_TASK_OF_CLASS[int(route_pred.item())]
            else:
                routed_task = _ROUTE_TASK_OF_CLASS[int(route_pred.item())]

            route_correct += int(int(route_pred.item()) == true_cls)
            route_total += 1

            # --- classify with routed domain head ---
            logits = model(audio, routed_task)
            pred_class = int(torch.argmax(logits, dim=1).item())
            dc += int(pred_class == gt_class)
            dt += 1

        per_dom[dom[0]] = 100.0 * dc / max(dt, 1)
        cls_correct += dc
        cls_total += dt
        print(f'  [{dom[0]}] routed-classification acc: {per_dom[dom[0]]:.2f}%  '
              f'({dc}/{dt})')

    avg = float(np.mean(list(per_dom.values())))
    print('=' * 56)
    print(f'  Routing accuracy (D2 vs D3): '
          f'{100*route_correct/max(route_total,1):.2f}%')
    for d, a in per_dom.items():
        print(f'  {d} classification acc: {a:.2f}%')
    print(f'  Average (D2,D3): {avg:.3f}%')
    print('=' * 56)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(description='D1-BN domain router (Angad idea)')
    sub = p.add_subparsers(dest='mode', required=True)

    pt = sub.add_parser('train')
    pt.add_argument('--model_type', type=str, default='bn_res')
    pt.add_argument('--learning_rate', type=float, default=1e-3)
    pt.add_argument('--batch_size', type=int, default=32)
    pt.add_argument('--epoch', type=int, default=30)
    pt.add_argument('--num_workers', type=int, default=8)
    pt.add_argument('--cuda', action='store_true', default=False)
    pt.add_argument('--d1_checkpoint', type=str, required=True)
    pt.add_argument('--router_out', type=str, required=True)

    pe = sub.add_parser('eval')
    pe.add_argument('--model_type', type=str, default='bn_res')
    pe.add_argument('--batch_size', type=int, default=1)
    pe.add_argument('--num_workers', type=int, default=8)
    pe.add_argument('--cuda', action='store_true', default=False)
    pe.add_argument('--model_checkpoint', type=str, required=True,
                    help='Final trained checkpoint of the best model (D3).')
    pe.add_argument('--router_in', type=str, required=True)
    pe.add_argument('--threshold', type=float, default=0.5)
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    if args.mode == 'train':
        train_router(args)
    elif args.mode == 'eval':
        eval_router(args)
    else:
        raise SystemExit('unknown mode')
