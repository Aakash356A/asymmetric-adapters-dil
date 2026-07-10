"""
Train a STANDALONE single-domain sanity-check classifier on ONE TAU2019
domain (a single city or pooled city group), independent of the
domain-incremental pipeline.

This answers "how learnable is this domain's data alone?", with the whole
backbone + BN + classifier fully fine-tuned from the PANNs init (no frozen
shared backbone, no per-domain adapters/heads, no incremental sequencing) --
useful as a reference ceiling before running the DIL protocol
(train_d1_base.py + baseline_DIL_task7.py train).

The manifest built by prepare_tau2019.py already labels rows 'D1'..'D4'
(or however many domains you built); this script trains on ANY ONE of them
in isolation, always as a single-task (nb_tasks=1, task=0) model, so the
--domain argument just selects which rows to use.

Usage
-----
    python train_domain_standalone.py \
        --domain D2 \
        --checkpoint_dir checkpoints/TAU_standalone_D2_lisbon \
        --panns_ckpt Cnn14_mAP=0.431.pth \
        --epochs 80 --batch_size 32 --lr 1e-3 --num_workers 8 --cuda

Add --resume to continue training from an existing checkpoint in
--checkpoint_dir. All stdout is appended to <checkpoint_dir>/train_<domain>.log.
"""
import argparse
import datetime
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
os.environ.setdefault('DCASE_DATASET', 'tau')
sys.path.insert(0, os.path.join(_REPO, 'code', 'utils'))
sys.path.insert(0, os.path.join(_REPO, 'code', 'baseline'))

import config_task7 as config          # real config, TAU mode via DCASE_DATASET
from domain_net_arch import MCnn14ResDeepClf
from datasetfactory_task7 import DILDatasetInc as DILDataset


class _Tee:
    """Duplicate writes to multiple streams (e.g. console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def map_panns_to_backbone(model, panns_path):
    """Copy PANNs Cnn14 conv/BN weights into the shared backbone + BN[0]."""
    ckpt = torch.load(panns_path, map_location='cpu')
    src = ckpt.get('model', ckpt)
    tgt = model.state_dict()
    copied = 0
    remap = {}
    for suf in ('weight', 'bias', 'running_mean', 'running_var',
                'num_batches_tracked'):
        remap[f'bn0.{suf}'] = f'bn0.0.{suf}'
    for n in range(1, 7):
        remap[f'conv_block{n}.conv1.weight'] = f'conv_block{n}.conv1.weight'
        remap[f'conv_block{n}.conv2.weight'] = f'conv_block{n}.conv2.weight'
        for suf in ('weight', 'bias', 'running_mean', 'running_var',
                    'num_batches_tracked'):
            remap[f'conv_block{n}.bn1.{suf}'] = f'conv_block{n}.bnF.0.{suf}'
            remap[f'conv_block{n}.bn2.{suf}'] = f'conv_block{n}.bnS.0.{suf}'
    for s_key, t_key in remap.items():
        if s_key in src and t_key in tgt and src[s_key].shape == tgt[t_key].shape:
            tgt[t_key] = src[s_key].clone()
            copied += 1
    model.load_state_dict(tgt, strict=False)
    print(f'  [PANNs] initialised {copied} backbone tensors from {os.path.basename(panns_path)}')


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for waveform, target, _ in loader:
        waveform = waveform.float().to(device)
        target = target.float().to(device)
        logits = model(waveform, task=0)
        pred = logits.argmax(1)
        gt = target.argmax(1)
        correct += (pred == gt).sum().item()
        total += gt.numel()
    return 100.0 * correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', required=True,
                    help='Manifest domain to train on in isolation, e.g. D1/D2/D3/D4.')
    ap.add_argument('--checkpoint_dir', required=True)
    ap.add_argument('--panns_ckpt', default=None)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--cuda', action='store_true')
    ap.add_argument('--resume', action='store_true',
                    help='continue training from an existing checkpoint in '
                         '--checkpoint_dir instead of starting fresh')
    args = ap.parse_args()

    device = ('cuda' if (args.cuda and torch.cuda.is_available())
              else 'mps' if torch.backends.mps.is_available() else 'cpu')

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f'checkpoint_{args.domain}.pth')

    log_file = open(os.path.join(args.checkpoint_dir, f'train_{args.domain}.log'), 'a')
    sys.stdout = _Tee(sys.stdout, log_file)
    print(f'\n===== run started {datetime.datetime.now().isoformat()} =====')
    print('args:', vars(args))
    print('device:', device)

    model = MCnn14ResDeepClf(
        config.sample_rate, config.window_size, config.hop_size,
        config.mel_bins, config.fmin, config.fmax,
        config.classes_num_DIL, nb_tasks=1)

    resuming = args.resume and os.path.isfile(ckpt_path)
    if args.panns_ckpt and not resuming:
        map_panns_to_backbone(model, args.panns_ckpt)

    model.to(device)

    if resuming:
        print(f'--resume: loading existing checkpoint from {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    elif args.resume:
        print(f'--resume set but no checkpoint found at {ckpt_path}; starting from scratch')

    train_df = config.df_DIL_dev_train[config.df_DIL_dev_train['domain'] == args.domain]
    test_df = config.df_DIL_dev_test[config.df_DIL_dev_test['domain'] == args.domain]
    if len(train_df) == 0:
        raise RuntimeError(f'no {args.domain} rows found — run prepare_tau2019.py first and '
                           'check TAU_DATA_DIR / PYTHONPATH.')
    print(f'{args.domain} train={len(train_df)} files, test={len(test_df)} files')

    train_ds = DILDataset(train_df, config.audio_folder_DIL, training=True)
    test_ds = DILDataset(test_df, config.audio_folder_DIL, training=False)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == 'cuda'))
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device == 'cuda'))

    # Train the full task-0 path (backbone conv + BN[0] + fc) -- a plain,
    # fully fine-tuned single-domain classifier.
    params = []
    for name, p in model.named_parameters():
        if ('conv_block' in name and ('conv1' in name or 'conv2' in name)) \
                or name.startswith('bn0.0.') \
                or '.bnF.0.' in name or '.bnS.0.' in name \
                or name.startswith('fc.'):
            p.requires_grad_(True)
            params.append(p)
        else:
            p.requires_grad_(False)
    print(f'trainable tensors: {len(params)}')

    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-7)

    best_acc = evaluate(model, test_loader, device) if resuming else -1.0
    if resuming:
        print(f'  resumed model {args.domain} test acc: {best_acc:.2f}%')
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for waveform, target, _ in train_loader:
            waveform = waveform.float().to(device)
            target = target.float().to(device)
            logits = model(waveform, task=0)
            loss = F.cross_entropy(logits, target.argmax(1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        acc = evaluate(model, test_loader, device)
        print(f'epoch {epoch:3d}  loss {running/len(train_loader):.4f}  {args.domain} test acc {acc:.2f}%')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), ckpt_path)
            print(f'  -> saved best {args.domain} standalone ({best_acc:.2f}%) to {ckpt_path}')

    print(f'\nDone. Best {args.domain} test acc {best_acc:.2f}%. Checkpoint: {ckpt_path}')


if __name__ == '__main__':
    main()
