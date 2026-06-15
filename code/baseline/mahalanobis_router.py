"""
Mahalanobis Distance-based Domain Router for DCASE-2026 Task-7
==============================================================

Implements a domain-routing system using Mahalanobis distance metrics.

Core idea
---------
Instead of entropy-based routing, compute Mahalanobis distance from each sample's
D1-BN embedding to each domain's distribution (mean and covariance). Route to the
domain with the minimum Mahalanobis distance.

Key components:
1. MahaFeature: 2048-d GAP feature from Conv6 using D1 BN (frozen)
2. Statistics: Compute μ_D2, Σ_D2, μ_D3, Σ_D3 from training data
3. Router with threshold:
   - If min(d_D2, d_D3) > threshold: route to D1 (rejection)
   - Else: route to nearest domain (D2 or D3)

Key steps:
1. Train: Extract D1-BN embeddings for D2 and D3, compute domain statistics
   (mean, covariance)
2. Eval: For each sample, compute Mahalanobis distance to each domain and route
   to the closest one (or reject to D1 if both distances exceed threshold)

Usage
-----
Train the Mahalanobis router (compute domain statistics):
    python3 mahalanobis_router.py train \
        --model_folder /path/to/checkpoint_folder \
        --model_type bn_res --batch_size 32 \
        --num_workers 8 --cuda --router_out /path/mahal_router.pth

Evaluate routed classification on the dev set:
    python3 mahalanobis_router.py eval \
        --model_folder /path/to/checkpoint_folder \
        --model_type bn_res --batch_size 1 --num_workers 8 --cuda \
        --router_in /path/mahal_router.pth --threshold 10.0

Full pipeline (train + eval):
    python3 mahalanobis_router.py full_pipeline \
        --model_folder /path/to/checkpoint_folder \
        --model_type bn_res --batch_size 32 --num_workers 8 --cuda \
        --threshold 10.0
"""

import os
import sys
import argparse
import pickle
import csv

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

import numpy as np
import torch
import librosa
from torch.utils.data import Dataset, ConcatDataset, DataLoader

import config_task7 as config
from config_task7 import (sample_rate, mel_bins, fmin, fmax,
                          window_size, hop_size)

from domain_net import MCnn14
from domain_net_arch import (MCnn14CLF, MCnn14Cos, MCnn14FiLM, MCnn14SE,
                             MCnn14Att, MCnn14SEDeep, MCnn14SEClf, MCnn14Res,
                             MCnn14GN, MCnn14GNSE, MCnn14ResAdil, MCnn14ResClf,
                             MCnn14ResDeep, MCnn14ResDeepClf, set_adapter_depths)
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
    'bn_res_deep': MCnn14ResDeep, 'bn_res_deep_clf': MCnn14ResDeepClf,
    'bn_grn':   MCnn14GRN,
}

# Domains used for routing.  Task index in the model: D1=0, D2=1, D3=2.
_ROUTE_DOMAINS = [['D2'], ['D3']]      # -> router classes [0, 1]
_ROUTE_TASK_OF_CLASS = {0: 1, 1: 2}    # router class -> model task index
_NUM_TASKS = 3                         # D1 + D2 + D3
_TASK_TO_DOMAIN = {0: 'D1', 1: 'D2', 2: 'D3'}
_IDX_TO_CLASS = {v: k for k, v in config.dict_class_labels.items()}


# ---------------------------------------------------------------------------
# Mahalanobis Router Data Structure
# ---------------------------------------------------------------------------

class MahalanobisRouter:
    """
    Stores domain statistics (mean, covariance) for Mahalanobis distance routing.
    
    Attributes:
        means: dict of domain -> (embedding_dim,) numpy array
        covs: dict of domain -> (embedding_dim, embedding_dim) numpy array
        inv_covs: dict of domain -> inverse covariance matrix
        embedding_dim: dimension of embeddings
    """
    
    def __init__(self, embedding_dim=2048, num_domains=2):
        self.embedding_dim = embedding_dim
        self.num_domains = num_domains
        self.means = {}
        self.covs = {}
        self.inv_covs = {}
        self.counts = {}
    
    def add_samples(self, domain_id, embeddings):
        """
        Add embeddings for a domain.
        
        Args:
            domain_id (int): Domain class ID (0 for D2, 1 for D3, etc.)
            embeddings (np.ndarray): (N, embedding_dim) array of embeddings
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        if domain_id not in self.means:
            self.means[domain_id] = np.zeros(self.embedding_dim)
            self.counts[domain_id] = 0
        
        n = embeddings.shape[0]
        # Running mean update
        total = self.counts[domain_id] + n
        self.means[domain_id] = (
            self.counts[domain_id] * self.means[domain_id] + 
            np.sum(embeddings, axis=0)
        ) / total
        self.counts[domain_id] = total
    
    def compute_statistics(self, embeddings_per_domain):
        """
        Compute covariance matrices after collecting all samples.
        
        Args:
            embeddings_per_domain (dict): domain_id -> list of (N, embedding_dim) arrays
        """
        for domain_id, emb_list in embeddings_per_domain.items():
            # Concatenate all embeddings for this domain
            all_emb = np.vstack(emb_list)  # (N, embedding_dim)
            
            # Compute mean
            self.means[domain_id] = np.mean(all_emb, axis=0)
            
            # Compute covariance with regularization (ridge regularization for stability)
            cov = np.cov(all_emb.T)  # (embedding_dim, embedding_dim)
            
            # Add small regularization to diagonal for numerical stability
            reg_term = 1e-4 * np.eye(self.embedding_dim)
            cov_reg = cov + reg_term
            
            self.covs[domain_id] = cov_reg
            try:
                self.inv_covs[domain_id] = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse
                print(f'  [warning] covariance singular for domain {domain_id}, using pseudo-inverse')
                self.inv_covs[domain_id] = np.linalg.pinv(cov_reg)
    
    def mahalanobis_distance(self, embedding, domain_id):
        """
        Compute Mahalanobis distance from embedding to domain.
        
        Args:
            embedding (np.ndarray): (embedding_dim,) vector
            domain_id (int): Domain class ID
        
        Returns:
            float: Mahalanobis distance
        """
        if domain_id not in self.means:
            return float('inf')
        
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()
        
        diff = embedding - self.means[domain_id]
        inv_cov = self.inv_covs.get(domain_id)
        if inv_cov is None:
            return float('inf')
        
        # D_M = sqrt((x - μ)^T Σ^-1 (x - μ))
        mahal_dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
        return mahal_dist
    
    def route(self, embedding, threshold=None):
        """
        Route embedding to nearest domain based on Mahalanobis distance.
        
        Args:
            embedding (np.ndarray or torch.Tensor): (embedding_dim,) vector
            threshold (float, optional): If min distance > threshold, reject to D1
        
        Returns:
            tuple: (domain_id, distances, is_rejected)
                - domain_id: routed domain class ID (or -1 if rejected to D1)
                - distances: dict of domain_id -> Mahalanobis distance
                - is_rejected: True if rejected to D1 (min distance > threshold)
        """
        distances = {}
        for domain_id in range(self.num_domains):
            distances[domain_id] = self.mahalanobis_distance(embedding, domain_id)
        
        min_distance = min(distances.values())
        routed_domain = min(distances, key=distances.get)
        
        # Check if should reject to D1
        is_rejected = False
        if threshold is not None and min_distance > threshold:
            is_rejected = True
            routed_domain = -1  # -1 indicates D1 (rejection)
        
        return routed_domain, distances, is_rejected
    
    def save(self, path):
        """Save router statistics to file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {
            'means': self.means,
            'covs': self.covs,
            'inv_covs': self.inv_covs,
            'counts': self.counts,
            'embedding_dim': self.embedding_dim,
            'num_domains': self.num_domains,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f'  [router] saved -> {path}')
    
    def load(self, path):
        """Load router statistics from file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.means = state['means']
        self.covs = state['covs']
        self.inv_covs = state['inv_covs']
        self.counts = state['counts']
        self.embedding_dim = state['embedding_dim']
        self.num_domains = state['num_domains']
        print(f'  [router] loaded from {path}')


# ---------------------------------------------------------------------------
# Dataset that tags each waveform with a fixed domain target
# ---------------------------------------------------------------------------

class _DomainTagged(Dataset):
    def __init__(self, base, domain_id):
        self.base = base
        self.domain_id = domain_id

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        wav, _, fname = self.base[i]
        return torch.as_tensor(wav, dtype=torch.float32), self.domain_id, fname


class _EvalFolderDataset(Dataset):
    """Dataset for unlabeled eval WAV files."""

    def __init__(self, eval_data_dir):
        self.eval_data_dir = eval_data_dir
        self.files = []
        for root, _, names in os.walk(eval_data_dir):
            for name in names:
                if name.lower().endswith('.wav'):
                    full = os.path.join(root, name)
                    rel = os.path.relpath(full, eval_data_dir)
                    self.files.append((full, rel))
        self.files.sort(key=lambda x: x[1])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        full_path, rel_name = self.files[i]
        audio, _ = librosa.core.load(full_path, sr=config.sample_rate, mono=True)
        if len(audio) < config.clip_samples:
            audio = np.concatenate((audio, np.zeros(config.clip_samples - len(audio), dtype=audio.dtype)))
        waveform = torch.as_tensor(audio, dtype=torch.float32)
        return waveform, rel_name


def _build_router_dataset(df, audio_folder, training):
    """Concatenate D2 and D3 with domain IDs."""
    parts = []
    for cls, dom in enumerate(_ROUTE_DOMAINS):
        sub = df[df['domain'].isin(dom)]
        base = DILDataset(sub, audio_folder, training=training)
        parts.append(_DomainTagged(base, cls))
        print(f'  [router-data] {dom} -> domain_id {cls}  ({len(base)} samples)')
    return ConcatDataset(parts)


# ---------------------------------------------------------------------------
# Backbone helpers
# ---------------------------------------------------------------------------

def _build_model(model_type, device):
    Model = _MODEL_ZOO.get(model_type, MCnn14)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                  config.classes_num_DIL, _NUM_TASKS)
    return model.to(device)


def _parse_adapter_depths(depths_str):
    if not depths_str:
        return None
    parts = [p.strip() for p in depths_str.split(',') if p.strip()]
    if len(parts) < 2:
        raise ValueError('adapter_depths must have at least two values for D2,D3 (e.g., 1,3)')
    vals = [max(0, int(v)) for v in parts]
    # task index: D1=0 fixed, D2=1, D3=2
    return {1: vals[0], 2: vals[1]}


def _maybe_set_adapter_depths(model_type, adapter_depths):
    if model_type in ('bn_res_deep', 'bn_res_deep_clf') and adapter_depths:
        set_adapter_depths(_parse_adapter_depths(adapter_depths))


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
# Train: Collect embeddings and compute statistics
# ---------------------------------------------------------------------------

def train_router(args):
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    d1_ckpt = _find_checkpoint(args.model_folder, domain='D1')

    _maybe_set_adapter_depths(args.model_type, getattr(args, 'adapter_depths', None))
    model = _build_model(args.model_type, device)
    _load_ckpt(model, d1_ckpt, device)
    # Freeze everything; BN_D1 must use fixed running stats.
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    router = MahalanobisRouter(embedding_dim=2048, num_domains=len(_ROUTE_DOMAINS))

    ds = _build_router_dataset(config.df_DIL_dev_train, config.audio_folder_DIL, training=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device == 'cuda'))

    # Streaming accumulators (memory-safe): keeps O(D^2) per domain, not O(N*D)
    emb_dim = router.embedding_dim
    counts = {i: 0 for i in range(len(_ROUTE_DOMAINS))}
    sum_vec = {i: np.zeros((emb_dim,), dtype=np.float64) for i in range(len(_ROUTE_DOMAINS))}
    sum_outer = {i: np.zeros((emb_dim, emb_dim), dtype=np.float64) for i in range(len(_ROUTE_DOMAINS))}

    print('  [training] collecting streaming D1-BN stats...')

    with torch.no_grad():
        for batch_idx, (audio, domain_id, _) in enumerate(loader):
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                print(f'  [training] early stop at {args.max_train_batches} batches')
                break

            audio = audio.float().to(device)
            z = _d1_embedding(model, audio).detach().cpu().numpy().astype(np.float64, copy=False)
            dom = domain_id.detach().cpu().numpy()

            for dom_id in np.unique(dom):
                dom_id = int(dom_id)
                x = z[dom == dom_id]
                if x.shape[0] == 0:
                    continue
                counts[dom_id] += x.shape[0]
                sum_vec[dom_id] += x.sum(axis=0)
                sum_outer[dom_id] += x.T @ x

    print('  [training] finalizing domain statistics...')
    for dom_id in range(len(_ROUTE_DOMAINS)):
        n = counts[dom_id]
        if n == 0:
            raise RuntimeError(f'No samples collected for domain id {dom_id}')

        mu = sum_vec[dom_id] / float(n)
        if n > 1:
            cov = (sum_outer[dom_id] - n * np.outer(mu, mu)) / float(n - 1)
        else:
            cov = np.zeros((emb_dim, emb_dim), dtype=np.float64)

        if args.diag_cov:
            cov = np.diag(np.diag(cov))

        cov_reg = cov + (args.cov_reg * np.eye(emb_dim, dtype=np.float64))
        router.means[dom_id] = mu
        router.covs[dom_id] = cov_reg
        router.counts[dom_id] = n

        if args.diag_cov:
            # Fast inverse for diagonal covariance
            diag = np.diag(cov_reg)
            inv_diag = 1.0 / np.clip(diag, args.cov_reg, None)
            router.inv_covs[dom_id] = np.diag(inv_diag)
        else:
            try:
                router.inv_covs[dom_id] = np.linalg.inv(cov_reg)
            except np.linalg.LinAlgError:
                print(f'  [warning] covariance singular for domain {dom_id}, using pseudo-inverse')
                router.inv_covs[dom_id] = np.linalg.pinv(cov_reg)

    # Print statistics
    for domain_id in range(len(_ROUTE_DOMAINS)):
        mean = router.means[domain_id]
        print(f'  [domain {domain_id}] mean shape: {mean.shape}, norm: {np.linalg.norm(mean):.4f}')

    os.makedirs(os.path.dirname(os.path.abspath(args.router_out)), exist_ok=True)
    router.save(args.router_out)


# ---------------------------------------------------------------------------
# Eval: Route using Mahalanobis distance, then classify with routed head
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_router(args):
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    d3_ckpt = _find_checkpoint(args.model_folder, domain='D3')

    _maybe_set_adapter_depths(args.model_type, getattr(args, 'adapter_depths', None))
    model = _build_model(args.model_type, device)
    _load_ckpt(model, d3_ckpt, device)
    model.eval()

    router = MahalanobisRouter()
    router.load(args.router_in)

    df = config.df_DIL_dev_test
    # Per-domain accuracy + routing accuracy
    cls_correct, cls_total = 0, 0
    route_correct, route_total = 0, 0
    d1_rejections = 0
    per_dom = {}
    per_dom_mahal_dists = {}

    for true_cls, dom in enumerate(_ROUTE_DOMAINS):
        sub = df[df['domain'].isin(dom)]
        ds = DILDataset(sub, config.audio_folder_DIL, training=False)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device == 'cuda'))
        dc, dt = 0, 0
        mahal_dists = []
        
        for audio, label, _ in loader:
            audio = audio.float().to(device)
            gt_class = int(torch.argmax(label, dim=-1).item())

            # --- route using Mahalanobis distance ---
            z = _d1_embedding(model, audio)  # (1, 2048)
            route_pred, distances, is_rejected = router.route(
                z[0].cpu().numpy(), 
                threshold=args.threshold if hasattr(args, 'threshold') else None
            )
            
            mahal_dists.append(distances[true_cls])
            
            # Check routing accuracy
            if is_rejected:
                d1_rejections += 1
                routed_task = 0  # D1 task
            else:
                route_correct += int(route_pred == true_cls)
                routed_task = _ROUTE_TASK_OF_CLASS[int(route_pred)]
            route_total += 1

            # --- classify with routed domain head ---
            logits = model(audio, routed_task)
            pred_class = int(torch.argmax(logits, dim=1).item())
            dc += int(pred_class == gt_class)
            dt += 1

        per_dom[dom[0]] = 100.0 * dc / max(dt, 1)
        per_dom_mahal_dists[dom[0]] = np.mean(mahal_dists)
        cls_correct += dc
        cls_total += dt
        print(f'  [{dom[0]}] routed-classification acc: {per_dom[dom[0]]:.2f}%  '
              f'(mahal_dist: {per_dom_mahal_dists[dom[0]]:.4f})  ({dc}/{dt})')

    avg = float(np.mean(list(per_dom.values())))
    print('=' * 56)
    print(f'  Routing accuracy (D2 vs D3): '
          f'{100*route_correct/max(route_total,1):.2f}%')
    if hasattr(args, 'threshold') and args.threshold is not None:
        print(f'  D1 rejections: {d1_rejections} ({100*d1_rejections/max(route_total,1):.2f}%)')
    for d, a in per_dom.items():
        print(f'  {d} classification acc: {a:.2f}%')
    print(f'  Average (D2,D3): {avg:.3f}%')
    print('=' * 56)


@torch.no_grad()
def predict_eval_folder(args):
    """Predict on unlabeled eval folder and save CSV: filename,class,domain,confidence."""
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'

    d3_ckpt = _find_checkpoint(args.model_folder, domain='D3')
    _maybe_set_adapter_depths(args.model_type, getattr(args, 'adapter_depths', None))

    model = _build_model(args.model_type, device)
    _load_ckpt(model, d3_ckpt, device)
    model.eval()

    router = MahalanobisRouter()
    router.load(args.router_in)

    ds = _EvalFolderDataset(args.eval_data_dir)
    if len(ds) == 0:
        raise RuntimeError(f'No .wav files found in {args.eval_data_dir}')

    # Variable-length waveforms are supported only with batch_size=1 here.
    if args.batch_size != 1:
        print('  [predict] forcing batch_size=1 for variable-length audio safety')
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device == 'cuda'))

    out_path = os.path.abspath(args.output_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    for audio, rel_name in loader:
        audio = audio.to(device)
        z = _d1_embedding(model, audio)
        route_pred, _, is_rejected = router.route(
            z[0].detach().cpu().numpy(),
            threshold=args.threshold,
        )

        if is_rejected:
            routed_task = 0
        else:
            routed_task = _ROUTE_TASK_OF_CLASS[int(route_pred)]

        logits = model(audio, routed_task)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

        pred_idx_i = int(pred_idx.item())
        pred_class = _IDX_TO_CLASS.get(pred_idx_i, str(pred_idx_i))
        domain_used = _TASK_TO_DOMAIN[int(routed_task)]
        confidence = float(conf.item())
        filename = rel_name[0]
        rows.append((filename, pred_class, domain_used, confidence))

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'class', 'domain', 'confidence'])
        for r in rows:
            writer.writerow([r[0], r[1], r[2], f'{r[3]:.6f}'])

    print(f'  [predict] wrote {len(rows)} rows to {out_path}')


# ---------------------------------------------------------------------------
# Full pipeline: train + eval
# ---------------------------------------------------------------------------

def full_pipeline(args):
    """Train router and immediately evaluate on dev-test set."""
    print('=' * 56)
    print('  Starting full Mahalanobis routing pipeline')
    print('=' * 56)
    
    # Create train args
    train_args = argparse.Namespace(
        model_folder=args.model_folder,
        model_type=args.model_type,
        adapter_depths=args.adapter_depths,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cuda=args.cuda,
        router_out=args.router_out,
        cov_reg=args.cov_reg,
        diag_cov=args.diag_cov,
        max_train_batches=args.max_train_batches,
    )
    
    # Train phase
    print('\n[PHASE 1] Training Mahalanobis router...')
    train_router(train_args)
    
    # Create eval args
    eval_args = argparse.Namespace(
        model_folder=args.model_folder,
        model_type=args.model_type,
        adapter_depths=args.adapter_depths,
        batch_size=1,
        num_workers=args.num_workers,
        cuda=args.cuda,
        router_in=args.router_out,
        threshold=args.threshold,
    )
    
    # Eval phase
    print('\n[PHASE 2] Evaluating on dev-test set...')
    eval_router(eval_args)
    
    print('\n' + '=' * 56)
    print('  Pipeline complete!')
    print(f'  Router saved to: {args.router_out}')
    print('=' * 56)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_checkpoint(model_folder, domain='D1'):
    """Find checkpoint in model folder."""
    ckpt_path = os.path.join(model_folder, f'checkpoint_{domain}.pth')
    if os.path.exists(ckpt_path):
        print(f'  [checkpoint] found {domain}: {ckpt_path}')
        return os.path.abspath(ckpt_path)
    
    raise FileNotFoundError(f'checkpoint_{domain}.pth not found in {model_folder}')


def _build_parser():
    p = argparse.ArgumentParser(description='Mahalanobis distance domain router')
    sub = p.add_subparsers(dest='mode', required=True)

    pt = sub.add_parser('train')
    pt.add_argument('--model_folder', type=str, required=True,
                    help='Path to folder containing checkpoint_D1.pth, checkpoint_D2.pth, checkpoint_D3.pth')
    pt.add_argument('--model_type', type=str, default='bn_res')
    pt.add_argument('--adapter_depths', type=str, default=None,
                    help='Per-task adapter depths for bn_res_deep/bn_res_deep_clf, e.g. 1,3 (D2,D3)')
    pt.add_argument('--batch_size', type=int, default=8)
    pt.add_argument('--num_workers', type=int, default=2)
    pt.add_argument('--cuda', action='store_true', default=False)
    pt.add_argument('--router_out', type=str, default='mahal_router.pkl')
    pt.add_argument('--cov_reg', type=float, default=1e-4,
                    help='Diagonal regularization added to covariance')
    pt.add_argument('--diag_cov', action='store_true', default=False,
                    help='Use only diagonal covariance (faster, lower memory)')
    pt.add_argument('--max_train_batches', type=int, default=None,
                    help='Optional cap for train batches (debug/quick run)')

    pe = sub.add_parser('eval')
    pe.add_argument('--model_folder', type=str, required=True,
                    help='Path to folder containing checkpoint_D1.pth, checkpoint_D2.pth, checkpoint_D3.pth')
    pe.add_argument('--model_type', type=str, default='bn_res')
    pe.add_argument('--adapter_depths', type=str, default=None,
                    help='Per-task adapter depths for bn_res_deep/bn_res_deep_clf, e.g. 1,3 (D2,D3)')
    pe.add_argument('--batch_size', type=int, default=1)
    pe.add_argument('--num_workers', type=int, default=2)
    pe.add_argument('--cuda', action='store_true', default=False)
    pe.add_argument('--router_in', type=str, default='mahal_router.pkl')
    pe.add_argument('--threshold', type=float, default=None,
                    help='Mahalanobis distance threshold for D1 rejection (if min(d2,d3) > threshold, route to D1)')

    pp = sub.add_parser('full_pipeline')
    pp.add_argument('--model_folder', type=str, required=True,
                    help='Path to folder containing checkpoint_D1.pth, checkpoint_D2.pth, checkpoint_D3.pth')
    pp.add_argument('--model_type', type=str, default='bn_res')
    pp.add_argument('--adapter_depths', type=str, default=None,
                    help='Per-task adapter depths for bn_res_deep/bn_res_deep_clf, e.g. 1,3 (D2,D3)')
    pp.add_argument('--batch_size', type=int, default=8)
    pp.add_argument('--num_workers', type=int, default=2)
    pp.add_argument('--cuda', action='store_true', default=False)
    pp.add_argument('--router_out', type=str, default='mahal_router.pkl')
    pp.add_argument('--threshold', type=float, default=None,
                    help='Mahalanobis distance threshold for D1 rejection (if min(d2,d3) > threshold, route to D1)')
    pp.add_argument('--cov_reg', type=float, default=1e-4,
                    help='Diagonal regularization added to covariance')
    pp.add_argument('--diag_cov', action='store_true', default=False,
                    help='Use only diagonal covariance (faster, lower memory)')
    pp.add_argument('--max_train_batches', type=int, default=None,
                    help='Optional cap for train batches (debug/quick run)')

    pr = sub.add_parser('predict_eval')
    pr.add_argument('--model_folder', type=str, required=True,
                    help='Path to folder containing checkpoint_D1.pth, checkpoint_D2.pth, checkpoint_D3.pth')
    pr.add_argument('--model_type', type=str, default='bn_res')
    pr.add_argument('--adapter_depths', type=str, default=None,
                    help='Per-task adapter depths for bn_res_deep/bn_res_deep_clf, e.g. 1,3 (D2,D3)')
    pr.add_argument('--eval_data_dir', type=str, required=True,
                    help='Path to eval folder containing unlabeled .wav files')
    pr.add_argument('--router_in', type=str, default='mahal_router.pkl')
    pr.add_argument('--output_csv', type=str, required=True,
                    help='Output CSV path (filename,class,domain,confidence)')
    pr.add_argument('--threshold', type=float, default=None,
                    help='Mahalanobis distance threshold for D1 rejection')
    pr.add_argument('--batch_size', type=int, default=1)
    pr.add_argument('--num_workers', type=int, default=2)
    pr.add_argument('--cuda', action='store_true', default=False)

    return p.parse_args()


if __name__ == '__main__':
    args = _build_parser()

    if args.mode == 'train':
        train_router(args)
    elif args.mode == 'eval':
        eval_router(args)
    elif args.mode == 'full_pipeline':
        full_pipeline(args)
    elif args.mode == 'predict_eval':
        predict_eval_folder(args)
    else:
        raise ValueError(f'Unknown mode: {args.mode}')
