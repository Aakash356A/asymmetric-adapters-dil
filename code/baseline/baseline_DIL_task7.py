import pandas as pd

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import config_task7 as config

import torch.nn.functional as F

from config_task7 import (sample_rate, mel_bins, fmin, fmax, window_size,
                    hop_size)

import torch.optim as optim

from domain_net import MCnn14
from domain_net_dtn import MCnn14DTN, MCnn14Mixed, MCnn14DTN2, MCnn14GRN
from domain_net_arch import (MCnn14CLF, MCnn14Cos, MCnn14FiLM,
                             MCnn14SE, MCnn14Att,
                             MCnn14SEDeep, MCnn14SEClf, MCnn14Res,
                             MCnn14GN, MCnn14GNSE,
                             MCnn14ResAdil, MCnn14ResClf,
                             MCnn14ResDeep, set_adapter_depths)

from datasetfactory_task7 import DILDatasetInc as DILDataset
from datasetfactory_task7_augment import DILDatasetInc as DILDatasetAug
from torch.utils.data import WeightedRandomSampler
from sklearn import metrics
from tqdm import tqdm
from utilities import *
import time

timestr = time.strftime("%Y%m%d-%H%M%S")


# ============================================================================
# Boundary-Aware Regularization  (adapted from PACE, Li et al. 2026)
#
# For each training sample we generate N_p time-frequency masked versions
# ("boundary probes") and apply a push-pull hinge loss:
#
#   L_reg = max(0, δ + pull_term − push_term)
#
#   pull_term = ||emb(clean) − centroid_c||²
#               (pulls clean embedding toward its class centre)
#
#   push_term = min_k ||emb(clean) − emb(probe_k)||²
#               (pushes clean embedding away from boundary probes)
#
# The log-mel spectrogram is extracted ONCE per batch (no gradient),
# then the conv stack is run ONCE with gradient (clean) and N_p times
# without gradient (probes), keeping the overhead manageable.
# ============================================================================

class BoundaryAwareReg:
    """Boundary-Aware Regularization from PACE (Li et al., arXiv 2602.03355)."""

    def __init__(self, num_classes, n_perturb=4,
                 mask_t_ratio=0.15, mask_f_ratio=0.15, margin=1.0):
        self.num_classes  = num_classes
        self.n_perturb    = n_perturb
        self.mask_t_ratio = mask_t_ratio
        self.mask_f_ratio = mask_f_ratio
        self.margin       = margin

    # ------------------------------------------------------------------
    @staticmethod
    def _tf_mask(spec, t_ratio, f_ratio):
        """Return a randomly time-frequency masked copy of spec (B,1,T,F)."""
        x = spec.clone()
        _, _, T, F = x.shape
        # time masking
        t_len   = max(1, int(T * t_ratio * (0.5 + torch.rand(1).item())))
        t_start = torch.randint(0, max(T - t_len, 1), (1,)).item()
        x[:, :, t_start:t_start + t_len, :] = 0.0
        # frequency masking
        f_len   = max(1, int(F * f_ratio * (0.5 + torch.rand(1).item())))
        f_start = torch.randint(0, max(F - f_len, 1), (1,)).item()
        x[:, :, :, f_start:f_start + f_len] = 0.0
        return x

    # ------------------------------------------------------------------
    def compute_loss(self, model, audio, labels, task):
        """
        model  : MCnn14 (or subclass) – must have spectrogram_extractor,
                 logmel_extractor, and get_embedding_from_spec()
        audio  : (B, T_audio)  raw waveform on the correct device
        labels : (B,)          integer class indices
        task   : int           current domain index
        """
        if not hasattr(model, 'get_embedding_from_spec'):
            return torch.tensor(0.0, device=audio.device)

        # ── extract log-mel once (no grad – spec extractor is frozen) ──
        with torch.no_grad():
            spec = model.spectrogram_extractor(audio)   # (B,1,T_stft,F)
            spec = model.logmel_extractor(spec)          # (B,1,T,mel_bins)

        # ── clean embedding WITH gradient ──────────────────────────────
        clean_emb = model.get_embedding_from_spec(spec, task)   # (B, D)
        B, D = clean_emb.shape

        # ── boundary probe embeddings  (no gradient) ───────────────────
        boundary_embs = []
        with torch.no_grad():
            for _ in range(self.n_perturb):
                masked = self._tf_mask(spec, self.mask_t_ratio, self.mask_f_ratio)
                boundary_embs.append(model.get_embedding_from_spec(masked, task))
        boundary_embs = torch.stack(boundary_embs, dim=1)   # (B, N_p, D)

        # ── batch centroids (mean clean emb per class, detached) ───────
        centroids = torch.zeros(self.num_classes, D, device=audio.device)
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                centroids[c] = clean_emb[mask].detach().mean(0)
        class_centroids = centroids[labels]                 # (B, D)

        # ── pull: distance to class centroid ───────────────────────────
        pull = torch.sum((clean_emb - class_centroids) ** 2, dim=-1)   # (B,)

        # ── push: min distance to boundary probes ──────────────────────
        dists = torch.sum(
            (clean_emb.unsqueeze(1) - boundary_embs) ** 2, dim=-1)     # (B, N_p)
        push  = dists.min(dim=1)[0]                                     # (B,)

        # ── hinge loss ─────────────────────────────────────────────────
        return torch.mean(torch.clamp(self.margin + pull - push, min=0.0))


# ============================================================================
# Supervised Contrastive Loss  (adapted from Khosla et al., NeurIPS 2020)
#
# Pulls embeddings of the same class together and pushes different-class
# embeddings apart — a cleaner alternative to BAR that works directly on
# clean embeddings without needing masked probes.
#
#   L_con = -1/|P(i)| Σ_{p∈P(i)}  log(
#               exp(sim(zi, zp)/τ)
#               ─────────────────────────────────────
#               Σ_{a≠i} exp(sim(zi, za)/τ)
#           )
#
# where P(i) = all samples with the same class label as i (excluding self).
# ============================================================================

class SupConLoss:
    """Supervised Contrastive Loss auxiliary objective."""

    def __init__(self, temperature=0.07):
        self.temperature = temperature

    def compute_loss(self, model, audio, labels, task):
        """
        model  : any model with get_embedding()
        audio  : (B, T_audio) on device
        labels : (B,) integer class indices
        task   : current domain index
        """
        if not hasattr(model, 'get_embedding'):
            return torch.tensor(0.0, device=audio.device)

        emb = model.get_embedding(audio, task)       # (B, D)
        emb = F.normalize(emb, dim=1)                # L2-normalise
        B   = emb.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=audio.device)

        # ── pairwise cosine similarity (B, B) ──────────────────────────
        sim = torch.matmul(emb, emb.T) / self.temperature

        # Numerical stability: subtract row max
        sim = sim - sim.max(dim=1, keepdim=True)[0].detach()

        # ── positive mask: same class, different sample ─────────────────
        lbl  = labels.view(-1, 1)
        pos  = (lbl == lbl.T).float()
        pos.fill_diagonal_(0.0)

        # ── log-softmax denominator (exclude self) ──────────────────────
        self_mask = torch.eye(B, dtype=torch.bool, device=audio.device)
        exp_sim   = torch.exp(sim).masked_fill(self_mask, 0.0)
        log_prob  = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # ── mean log-prob over positives (only for anchor with ≥1 pos) ──
        num_pos = pos.sum(dim=1)
        valid   = num_pos > 0
        if not valid.any():
            return torch.tensor(0.0, device=audio.device)

        loss = -(pos * log_prob).sum(dim=1) / (num_pos + 1e-8)
        return loss[valid].mean()


def _compute_accuracy(model, loader, task, device):

    correct, total = 0, 0
    model.eval()
    correct, total = 0, 0
    output_dict = {}

    for i, (inputs, targets, audio_file) in enumerate(loader):
        inputs = inputs.float()
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs, task)
        outputs = torch.softmax(outputs, dim=1)
        predicts = torch.max(outputs, dim=1)[1]
        target_labels = targets
        targets = torch.argmax(targets, dim=-1)
        correct += (predicts.cpu() == targets.cpu()).sum()
        total += len(targets)
        append_to_dict(output_dict, 'clipwise_output',
                       outputs.data.cpu().numpy())
        append_to_dict(output_dict, 'target', target_labels.cpu().numpy())
        #print(total, correct)
    #print(total, correct)
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    cm = metrics.confusion_matrix(np.argmax(output_dict['target'], axis=-1), np.argmax(output_dict['clipwise_output'], axis=-1), labels=None)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    #print(class_acc)
    return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


def _compute_uncertainity(model, loader, seen_domains, device):
    model.eval()
    nb_tasks = len(seen_domains) + 1  # +1 is D1
    correct, total = 0, 0
    output_dict = {}
    output_path = config.output_folder
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i, (inputs, targets, audio_file) in enumerate(loader):
        inputs = inputs.float().to(device)
        batch_size = inputs.shape[0]

        # Collect per-task softmax outputs; each element: (batch_size, classes_num)
        all_task_outputs = []
        for task in range(nb_tasks):
            with torch.no_grad():
                out = model(inputs, task)
            all_task_outputs.append(torch.softmax(out, dim=1).detach())

        # Stack → (nb_tasks, batch_size, classes_num)
        outputs_stacked = torch.stack(all_task_outputs, dim=0)

        epsilon = sys.float_info.min
        # Entropy per task per sample → (nb_tasks, batch_size)
        entropy = -torch.sum(outputs_stacked * torch.log(outputs_stacked + epsilon), dim=-1)

        # Per-sample task with minimum entropy → (batch_size,)
        _, task_ids = torch.min(entropy, dim=0)

        # Gather each sample's output from its best-task domain
        # outputs_stacked[task_ids[b], b, :] for each b in batch
        sample_idx = torch.arange(batch_size, device=device)
        final_outputs = outputs_stacked[task_ids, sample_idx, :]   # (batch_size, classes_num)

        predicts = torch.max(final_outputs, dim=1)[1]
        target_labels = targets
        targets_idx = torch.argmax(targets, dim=-1)
        correct += (predicts.cpu() == targets_idx.cpu()).sum()
        total += len(targets_idx)
        append_to_dict(output_dict, 'clipwise_output', final_outputs.data.cpu().numpy())
        append_to_dict(output_dict, 'target', target_labels.cpu().numpy())

        with open(os.path.join(output_path, 'output_' + timestr + '.txt'), 'a') as f:
            for b in range(batch_size):
                class_label = list(config.dict_class_labels.keys())[
                    list(config.dict_class_labels.values()).index(int(targets_idx[b].cpu().numpy()))
                ]
                f.write(audio_file[b] + '\t' + class_label + '\n')

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    cm = metrics.confusion_matrix(np.argmax(output_dict['target'], axis=-1), np.argmax(output_dict['clipwise_output'], axis=-1), labels=None)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    #print('confusion matrix:', cm)
    #print('class-wise accuracy', class_acc)
    return np.around(tensor2numpy(correct) * 100 / total, decimals=2)


class Learner():
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                 classes_num, num_tasks, model_type='dtn'):
        super(Learner, self).__init__()

        _model_zoo = {
            'dtn':        MCnn14DTN,
            'dtn2':       MCnn14DTN2,
            'mixed':      MCnn14Mixed,
            'bn_clf':     MCnn14CLF,
            'bn_cos':     MCnn14Cos,
            'bn_film':    MCnn14FiLM,
            'bn_se':      MCnn14SE,
            'bn_att':     MCnn14Att,
            'bn_se_deep': MCnn14SEDeep,
            'bn_se_clf':  MCnn14SEClf,
            'bn_res':     MCnn14Res,
            'bn_gn':       MCnn14GN,
            'bn_gn_se':    MCnn14GNSE,
            'bn_res_adil': MCnn14ResAdil,
            'bn_res_clf':  MCnn14ResClf,
            'bn_res_deep': MCnn14ResDeep,
            'bn_grn':      MCnn14GRN,
        }
        Model = _model_zoo.get(model_type, MCnn14)   # default / 'bn' -> baseline
        self.model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                           classes_num, num_tasks)

        self.classes_seen = 0
        self.known_classes = 0
        self.cur_task = -1
        self.class_increments = []

    def incremental_train(self, train_loader, val_loader, device, args):

        step = 0
        total = 0
        correct = 0
        check_point = 50
        self.model.to(device)

        self.model.freeze_weight()
        _norm_suffixes = ('weight', 'bias', 'alpha', 'gamma', 'beta')
        def _is_task_norm_param(name, task_idx):
            prefix = '.{}.'.format(task_idx)
            return any(prefix + s in name for s in _norm_suffixes)

        if hasattr(self.model, 'set_task_trainable'):
            # New architecture variants control their own per-domain freezing.
            self.model.set_task_trainable(self.cur_task)
            non_frozen_parameters = [p for p in self.model.parameters() if p.requires_grad]
            lr = args.learning_rate
        elif self.cur_task == 0:
            for name, param in self.model.named_parameters():
                if 'conv1' in name or 'conv2' in name or 'fc' in name:
                    param.requires_grad = True
                if 'bn' in name or 'domain_' in name:
                    if _is_task_norm_param(name, self.cur_task):
                        param.requires_grad = True
            non_frozen_parameters = [p for p in self.model.parameters() if p.requires_grad]
            lr = args.learning_rate
        elif self.cur_task > 0:
            for name, param in self.model.named_parameters():
                if 'bn' in name or 'domain_' in name:
                    if _is_task_norm_param(name, self.cur_task):
                        param.requires_grad = True
            non_frozen_parameters = [p for p in self.model.parameters() if p.requires_grad]
            # lr = args.learning_rate / 10
            lr = args.learning_rate

        print(f"params to be adapted")

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
        criteria = nn.CrossEntropyLoss(ignore_index=-1)

        optimizer = torch.optim.Adam(non_frozen_parameters, lr=lr, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=0., amsgrad=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-7)

        best_loss = float('inf')
        if args.save:
            save_path = config.save_resume_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        # -- Boundary-Aware Regularization (PACE) -----------------------
        lambda_reg = getattr(args, 'lambda_reg', 0.0)
        bar = BoundaryAwareReg(
            num_classes  = config.classes_num_DIL,
            n_perturb    = getattr(args, 'n_perturb', 4),
            mask_t_ratio = 0.15,
            mask_f_ratio = 0.15,
            margin       = 1.0,
        ) if lambda_reg > 0.0 else None
        if bar:
            print(f'  [BAR] boundary-aware reg enabled  '
                  f'lambda={lambda_reg}  n_perturb={bar.n_perturb}')

        # -- Supervised Contrastive Loss ---------------------------------
        lambda_con = getattr(args, 'lambda_con', 0.0)
        supcon = SupConLoss(
            temperature=getattr(args, 'con_temp', 0.07)
        ) if lambda_con > 0.0 else None
        if supcon:
            print(f'  [SupCon] contrastive loss enabled  '
                  f'lambda={lambda_con}  temperature={supcon.temperature}')
        # ---------------------------------------------------------------

        for epoch_idx in range(1, args.epoch + 1):
            self.model.train()

            sum_loss = 0
            sum_aux_loss = 0
            for batch_idx, (audio, target, _) in enumerate(train_loader):
                optimizer.zero_grad()
                audio = audio.float()
                target = target.float()
                audio = audio.to(device)
                target = target.to(device)
                target_indices = torch.argmax(target, dim=-1)

                logits = self.model(audio, self.cur_task) #cur_task is from 0, D1:0, D2:1, D3:2
                ce_loss = criteria(logits, target_indices)
                loss = ce_loss

                if bar is not None:
                    aux = bar.compute_loss(
                        self.model, audio, target_indices, self.cur_task)
                    loss = loss + lambda_reg * aux
                    sum_aux_loss += aux.item()

                if supcon is not None:
                    aux = supcon.compute_loss(
                        self.model, audio, target_indices, self.cur_task)
                    loss = loss + lambda_con * aux
                    sum_aux_loss += aux.item()

                sum_loss += loss.item()
                loss.backward()
                optimizer.step()
                step += 1

                if (batch_idx + 1) % check_point == 0 or (batch_idx + 1) == len(train_loader):
                    aux_str = (f'  aux: {sum_aux_loss / (batch_idx + 1):.3f}'
                               if (bar or supcon) else '')
                    print('==>>> epoch: {}, batch index: {}, step: {}, '
                          'train loss: {:.3f}{}'.format(
                              epoch_idx, batch_idx + 1, step,
                              sum_loss / (batch_idx + 1), aux_str))

            scheduler.step()
            torch.cuda.empty_cache()

            avg_loss = sum_loss / len(train_loader)
            if args.save and avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(),
                           os.path.join(save_path, 'checkpoint_' + 'D' + str(self.cur_task + 1) + '.pth'))
                print('==>>> epoch: {}, best loss improved to {:.4f}, checkpoint saved.'.format(epoch_idx, best_loss))

    def load_checkpoint(self, device):
        resume_path = os.path.join(config.save_resume_path, 'checkpoint_' + 'D' + str(self.cur_task + 1) + '.pth')
        incompatible = self.model.load_state_dict(
            torch.load(resume_path, map_location=torch.device(device)), strict=False)
        if incompatible.missing_keys:
            print(f'  [checkpoint] missing keys (DTN params, will use init): {len(incompatible.missing_keys)}')
        if incompatible.unexpected_keys:
            print(f'  [checkpoint] unexpected keys (BN params, ignored):     {len(incompatible.unexpected_keys)}')
        # Let architecture variants warm-start per-domain heads from the D1 classifier.
        if hasattr(self.model, 'on_checkpoint_loaded'):
            self.model.on_checkpoint_loaded(self.cur_task)
        print('model trained on Task D{} is loaded'.format(self.cur_task + 1))

    def incremental_setup(self, train_df, valid_df, seen_domains, batch_size, num_workers, device, args):

        self.cur_task += 1

        if self.cur_task == 0:
            self.load_checkpoint(device)
            self.cur_task += 1 #Skip the domain D1

        print("Starting DIL Task D{}".format(self.cur_task + 1))

        use_augment = getattr(args, 'augment', False)
        balance = getattr(args, 'class_balance', False)

        if use_augment:
            # Waveform-level augmentation (AudioAugmenter) on the training set.
            dataset_train = DILDatasetAug(train_df, config.audio_folder_DIL,
                                          training=True, augment=True)
            print('  [data] waveform augmentation ENABLED for training')
        else:
            dataset_train = DILDataset(train_df, config.audio_folder_DIL, training=True)
        dataset_val = DILDataset(valid_df, config.audio_folder_DIL, training=False)
        #dataset_test = DILDataset(test_df, config.audio_folder_DIL)

        # Data loader — optionally use a class-balanced sampler so every class
        # is drawn with equal probability (mutually exclusive with shuffle).
        if balance and hasattr(dataset_train, 'targets'):
            targets = np.asarray(dataset_train.targets)
            class_counts = np.bincount(targets, minlength=config.classes_num_DIL)
            class_counts = np.clip(class_counts, 1, None)
            sample_weights = 1.0 / class_counts[targets]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights), replacement=True)
            print('  [data] class-balanced sampler ENABLED  '
                  f'(class counts: {class_counts.tolist()})')
            train_loader = torch.utils.data.DataLoader(
                dataset=dataset_train, batch_size=batch_size, sampler=sampler,
                num_workers=num_workers, pin_memory=(device == 'cuda'))
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset=dataset_train, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=(device == 'cuda'))

        validate_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, shuffle=False,
                                                      num_workers=num_workers, pin_memory=(device == 'cuda'))


        if args.resume:
            # if i == 0:
            self.load_checkpoint(device)
        else:
            self.incremental_train(train_loader, validate_loader, device, args)

        #self.acc_prev(seen_domains, config.df_DIL_dev, config.df_DIL_eval, batch_size, num_workers, device)

    def acc_prev(self, seen_domains, df_dev_train, df_dev_test, batch_size, num_workers, device):
        self.model.to(device)
        self.model.eval()
        num_domains = len(seen_domains)
        domain_dict = {}
        accuracy_previous = []
        for domain in range(num_domains):
            #print('previous domain', seen_domains[domain])
            train_df = df_dev_train[df_dev_train['domain'].isin(seen_domains[domain])]
            valid_df = df_dev_test[df_dev_test['domain'].isin(seen_domains[domain])]
            dataset_val = DILDataset(valid_df, config.audio_folder_DIL, training=False)
            validate_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=1, shuffle=False,
                                                          num_workers=num_workers, pin_memory=True)

            #prev_accuracy = _compute_accuracy(self.model, validate_loader, domain, device)
            accuracy = _compute_uncertainity(self.model, validate_loader, seen_domains, device)

            print('seen domain: {} and its accuracy: {}'.format(seen_domains[domain], accuracy))

            accuracy_previous.append(accuracy)

        average_accuracy = np.mean(accuracy_previous).item()
        return average_accuracy

def train(args):
    # Arugments & parameters
    classes_num = config.classes_num_DIL
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epoch = args.epoch
    df_dev_train = config.df_DIL_dev_train
    df_dev_test = config.df_DIL_dev_test
    #df_eval =

    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    num_workers = args.num_workers

    dil_tasks_0 = ['D1'] # Model is trained on D1 and its checkpoint is released without data

    dil_task_1 = ['D2']

    dil_task_2 = ['D3']

    dil_tasks = [dil_task_1, dil_task_2]
    print('Tasks:', dil_tasks)

    np.random.seed(1193)

    num_tasks = len(dil_tasks) + 1 # + 1 is the domain D1

    if args.checkpoint_dir:
        config.save_resume_path = args.checkpoint_dir
    else:
        # Auto-generate a unique folder name from run hyperparameters
        # so runs never silently overwrite each other
        auto_name = '{}_lr{}_bs{}_ep{}'.format(
            args.model_type.upper(), args.learning_rate, args.batch_size, args.epoch)
        config.save_resume_path = os.path.join(
            '/home/aakash/dcase_task7/checkpoints', auto_name)

    # Per-task adapter depths for bn_res_deep (e.g. "1,3" -> D2=1, D3=3).
    if getattr(args, 'adapter_depths', None):
        depth_list = [int(d) for d in args.adapter_depths.split(',') if d.strip() != '']
        # depth_list maps to tasks 1..N (D2, D3, ...); task 0 (D1) stays identity.
        depths = {t + 1: depth_list[t] for t in range(len(depth_list))}
        set_adapter_depths(depths)

    seen_domains = []
    model = Learner(sample_rate, window_size, hop_size, mel_bins, fmin, fmax,
                    classes_num, num_tasks, model_type=args.model_type)

    for task in range(len(dil_tasks)):
        #print('Training on domain:', dil_tasks[task])
        seen_domains.append(dil_tasks[task])
        train_df = df_dev_train[df_dev_train['domain'].isin(dil_tasks[task])]
        test_df = df_dev_test[df_dev_test['domain'].isin(dil_tasks[task])]
        #eval_df = df_eval #Domain id is not available


        model.incremental_setup(train_df, test_df, seen_domains, batch_size, num_workers, device, args)
        seen_accuracy = model.acc_prev(seen_domains, df_dev_train, df_dev_test, batch_size, num_workers, device)
        print('Average Accuracy: ', seen_accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')

    #parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--num_workers', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--epoch', type=int, required=True)
    parser_train.add_argument('--resume', action='store_true', default=False)
    parser_train.add_argument('--save', action='store_true', default=False)
    parser_train.add_argument('--model_type', type=str,
                              choices=['bn', 'dtn', 'dtn2', 'mixed',
                                       'bn_clf', 'bn_cos', 'bn_film', 'bn_se', 'bn_att',
                                       'bn_se_deep', 'bn_se_clf', 'bn_res',
                                       'bn_gn', 'bn_gn_se',
                                       'bn_res_adil', 'bn_res_clf', 'bn_res_deep',
                                       'bn_grn'],
                              default='dtn',
                              help='bn=BatchNorm baseline; dtn=DTN for D2+D3; dtn2=LayerNorm+tanh gate; mixed=BN D2/DTN D3; '
                                   'bn_clf=per-domain linear head; bn_cos=per-domain cosine head; '
                                   'bn_film=FiLM adapters; bn_se=SE adapters; bn_att=attention pooling; '
                                   'bn_se_deep=deeper 2-layer SE adapters; '
                                   'bn_se_clf=SE adapters + per-domain head; '
                                   'bn_res=residual bottleneck conv adapters; '
                                   'bn_gn=GroupNorm baseline (no running stats); '
                                   'bn_gn_se=GroupNorm + SE adapters; '
                                   'bn_res_adil=Res adapters + ADIL residual head; '
                                   'bn_res_clf=Res adapters + per-domain head; '
                                   'bn_grn=GRN (Global Response Norm) for D2/D3, BN for D1')
    parser_train.add_argument('--lambda_con', type=float, default=0.0,
                              help='Weight for supervised contrastive loss (0=off). Try 0.1 or 0.3.')
    parser_train.add_argument('--con_temp', type=float, default=0.07,
                              help='Temperature for supervised contrastive loss')
    parser_train.add_argument('--checkpoint_dir', type=str, default=None,
                              help='Override config.save_resume_path for this run')
    parser_train.add_argument('--lambda_reg', type=float, default=0.0,
                              help='Weight for boundary-aware regularization loss '
                                   '(0 = disabled). Try 0.5 or 1.0.')
    parser_train.add_argument('--n_perturb', type=int, default=4,
                              help='Number of TF-masked probes per sample for BAR')
    parser_train.add_argument('--adapter_depths', type=str, default=None,
                              help='Per-task ResBottleneck adapter depth for '
                                   'bn_res_deep, comma-separated for D2,D3,... '
                                   'e.g. "1,3" gives D2=1 adapter/block, D3=3. '
                                   'D1 is always identity.')
    parser_train.add_argument('--augment', action='store_true', default=False,
                              help='Enable waveform-level AudioAugmenter on training data')
    parser_train.add_argument('--class_balance', action='store_true', default=False,
                              help='Use a class-balanced WeightedRandomSampler for training')
    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)

    else:
        raise Exception('Error argument!')
