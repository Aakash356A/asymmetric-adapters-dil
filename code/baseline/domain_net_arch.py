'''
Architectural variants of the DCASE-2026 Task-7 baseline (domain_net.py / MCnn14).

ALL variants keep the *exact* baseline backbone:
  * shared, frozen CNN14 conv weights (loaded from the released D1 checkpoint)
  * per-domain BatchNorm (bn0, bnF, bnS) -- still the core domain adaptation

and, on TOP of that, add a small new per-domain component that gives the weak
domains (D2 / D3) more capacity / a better classifier.  Domain D1 (task index 0)
ALWAYS uses the original pretrained path, so the released-checkpoint behaviour is
preserved bit-for-bit; the new machinery only kicks in for task >= 1.

New --model_type values wired in baseline_DIL_task7.py:
  bn_clf   -> MCnn14CLF   : per-domain linear classifier head      (idea #1)
  bn_cos   -> MCnn14Cos   : per-domain cosine / NCM classifier head (idea #2)
  bn_film  -> MCnn14FiLM  : per-domain FiLM feature adapters        (idea #3a)
  bn_se    -> MCnn14SE    : per-domain Squeeze-Excitation adapters  (idea #3b)
  bn_att   -> MCnn14Att   : per-domain attention-pooling head       (idea #4)

The original MCnn14 / MCnn14DTN / MCnn14Mixed classes are untouched.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from domain_net import MCnn14   # reuse the exact baseline backbone + ConvBlock


# Channels produced by each of the 6 conv blocks (after its pooling).
_BLOCK_CHANNELS = (64, 128, 256, 512, 1024, 2048)


# ---------------------------------------------------------------------------
# Common base: per-domain trainability + checkpoint warm-start hooks
# ---------------------------------------------------------------------------

class _ArchBase(MCnn14):
    """
    Adds two hooks used by the training loop:

      set_task_trainable(cur_task)
          Freeze everything, then unfreeze ONLY the parameters that belong to
          the current domain: its BatchNorm (bn0/bnF/bnS) plus whatever new
          per-domain module this variant adds. This reproduces the baseline's
          "adapt BN of this domain" behaviour and extends it to the new module.

      on_checkpoint_loaded(loaded_task)
          Called right after the D1 checkpoint is loaded. Lets a variant
          warm-start its per-domain heads from the pretrained D1 classifier.

    Matching is done with *container-qualified* prefixes (e.g. ``.bnF.1.``,
    ``fc_heads.1.``) so an inner block index can never be mistaken for a task
    index -- this is robust where a plain ``.1.`` substring check would not be.
    """

    # Prefixes that identify a parameter as belonging to task ``t``.
    _TASK_PREFIXES = (
        'bn0.{t}.', '.bnF.{t}.', '.bnS.{t}.',     # per-domain BatchNorm (baseline)
        'fc_heads.{t}.', 'cos_heads.{t}.',         # per-domain classifier heads
        'adapters.{t}.', 'att_heads.{t}.',         # per-domain adapters / pooling
    )

    def set_task_trainable(self, cur_task):
        for p in self.parameters():
            p.requires_grad_(False)
        prefixes = [s.format(t=cur_task) for s in self._TASK_PREFIXES]
        for name, p in self.named_parameters():
            if any(pref in name for pref in prefixes):
                p.requires_grad_(True)

    def on_checkpoint_loaded(self, loaded_task):
        pass


# ---------------------------------------------------------------------------
# Idea #1 -- per-domain LINEAR classifier head
# ---------------------------------------------------------------------------

class MCnn14CLF(_ArchBase):
    """
    Baseline backbone + a separate Linear(2048 -> classes) head per domain.

    The baseline freezes the D1 classifier and reuses it for D2/D3 even though
    those domains adapted their BatchNorm -- a mismatch. Here D2/D3 get their
    own head (warm-started from the D1 head) trained alongside their BN.
    D1 still uses the original self.fc.
    """

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, nb_tasks=1):
        super().__init__(sample_rate, window_size, hop_size, mel_bins, fmin,
                         fmax, classes_num, nb_tasks)
        self.fc_heads = nn.ModuleList(
            [nn.Linear(2048, classes_num) for _ in range(nb_tasks)])

    def on_checkpoint_loaded(self, loaded_task):
        # Warm-start every per-domain head from the pretrained D1 classifier.
        if loaded_task == 0:
            with torch.no_grad():
                for head in self.fc_heads:
                    head.weight.copy_(self.fc.weight)
                    head.bias.copy_(self.fc.bias)

    def forward(self, input, task=1):
        emb = self.get_embedding(input, task)          # (B, 2048), per-domain BN
        if task == 0:
            return self.fc(emb)                        # D1: original pretrained head
        return self.fc_heads[task](emb)


# ---------------------------------------------------------------------------
# Idea #2 -- per-domain COSINE / NCM classifier head
# ---------------------------------------------------------------------------

class CosineHead(nn.Module):
    """L2-normalised cosine classifier with a learnable temperature/scale."""

    def __init__(self, in_features, classes_num, scale=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes_num, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return self.scale * F.linear(x, w)


class MCnn14Cos(_ArchBase):
    """
    Baseline backbone + a per-domain cosine (NCM-style) head for D2/D3.

    Cosine heads remove feature-magnitude bias between domains, which both
    helps the weak domains and makes the entropy-based domain routing more
    stable. D1 keeps the original linear self.fc.
    """

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, nb_tasks=1):
        super().__init__(sample_rate, window_size, hop_size, mel_bins, fmin,
                         fmax, classes_num, nb_tasks)
        self.cos_heads = nn.ModuleList(
            [CosineHead(2048, classes_num) for _ in range(nb_tasks)])

    def on_checkpoint_loaded(self, loaded_task):
        # Warm-start cosine directions from the (normalised) D1 classifier.
        if loaded_task == 0:
            with torch.no_grad():
                for head in self.cos_heads:
                    head.weight.copy_(F.normalize(self.fc.weight, dim=1))

    def forward(self, input, task=1):
        emb = self.get_embedding(input, task)
        if task == 0:
            return self.fc(emb)
        return self.cos_heads[task](emb)


# ---------------------------------------------------------------------------
# Idea #3 -- per-domain FEATURE ADAPTERS (FiLM / SE)
# ---------------------------------------------------------------------------

class FiLM(nn.Module):
    """Per-channel feature-wise affine: y = gamma * x + beta  (init = identity)."""

    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return self.gamma * x + self.beta


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (init ~ identity-ish)."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        s = x.mean(dim=(2, 3))                 # (B, C) global average pool
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))         # (B, C) gate in [0, 1]
        return x * s.unsqueeze(-1).unsqueeze(-1)


class _AdapterNet(_ArchBase):
    """
    Shared machinery for FiLM / SE variants: a per-domain stack of 6 adapters
    (one after each conv block). Task 0 uses Identity adapters so D1 is the
    untouched baseline; D2/D3 learn adapters on top of their BN. The shared,
    frozen self.fc classifier is reused for all domains -- the adapters' job is
    to realign each domain's features to that fixed classifier.
    """

    def _make_adapter(self, channels):
        raise NotImplementedError

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, nb_tasks=1):
        super().__init__(sample_rate, window_size, hop_size, mel_bins, fmin,
                         fmax, classes_num, nb_tasks)
        adapters = []
        for t in range(nb_tasks):
            if t == 0:
                adapters.append(nn.ModuleList(
                    [nn.Identity() for _ in _BLOCK_CHANNELS]))
            else:
                adapters.append(nn.ModuleList(
                    [self._make_adapter(c) for c in _BLOCK_CHANNELS]))
        self.adapters = nn.ModuleList(adapters)

    def forward(self, input, task=1):
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0[task](x)
        x = x.transpose(1, 3)

        blocks = [self.conv_block1, self.conv_block2, self.conv_block3,
                  self.conv_block4, self.conv_block5, self.conv_block6]
        adapt = self.adapters[task]
        for i, blk in enumerate(blocks):
            x = blk(x, pool_size=(2, 2), pool_type='avg', task=task)
            x = adapt[i](x)
            x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        return self.fc(x)


class MCnn14FiLM(_AdapterNet):
    def _make_adapter(self, channels):
        return FiLM(channels)


class MCnn14SE(_AdapterNet):
    def _make_adapter(self, channels):
        return SEBlock(channels)


# ---------------------------------------------------------------------------
# Idea #4 -- per-domain ATTENTION-POOLING head (PANNs AttBlock style)
# ---------------------------------------------------------------------------

class AttBlock(nn.Module):
    """
    Attention pooling + classification over the (T*F) tokens of the final
    feature map. Learns *where* to attend per domain instead of the fixed
    mean+max pooling, and outputs class logits directly.
    """

    def __init__(self, in_dim, classes_num):
        super().__init__()
        self.att = nn.Conv1d(in_dim, classes_num, kernel_size=1)
        self.cla = nn.Conv1d(in_dim, classes_num, kernel_size=1)

    def forward(self, x):                       # x: (B, C, L)
        norm_att = torch.softmax(self.att(x), dim=-1)   # attention over tokens
        cla = self.cla(x)                                # per-token class scores
        return torch.sum(norm_att * cla, dim=-1)         # (B, classes)


class MCnn14Att(_ArchBase):
    """
    Baseline backbone + a per-domain attention-pooling head for D2/D3.
    D1 keeps the original mean+max pooling and self.fc.
    """

    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, nb_tasks=1):
        super().__init__(sample_rate, window_size, hop_size, mel_bins, fmin,
                         fmax, classes_num, nb_tasks)
        self.att_heads = nn.ModuleList(
            [AttBlock(2048, classes_num) for _ in range(nb_tasks)])

    def _feature_map(self, input, task):
        """Run the conv stack, return the (B, 2048, T', F') feature map."""
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0[task](x)
        x = x.transpose(1, 3)
        for blk in (self.conv_block1, self.conv_block2, self.conv_block3,
                    self.conv_block4, self.conv_block5, self.conv_block6):
            x = blk(x, pool_size=(2, 2), pool_type='avg', task=task)
            x = F.dropout(x, p=0.2, training=self.training)
        return x

    def forward(self, input, task=1):
        x = self._feature_map(input, task)
        if task == 0:
            # D1: original mean+max pooling + pretrained classifier
            x = torch.mean(x, dim=3)
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            return self.fc(x1 + x2)
        B, C, T, Fr = x.shape
        return self.att_heads[task](x.reshape(B, C, T * Fr))
