"""
DCASE 2026 Task 7 — submission ResAdapt

Standalone definition of the submitted system:
    Per-domain residual bottleneck adapters with asymmetric depth, a
    warm-started per-domain classifier head, on top of a shared, frozen
    CNN14 backbone with per-domain BatchNorm.

The backbone (CNN14) is adapted from PANNs by Qiuqiang Kong
(https://github.com/qiuqiangkong/audioset_tagging_cnn).

Submitted configuration:
    adapter depths  : D1 = identity, D2 = 1, D3 = 3   (asymmetric)
    classifier      : per-domain linear head (warm-started from D1)
    auxiliary loss  : supervised contrastive (training only)

Per-domain weights are provided as separate state dicts:
    checkpoint/checkpoint_D2.pth   (model state after D1->D2)
    checkpoint/checkpoint_D3.pth   (model state after D1->D2->D3)

Usage
-----
    from ResAdapt import build_model, ADAPTER_DEPTHS

    model = build_model(nb_tasks=3, adapter_depths=ADAPTER_DEPTHS)
    # final model holding all domains' parameters:
    state = torch.load("checkpoint/checkpoint_D3.pth", map_location="cpu")
    model.load_state_dict(state)   # strict=True: keys match exactly
    model.eval()

    # Domain-agnostic (ADIL) inference: forward through every seen domain's
    # specific layers and keep the minimum-entropy prediction.
    logits_per_task = [model(waveform, task=t) for t in range(nb_tasks)]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank


# ---------------------------------------------------------------------------
# Front-end / spectrogram parameters (must match training, config_task7.py)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE    = 320
MEL_BINS    = 64
FMIN        = 50
FMAX        = 14000
CLASSES_NUM = 10

# Submitted per-domain adapter depths: D1 (task 0) is always identity;
# task 1 (D2) = 1 adapter/block, task 2 (D3) = 3 adapters/block.
ADAPTER_DEPTHS = {1: 1, 2: 3}

# Channels produced by each of the 6 CNN14 conv blocks.
_BLOCK_CHANNELS = (64, 128, 256, 512, 1024, 2048)


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


# ---------------------------------------------------------------------------
# CNN14 convolutional block with per-domain BatchNorm
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_tasks):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1), bias=False)
        # Per-domain BN (one set per task).
        self.bnF = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(nb_tasks)])
        self.bnS = nn.ModuleList([nn.BatchNorm2d(out_channels) for _ in range(nb_tasks)])

    def forward(self, input, pool_size=(2, 2), pool_type='avg', task=1):
        x = F.relu_(self.bnF[task](self.conv1(input)))
        x = F.relu_(self.bnS[task](self.conv2(x)))
        if pool_type == 'max':
            return F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            return F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            return F.avg_pool2d(x, kernel_size=pool_size) + F.max_pool2d(x, kernel_size=pool_size)
        raise ValueError('Incorrect pool_type argument!')


# ---------------------------------------------------------------------------
# Residual bottleneck adapter (zero-init -> identity at start) and stack
# ---------------------------------------------------------------------------
class ResBottleneck(nn.Module):
    def __init__(self, channels, bottleneck_ratio=4):
        super().__init__()
        hidden = max(channels // bottleneck_ratio, 8)
        self.down = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.up   = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.up.weight)   # identity residual at init
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.relu(self.down(x)))


class ResAdapterStack(nn.Module):
    """A stack of `depth` residual bottleneck adapters (identity at init)."""

    def __init__(self, channels, depth=1, bottleneck_ratio=4):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ResBottleneck(channels, bottleneck_ratio) for _ in range(max(1, depth))])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


# ---------------------------------------------------------------------------
# CNN14 backbone (shared, frozen) + per-domain BN, adapters and head
# ---------------------------------------------------------------------------
class MCnn14ResDeepClf(nn.Module):
    """
    Shared CNN14 backbone with per-domain BatchNorm, a per-domain stack of
    residual adapters (depth configurable per task), and a per-domain linear
    classifier head. Task 0 (D1) uses identity adapters and the original head.
    """

    def __init__(self, classes_num=CLASSES_NUM, nb_tasks=3, adapter_depths=None):
        super().__init__()
        adapter_depths = dict(adapter_depths or ADAPTER_DEPTHS)

        # --- front end ---
        self.spectrogram_extractor = Spectrogram(
            n_fft=WINDOW_SIZE, hop_length=HOP_SIZE, win_length=WINDOW_SIZE,
            window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(
            sr=SAMPLE_RATE, n_fft=WINDOW_SIZE, n_mels=MEL_BINS, fmin=FMIN,
            fmax=FMAX, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)

        # --- per-domain input BN + shared conv backbone ---
        self.bn0 = nn.ModuleList([nn.BatchNorm2d(MEL_BINS) for _ in range(nb_tasks)])
        self.conv_block1 = ConvBlock(1,    64,   nb_tasks)
        self.conv_block2 = ConvBlock(64,   128,  nb_tasks)
        self.conv_block3 = ConvBlock(128,  256,  nb_tasks)
        self.conv_block4 = ConvBlock(256,  512,  nb_tasks)
        self.conv_block5 = ConvBlock(512,  1024, nb_tasks)
        self.conv_block6 = ConvBlock(1024, 2048, nb_tasks)

        # --- original (D1) classifier ---
        self.fc = nn.Linear(2048, classes_num)

        # --- per-domain residual adapter stacks (D1 = identity) ---
        adapters = []
        for t in range(nb_tasks):
            if t == 0:
                adapters.append(nn.ModuleList([nn.Identity() for _ in _BLOCK_CHANNELS]))
            else:
                depth = adapter_depths.get(t, 1)
                adapters.append(nn.ModuleList(
                    [ResAdapterStack(c, depth) for c in _BLOCK_CHANNELS]))
        self.adapters = nn.ModuleList(adapters)

        # --- per-domain classifier heads (warm-started from fc during training) ---
        self.fc_heads = nn.ModuleList(
            [nn.Linear(2048, classes_num) for _ in range(nb_tasks)])

    # -- adapter-aware embedding ------------------------------------------
    def _embedding_from_spec(self, spec, task):
        x = spec.transpose(1, 3)
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
        return x1 + x2   # (B, 2048)

    def get_embedding(self, input, task=1):
        spec = self.spectrogram_extractor(input)
        spec = self.logmel_extractor(spec)
        return self._embedding_from_spec(spec, task)

    def forward(self, input, task=1):
        emb = self.get_embedding(input, task)
        if task == 0:
            return self.fc(emb)          # D1: original head
        return self.fc_heads[task](emb)  # D2/D3: per-domain head


def build_model(nb_tasks=3, adapter_depths=None, classes_num=CLASSES_NUM):
    """Instantiate the submitted model with the submitted adapter depths."""
    return MCnn14ResDeepClf(classes_num=classes_num, nb_tasks=nb_tasks,
                            adapter_depths=adapter_depths or ADAPTER_DEPTHS)


@torch.no_grad()
def predict_domain_agnostic(model, waveform, nb_tasks=3):
    """
    Domain-agnostic (ADIL) inference: forward through every seen domain's
    specific layers and keep, per sample, the minimum-entropy prediction.

    waveform : (B, T) float tensor, 4 s at 32 kHz.
    returns  : (B,) predicted class indices.
    """
    probs = [torch.softmax(model(waveform, task=t), dim=1) for t in range(nb_tasks)]
    stacked = torch.stack(probs, dim=0)                       # (T, B, C)
    entropy = -(stacked * torch.log(stacked + 1e-12)).sum(-1)  # (T, B)
    best = entropy.argmin(dim=0)                               # (B,)
    idx = torch.arange(waveform.shape[0], device=waveform.device)
    final = stacked[best, idx, :]                             # (B, C)
    return final.argmax(dim=1)
