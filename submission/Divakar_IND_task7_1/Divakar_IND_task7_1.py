"""
DCASE 2026 Task 7 — submission ResAdapt
Divakar_IND_task7_1_model.py

System: Per-domain residual bottleneck adapters (asymmetric depth) with a
        per-domain classifier head on top of a shared CNN14 backbone with
        per-domain BatchNorm.

Backbone adapted from PANNs (Qiuqiang Kong):
    https://github.com/qiuqiangkong/audioset_tagging_cnn

Usage
-----
Run directly to verify the model loads and print its architecture:

    python Divakar_IND_task7_1_model.py --task 1   # load after D2
    python Divakar_IND_task7_1_model.py --task 2   # load after D3

The .pth files must be in the same directory as this script:
    Divakar_IND_task7_1_D2_dictionary.pth
    Divakar_IND_task7_1_D3_dictionary.pth
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

# Paths to saved state-dictionaries for each incremental step.
# task=1 → after D2 training; task=2 → after D3 training.
CHECKPOINT_D2 = "Divakar_IND_task7_1_D2_dictionary.pth"
CHECKPOINT_D3 = "Divakar_IND_task7_1_D3_dictionary.pth"


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


def load_model(task: int = 1):
    """
    Return the submitted model with the checkpoint for the given incremental
    step loaded.

    task=1 : model after training on D2  (loads CHECKPOINT_D2)
    task=2 : model after training on D3  (loads CHECKPOINT_D3)
    """
    import os
    _CKPTS = {1: CHECKPOINT_D2, 2: CHECKPOINT_D3}
    if task not in _CKPTS:
        raise ValueError(f"task must be 1 or 2, got {task}")

    model = build_model(nb_tasks=3, adapter_depths=ADAPTER_DEPTHS)
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _CKPTS[task])
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=2,
                        help="Incremental step to load: 1=D2, 2=D3")
    args = parser.parse_args()

    model = load_model(task=args.task)
    print(f"\nModel loaded for task={args.task}\n")
    print(model)
