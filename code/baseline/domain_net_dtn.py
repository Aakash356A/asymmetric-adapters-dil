'''
Variant of domain_net.py that uses DynamicTanhNorm2d (DTN) for D2 / D3
instead of BatchNorm2d, while keeping BatchNorm2d for D1 (task index 0).

Drop-in replacement: swap
    from domain_net import *
for
    from domain_net_dtn import *
and change Model = MCnn14 → Model = MCnn14DTN in baseline_DIL_task7.py.
'''

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# ---------------------------------------------------------------------------
# Helpers (identical to domain_net.py)
# ---------------------------------------------------------------------------

def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


# ---------------------------------------------------------------------------
# DynamicTanhNorm2d  –  drop-in for BatchNorm2d on (B, C, H, W) tensors
# ---------------------------------------------------------------------------

class DynamicTanhNorm2d(nn.Module):
    """
    Dynamic Tanh normalisation for 4-D feature maps (B, C, H, W).

        y = gamma * tanh(alpha * x) + beta

    All three parameters (alpha, gamma, beta) are per-channel and learnable.
    No running statistics — purely activation-based, like DTN in:
      "Transformers without Normalization" (arXiv 2503.10622).
    """

    def __init__(self, num_features: int, init_alpha: float = 1.0):
        super().__init__()
        # (1, C, 1, 1) broadcasts correctly over (B, C, H, W)
        shape = (1, num_features, 1, 1)
        self.alpha = nn.Parameter(torch.full(shape, float(init_alpha)))
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta  = nn.Parameter(torch.zeros(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


# ---------------------------------------------------------------------------
# Helper: pick the right norm type per task index
# ---------------------------------------------------------------------------

def _make_norm2d(num_features: int, task_idx: int, bn_up_to: int = 0) -> nn.Module:
    """Return BN for tasks <= bn_up_to, DTN for the rest."""
    if task_idx <= bn_up_to:
        return nn.BatchNorm2d(num_features)
    return DynamicTanhNorm2d(num_features)


# ---------------------------------------------------------------------------
# ConvBlock with mixed BN / DTN norms
# ---------------------------------------------------------------------------

class ConvBlockDTN(nn.Module):
    def __init__(self, in_channels, out_channels, nb_tasks, bn_up_to=0):
        super(ConvBlockDTN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bnF = nn.ModuleList([_make_norm2d(out_channels, i, bn_up_to) for i in range(nb_tasks)])
        self.bnS = nn.ModuleList([_make_norm2d(out_channels, i, bn_up_to) for i in range(nb_tasks)])

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg', task=1):
        x = input
        x = F.relu_(self.bnF[task](self.conv1(x)))
        x = F.relu_(self.bnS[task](self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        return x


# ---------------------------------------------------------------------------
# MCnn14DTN  –  same as MCnn14 but with DTN for D2 / D3
# ---------------------------------------------------------------------------

class MCnn14DTN(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, nb_tasks=1, bn_up_to=0):

        super(MCnn14DTN, self).__init__()

        window   = 'hann'
        center   = True
        pad_mode = 'reflect'
        ref      = 1.0
        amin     = 1e-10
        top_db   = None

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center,
            pad_mode=pad_mode, freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax,
            ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)

        # Input norm: BN for D1, DTN for D2/D3
        # Applied to (B, mel_bins, T, 1) after transpose → works for both.
        self.bn0 = nn.ModuleList([_make_norm2d(mel_bins, i, bn_up_to) for i in range(nb_tasks)])

        self.conv_block1 = ConvBlockDTN(in_channels=1,    out_channels=64,   nb_tasks=nb_tasks, bn_up_to=bn_up_to)
        self.conv_block2 = ConvBlockDTN(in_channels=64,   out_channels=128,  nb_tasks=nb_tasks, bn_up_to=bn_up_to)
        self.conv_block3 = ConvBlockDTN(in_channels=128,  out_channels=256,  nb_tasks=nb_tasks, bn_up_to=bn_up_to)
        self.conv_block4 = ConvBlockDTN(in_channels=256,  out_channels=512,  nb_tasks=nb_tasks, bn_up_to=bn_up_to)
        self.conv_block5 = ConvBlockDTN(in_channels=512,  out_channels=1024, nb_tasks=nb_tasks, bn_up_to=bn_up_to)
        self.conv_block6 = ConvBlockDTN(in_channels=1024, out_channels=2048, nb_tasks=nb_tasks, bn_up_to=bn_up_to)

        self.fc = nn.Linear(2048, classes_num)

    # ------------------------------------------------------------------
    # Identical utility methods from MCnn14
    # ------------------------------------------------------------------

    def get_output_dim(self):
        return self.fc.out_features

    def change_output_dim(self, new_dim, second_iter=False):
        in_features  = self.fc.in_features
        out_features = self.fc.out_features
        print("in_features:", in_features, "out_features:", out_features)
        num_new_classes = new_dim - out_features
        new_fc = nn.Linear(in_features, out_features + num_new_classes)
        new_fc.weight.data[:out_features] = self.fc.weight.data
        new_fc.bias.data[:out_features]   = self.fc.bias.data
        self.fc = new_fc
        self.n_classes = new_dim

    def freeze_weight_conv(self):
        for param in self.conv_block1.parameters():
            param.requires_grad = False
        for param in self.conv_block2.parameters():
            param.requires_grad = False
        for param in self.conv_block3.parameters():
            param.requires_grad = False

    def freeze_weight(self):
        for param in self.parameters():
            param.requires_grad = False

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.fill_(0.)
            elif isinstance(m, nn.BatchNorm2d):
                m.bias.data.fill_(0.)
                m.weight.data.fill_(1.)
            elif isinstance(m, DynamicTanhNorm2d):
                nn.init.ones_(m.gamma)
                nn.init.zeros_(m.beta)
                nn.init.ones_(m.alpha)

    # ------------------------------------------------------------------
    # Forward  (identical call signature to MCnn14)
    # ------------------------------------------------------------------

    def forward(self, input, task=1):
        """Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(input)   # (B, 1, T, F)
        x = self.logmel_extractor(x)

        x = x.transpose(1, 3)                  # (B, F, T, 1)
        x = self.bn0[task](x)
        x = x.transpose(1, 3)                  # (B, 1, T, F)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = self.fc(x)
        return x

    def get_embedding(self, input, task=1):
        """Returns the 2048-d global pooling embedding (before the FC layer)."""
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)

        x = x.transpose(1, 3)
        x = self.bn0[task](x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        return x1 + x2   # (B, 2048)


# ---------------------------------------------------------------------------
# MCnn14Mixed  –  BN for D1 + D2, DTN only for D3
# ---------------------------------------------------------------------------

class MCnn14Mixed(MCnn14DTN):
    """BN normalisation for D1 (task 0) and D2 (task 1), DTN for D3 (task 2+)."""
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, nb_tasks=1):
        super().__init__(sample_rate, window_size, hop_size, mel_bins, fmin,
                         fmax, classes_num, nb_tasks, bn_up_to=1)


# ---------------------------------------------------------------------------
# DomainDiscriminator  –  lightweight MLP trained on top of the shared embedding
# ---------------------------------------------------------------------------

class DomainDiscriminator(nn.Module):
    """
    Small 2-layer MLP that predicts the domain index (0=D1, 1=D2, 2=D3)
    from the 2048-d backbone embedding.

    Trained jointly with the classification head during incremental phases.
    At inference, replaces entropy-based routing with a direct argmax.
    """

    def __init__(self, embed_dim: int = 2048, hidden_dim: int = 256, num_domains: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """embedding: (B, embed_dim)  →  logits: (B, num_domains)"""
        return self.net(embedding)


# ---------------------------------------------------------------------------
# DynamicTanhNorm2_4d  –  LayerNorm-style normalization + tanh gate
#
# Normalises per-sample over the *channel* axis (batch-agnostic, like
# LayerNorm applied to each spatial position), then applies a per-channel
# learnable tanh gate and affine transform:
#
#     y = gamma * tanh(alpha * norm(x)) + beta
#
# where norm(x) = (x - mean_C) / sqrt(var_C + eps).
#
# Key difference from DynamicTanhNorm2d above: that class gates the *raw*
# activations (no normalisation step); this class normalises first, so the
# tanh gate always operates on a well-scaled input regardless of the conv
# output magnitude.
#
# Drop-in for BatchNorm2d on 4-D feature maps (B, C, H, W).
# ---------------------------------------------------------------------------

class DynamicTanhNorm2_4d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6, init_alpha: float = 1.0):
        super().__init__()
        self.eps = eps
        shape = (1, num_features, 1, 1)
        self.alpha = nn.Parameter(torch.full(shape, float(init_alpha)))
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta  = nn.Parameter(torch.zeros(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise over the channel axis at each spatial location
        mean = x.mean(dim=1, keepdim=True)                   # (B, 1, H, W)
        var  = x.var(dim=1, keepdim=True, unbiased=False)    # (B, 1, H, W)
        x_norm = (x - mean) * torch.rsqrt(var + self.eps)
        return self.gamma * torch.tanh(self.alpha * x_norm) + self.beta


def _make_norm2d_v2(num_features: int, task_idx: int, bn_up_to: int = 0) -> nn.Module:
    """Return BN for tasks <= bn_up_to, DynamicTanhNorm2_4d for the rest."""
    if task_idx <= bn_up_to:
        return nn.BatchNorm2d(num_features)
    return DynamicTanhNorm2_4d(num_features)


# ---------------------------------------------------------------------------
# ConvBlockDTN2  –  ConvBlock using DynamicTanhNorm2_4d
# ---------------------------------------------------------------------------

class ConvBlockDTN2(nn.Module):
    def __init__(self, in_channels, out_channels, nb_tasks, bn_up_to=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bnF = nn.ModuleList([_make_norm2d_v2(out_channels, i, bn_up_to) for i in range(nb_tasks)])
        self.bnS = nn.ModuleList([_make_norm2d_v2(out_channels, i, bn_up_to) for i in range(nb_tasks)])

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg', task=1):
        x = input
        x = F.relu_(self.bnF[task](self.conv1(x)))
        x = F.relu_(self.bnS[task](self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        return x


# ---------------------------------------------------------------------------
# MCnn14DTN2  –  CNN14 backbone with DynamicTanhNorm2_4d (normalise-then-gate)
# ---------------------------------------------------------------------------

class MCnn14DTN2(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, nb_tasks=1, bn_up_to=0):

        super().__init__()

        window   = 'hann'
        center   = True
        pad_mode = 'reflect'
        ref      = 1.0
        amin     = 1e-10
        top_db   = None

        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center,
            pad_mode=pad_mode, freeze_parameters=True)

        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax,
            ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)

        # Input norm: BN for D1, DTN2 for D2/D3
        self.bn0 = nn.ModuleList([_make_norm2d_v2(mel_bins, i, bn_up_to) for i in range(nb_tasks)])

        self.conv_block1 = ConvBlockDTN2(in_channels=1,    out_channels=64,   nb_tasks=nb_tasks, bn_up_to=bn_up_to)
        self.conv_block2 = ConvBlockDTN2(in_channels=64,   out_channels=128,  nb_tasks=nb_tasks, bn_up_to=bn_up_to)
        self.conv_block3 = ConvBlockDTN2(in_channels=128,  out_channels=256,  nb_tasks=nb_tasks, bn_up_to=bn_up_to)
        self.conv_block4 = ConvBlockDTN2(in_channels=256,  out_channels=512,  nb_tasks=nb_tasks, bn_up_to=bn_up_to)
        self.conv_block5 = ConvBlockDTN2(in_channels=512,  out_channels=1024, nb_tasks=nb_tasks, bn_up_to=bn_up_to)
        self.conv_block6 = ConvBlockDTN2(in_channels=1024, out_channels=2048, nb_tasks=nb_tasks, bn_up_to=bn_up_to)

        self.fc = nn.Linear(2048, classes_num)

    def get_output_dim(self):
        return self.fc.out_features

    def change_output_dim(self, new_dim, second_iter=False):
        in_features  = self.fc.in_features
        out_features = self.fc.out_features
        num_new_classes = new_dim - out_features
        new_fc = nn.Linear(in_features, out_features + num_new_classes)
        new_fc.weight.data[:out_features] = self.fc.weight.data
        new_fc.bias.data[:out_features]   = self.fc.bias.data
        self.fc = new_fc
        self.n_classes = new_dim

    def freeze_weight(self):
        for param in self.parameters():
            param.requires_grad = False

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.fill_(0.)
            elif isinstance(m, nn.BatchNorm2d):
                m.bias.data.fill_(0.)
                m.weight.data.fill_(1.)
            elif isinstance(m, DynamicTanhNorm2_4d):
                nn.init.ones_(m.gamma)
                nn.init.zeros_(m.beta)
                nn.init.ones_(m.alpha)

    def forward(self, input, task=1):
        """Input: (batch_size, data_length)"""
        x = self.spectrogram_extractor(input)   # (B, 1, T, F)
        x = self.logmel_extractor(x)

        x = x.transpose(1, 3)                  # (B, F, T, 1)
        x = self.bn0[task](x)
        x = x.transpose(1, 3)                  # (B, 1, T, F)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = self.fc(x)
        return x

    def get_embedding(self, input, task=1):
        """Returns the 2048-d global pooling embedding (before the FC layer)."""
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)

        x = x.transpose(1, 3)
        x = self.bn0[task](x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(2, 2), pool_type='avg', task=task)
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        return x1 + x2   # (B, 2048)
