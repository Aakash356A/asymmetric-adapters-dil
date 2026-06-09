
from torch.utils.data import Dataset
import pandas as pd
import os
import tqdm
import librosa
import numpy as np
import config_task7 as config


def to_one_hot(k, classes_num):
    target = np.zeros(classes_num, dtype=np.float32)
    target[k] = 1.0
    return target


def pad_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x), dtype=x.dtype)))
    else:
        return x


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x), dtype=x.dtype)))
    else:
        return x[0:max_len]


def crop_or_pad(x, target_len):
    """Return exactly target_len samples, zero-padding if shorter."""
    if len(x) < target_len:
        return np.concatenate((x, np.zeros(target_len - len(x), dtype=x.dtype)))
    return x[:target_len]


class AudioAugmenter:
    """
    Waveform-level augmentations for DCASE 2026 Task 7.

    Intended for 4-second, 32-kHz mono clips.

    Includes:
      - gain / loudness perturbation
      - time shift
      - random crop jitter
      - time stretching
      - pitch shifting

    No external data is used.
    """

    def __init__(
        self,
        sample_rate=32000,
        clip_samples=128000,
        enable=True,
        seed=None,
    ):
        self.sample_rate = sample_rate
        self.clip_samples = clip_samples
        self.enable = enable
        self.rng = np.random.default_rng(seed)

        # Class-specific conservative settings.
        # Labels from config:
        # 0 alarm
        # 1 baby_cry
        # 2 dog_bark
        # 3 engine
        # 4 fire
        # 5 footsteps
        # 6 knocking
        # 7 telephone_ringing
        # 8 piano
        # 9 speech
        self.class_policy = {
            0: dict(pitch_semitones=(-1.0, 1.0), stretch=(0.95, 1.05)),  # alarm
            1: dict(pitch_semitones=(-1.0, 1.0), stretch=(0.97, 1.03)),  # baby_cry
            2: dict(pitch_semitones=(-2.0, 2.0), stretch=(0.95, 1.08)),  # dog_bark
            3: dict(pitch_semitones=(-0.5, 0.5), stretch=(0.90, 1.10)),  # engine
            4: dict(pitch_semitones=(-0.5, 0.5), stretch=(0.90, 1.10)),  # fire
            5: dict(pitch_semitones=(-0.5, 0.5), stretch=(0.90, 1.10)),  # footsteps
            6: dict(pitch_semitones=(-0.5, 0.5), stretch=(0.90, 1.10)),  # knocking
            7: dict(pitch_semitones=(-1.0, 1.0), stretch=(0.95, 1.05)),  # telephone_ringing
            8: dict(pitch_semitones=(-1.0, 1.0), stretch=(0.97, 1.03)),  # piano
            9: dict(pitch_semitones=(-1.0, 1.0), stretch=(0.97, 1.03)),  # speech
        }

        # Global probabilities tuned to be conservative for DIL.
        self.p_gain = 0.70
        self.p_time_shift = 0.60
        self.p_random_crop_jitter = 0.50
        self.p_time_stretch = 0.25
        self.p_pitch_shift = 0.25

        # Hyperparameters.
        self.gain_db_min = -6.0
        self.gain_db_max = 6.0
        self.max_time_shift_sec = 0.50

    def __call__(self, waveform, class_idx):
        if not self.enable:
            return crop_or_pad(waveform, self.clip_samples).astype(np.float32)

        waveform = np.asarray(waveform, dtype=np.float32)

        # Always make sure we have enough material before transforms.
        waveform = crop_or_pad(waveform, self.clip_samples)

        # 1. Random crop jitter.
        # Since training segments are already 4 s, this mostly matters if you later load
        # slightly longer excerpts. It is harmless here and keeps exact length.
        if self.rng.random() < self.p_random_crop_jitter:
            waveform = self.random_crop_jitter(waveform)

        # 2. Time shift.
        if self.rng.random() < self.p_time_shift:
            waveform = self.time_shift(waveform)

        # 3. Time stretch.
        if self.rng.random() < self.p_time_stretch:
            waveform = self.time_stretch(waveform, class_idx)

        # 4. Pitch shift.
        if self.rng.random() < self.p_pitch_shift:
            waveform = self.pitch_shift(waveform, class_idx)

        # 5. Gain / loudness.
        if self.rng.random() < self.p_gain:
            waveform = self.random_gain(waveform)

        waveform = crop_or_pad(waveform, self.clip_samples)
        waveform = self.safe_clip(waveform)

        return waveform.astype(np.float32)

    def random_gain(self, waveform):
        gain_db = self.rng.uniform(self.gain_db_min, self.gain_db_max)
        gain = 10.0 ** (gain_db / 20.0)
        return waveform * gain

    def time_shift(self, waveform):
        max_shift = int(self.max_time_shift_sec * self.sample_rate)
        shift = int(self.rng.integers(-max_shift, max_shift + 1))
        if shift == 0:
            return waveform

        # Zero-fill shift instead of circular shift.
        shifted = np.zeros_like(waveform)
        if shift > 0:
            shifted[shift:] = waveform[:-shift]
        else:
            shifted[:shift] = waveform[-shift:]
        return shifted

    def random_crop_jitter(self, waveform):
        # For exact 4-second clips, this is equivalent to a small zero-padded shift.
        # Useful if future loader returns slightly longer clips.
        if len(waveform) <= self.clip_samples:
            return waveform

        max_start = len(waveform) - self.clip_samples
        start = int(self.rng.integers(0, max_start + 1))
        return waveform[start:start + self.clip_samples]

    def time_stretch(self, waveform, class_idx):
        policy = self.class_policy.get(class_idx, dict(stretch=(0.95, 1.05)))
        low, high = policy["stretch"]
        rate = float(self.rng.uniform(low, high))

        try:
            stretched = librosa.effects.time_stretch(waveform.astype(np.float32), rate=rate)
            return crop_or_pad(stretched, self.clip_samples)
        except Exception:
            return waveform

    def pitch_shift(self, waveform, class_idx):
        policy = self.class_policy.get(class_idx, dict(pitch_semitones=(-1.0, 1.0)))
        low, high = policy["pitch_semitones"]
        n_steps = float(self.rng.uniform(low, high))

        # Avoid tiny no-op shifts.
        if abs(n_steps) < 0.10:
            return waveform

        try:
            shifted = librosa.effects.pitch_shift(
                waveform.astype(np.float32),
                sr=self.sample_rate,
                n_steps=n_steps,
            )
            return crop_or_pad(shifted, self.clip_samples)
        except Exception:
            return waveform

    def safe_clip(self, waveform):
        peak = np.max(np.abs(waveform)) + 1e-8
        if peak > 1.0:
            waveform = waveform / peak
        return waveform


class DILDatasetInc(Dataset):
    def __init__(self, df, audio_folder, training=False, augment=False):
        """
        Args:
            df: DataFrame with filename and new_target columns.
            audio_folder: Root directory containing audio files.
            training: If True, enumerate all non-overlapping 4-second windows.
            augment: If True, apply waveform-level augmentations.
        """
        self.audio_folder = audio_folder
        self.training = training
        self.augment = augment and training
        self.augmenter = AudioAugmenter(
            sample_rate=config.sample_rate,
            clip_samples=config.clip_samples,
            enable=self.augment,
        )

        self.samples = []
        self.targets = []   # per-sample class index, for class-balanced sampling
        tgt_len = config.clip_samples
        min_len = tgt_len // 2

        for _, row in df.iterrows():
            file_name = row["filename"]
            class_idx = int(row["new_target"])
            label = to_one_hot(class_idx, config.classes_num_DIL)
            file_path = os.path.join(audio_folder, file_name)

            if training:
                info = librosa.get_duration(path=file_path)
                n_samples = int(info * config.sample_rate)
                n_segments = max(1, (n_samples + tgt_len - 1) // tgt_len)

                for k in range(n_segments):
                    start = k * tgt_len
                    seg_len = min(tgt_len, n_samples - start)

                    if n_segments > 1 and seg_len < min_len:
                        break

                    self.samples.append((file_path, label, class_idx, start, file_name))
                    self.targets.append(class_idx)
            else:
                self.samples.append((file_path, label, class_idx, 0, file_name))
                self.targets.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label, class_idx, start, file_name = self.samples[idx]

        if self.training:
            audio, _ = librosa.core.load(
                file_path,
                sr=config.sample_rate,
                mono=True,
                offset=start / config.sample_rate,
                duration=config.clip_samples / config.sample_rate,
            )
            waveform = crop_or_pad(audio, config.clip_samples)

            if self.augment:
                waveform = self.augmenter(waveform, class_idx)
        else:
            audio, _ = librosa.core.load(file_path, sr=config.sample_rate, mono=True)
            waveform = pad_sequence(audio, config.clip_samples)

        return waveform.astype(np.float32), label.astype(np.float32), file_name
