from torch.utils.data import Dataset
import pandas as pd
import os
import tqdm
import librosa
import numpy as np
import config_task7 as config


def to_one_hot(k, classes_num):
    target = np.zeros(classes_num)
    target[k] = 1
    return target


def pad_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0: max_len]


def crop_or_pad(x, target_len):
    """Return exactly target_len samples, zero-padding if shorter."""
    if len(x) < target_len:
        return np.concatenate((x, np.zeros(target_len - len(x))))
    return x[:target_len]


class DILDatasetInc(Dataset):
    def __init__(self, df, audio_folder, training=False):
        """
        Args:
            df (DataFrame): DataFrame with filename, new_target columns.
            audio_folder (str): Root directory containing audio files.
            training (bool): If True, enumerate all non-overlapping 4-second
                             windows per file (matching chunking.py behaviour).
                             If False, use the full file with first-window crop
                             for deterministic evaluation.
        """
        self.audio_folder = audio_folder
        self.training = training
        self.samples = []   # list of (file_path, label, start_sample, file_name)

        tgt_len  = config.clip_samples          # 4 s × 32 kHz = 128 000
        min_len  = tgt_len // 2                 # 2 s minimum for last chunk

        for _, row in df.iterrows():
            file_name  = row["filename"]
            label      = to_one_hot(int(row["new_target"]), 10)
            file_path  = os.path.join(audio_folder, file_name)

            if training:
                # Probe duration without loading the whole file
                info       = librosa.get_duration(path=file_path)
                n_samples  = int(info * config.sample_rate)
                n_segments = max(1, (n_samples + tgt_len - 1) // tgt_len)

                for k in range(n_segments):
                    start    = k * tgt_len
                    seg_len  = min(tgt_len, n_samples - start)
                    # Drop the last segment if it is shorter than min_len
                    # (only when there are multiple segments — matches chunking.py)
                    if n_segments > 1 and seg_len < min_len:
                        break
                    self.samples.append((file_path, label, start, file_name))
            else:
                # Evaluation: one entry per file, start at 0
                self.samples.append((file_path, label, 0, file_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label, start, file_name = self.samples[idx]
        if self.training:
            audio, _ = librosa.core.load(file_path, sr=config.sample_rate, mono=True,
                                         offset=start / config.sample_rate,
                                         duration=config.clip_samples / config.sample_rate)
            waveform = crop_or_pad(audio, config.clip_samples)
        else:
            audio, _ = librosa.core.load(file_path, sr=config.sample_rate, mono=True)
            waveform = pad_sequence(audio, config.clip_samples)  # pad if short, keep full length if longer
        return waveform, label, file_name


