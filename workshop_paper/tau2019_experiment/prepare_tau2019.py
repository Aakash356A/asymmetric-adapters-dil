"""
Build a task7-compatible DIL dataset from the TAU Urban Acoustic Scenes 2019
development set, using three cities as domains D1 -> D2 -> D3.

Input  : the extracted TAU2019 development download, i.e. a directory that
         contains
             audio/                 (all *.wav)
             meta.csv               (filename, scene_label, identifier, source)
             evaluation_setup/fold1_train.csv
             evaluation_setup/fold1_evaluate.csv
Output : <out>/  (default <repo>/tau_data)
             D1-train/ D1-test/ D2-train/ ... D3-test/     (32 kHz mono wav)
             evaluation_setup/development_train.txt
             evaluation_setup/development_test.txt
         with rows  "<Dk>-train/<idx>.wav \t <scene> \t D<k> \t <label_idx>"
         exactly matching the task7 manifest schema.

The official fold1 train/evaluate split is used per file so that no
recording location leaks between train and test. Each city becomes one
domain, so the domain shift is the change of city (as in the ADIL paper).

Example
-------
    python prepare_tau2019.py \
        --tau_root /data/TAU-urban-acoustic-scenes-2019-development \
        --cities lisbon,lyon,prague \
        --out /home/aakashd/dcase_task7/tau_data
"""
import argparse
import csv
import os

import librosa
import soundfile as sf
from tqdm import tqdm

# Must match config_task7.py (shadow) TAU_SCENES order exactly.
TAU_SCENES = [
    'airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
    'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram',
]
SCENE_TO_IDX = {s: i for i, s in enumerate(TAU_SCENES)}
TARGET_SR = 32000


def city_of(filename):
    """`metro_station-lisbon-1-2-a.wav` -> `lisbon` (scene may hold '_')."""
    base = os.path.basename(filename)
    return base.split('-')[1]


def scene_of(filename):
    return os.path.basename(filename).split('-')[0]


def load_split(tau_root):
    """Return {relative_audiopath: 'train'|'test'} from fold1 csv files."""
    setup = os.path.join(tau_root, 'evaluation_setup')
    split = {}

    def _read(csv_name, tag):
        path = os.path.join(setup, csv_name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f'missing split file: {path}')
        with open(path) as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                split[row[0].strip()] = tag

    _read('fold1_train.csv', 'train')
    # DCASE2019 Task1A names the val/test split "evaluate".
    _read('fold1_evaluate.csv', 'test')
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tau_root', required=True,
                    help='Extracted TAU2019 development directory.')
    ap.add_argument('--cities', default='lisbon,lyon,prague',
                    help='Comma-separated city names for D1,D2,D3 (in order).')
    ap.add_argument('--out', default=None,
                    help='Output dir (default <repo>/tau_data).')
    ap.add_argument('--limit_per_city', type=int, default=None,
                    help='Debug: cap files per city.')
    args = ap.parse_args()

    cities = [c.strip().lower() for c in args.cities.split(',') if c.strip()]
    assert len(cities) == 3, 'need exactly 3 cities for D1,D2,D3'

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = args.out or os.path.join(repo, 'tau_data')
    os.makedirs(os.path.join(out, 'evaluation_setup'), exist_ok=True)

    split = load_split(args.tau_root)

    # Read meta.csv -> list of (audio_relpath, scene) for the chosen cities.
    meta_path = os.path.join(args.tau_root, 'meta.csv')
    per_city = {c: [] for c in cities}
    with open(meta_path) as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            rel = row[0].strip()            # e.g. audio/airport-lisbon-...-a.wav
            scene = row[1].strip() if len(row) > 1 else scene_of(rel)
            c = city_of(rel)
            if c in per_city and scene in SCENE_TO_IDX:
                per_city[c].append((rel, scene))

    train_rows, test_rows = [], []
    for di, city in enumerate(cities, start=1):
        dom = f'D{di}'
        files = sorted(per_city[city])
        if args.limit_per_city:
            files = files[:args.limit_per_city]
        if not files:
            raise RuntimeError(f'no files found for city "{city}" — check --tau_root/--cities')
        n_train = n_test = 0
        for rel, scene in tqdm(files, desc=f'{dom}={city}'):
            tag = split.get(rel, 'train')      # default to train if unlisted
            sub = f'{dom}-{tag}'
            os.makedirs(os.path.join(out, sub), exist_ok=True)
            idx = n_train if tag == 'train' else n_test
            out_rel = f'{sub}/{idx:05d}.wav'
            # resample to 32 kHz mono and write
            src = os.path.join(args.tau_root, rel)
            audio, _ = librosa.load(src, sr=TARGET_SR, mono=True)
            sf.write(os.path.join(out, out_rel), audio, TARGET_SR)
            row = [out_rel, scene, dom, str(SCENE_TO_IDX[scene])]
            (train_rows if tag == 'train' else test_rows).append(row)
            if tag == 'train':
                n_train += 1
            else:
                n_test += 1
        print(f'  {dom} ({city}): {n_train} train, {n_test} test')

    def _write(path, rows):
        with open(path, 'w', newline='') as f:
            w = csv.writer(f, delimiter='\t')
            w.writerows(rows)

    _write(os.path.join(out, 'evaluation_setup', 'development_train.txt'), train_rows)
    _write(os.path.join(out, 'evaluation_setup', 'development_test.txt'), test_rows)
    print(f'\nWrote {len(train_rows)} train / {len(test_rows)} test rows to {out}')
    print('Domains: ' + ', '.join(f'D{i+1}={c}' for i, c in enumerate(cities)))


if __name__ == '__main__':
    main()
