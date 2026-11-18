"""
Build a task7-compatible DIL dataset from a TAU Urban Acoustic Scenes
*Mobile* development set (2019 or 2020), using recording DEVICES as the
incremental domains D1 -> D2 -> D3 (the domain shift is the change of
device/microphone — same scenes, same locations, recorded simultaneously).

This is the device-mismatch axis. It reuses the same 10 acoustic-scene label
space as the cities setup, so `config_task7.py` (DCASE_DATASET=tau) and
`train_d1_base.py` work unchanged — only the data folder differs. Point
TAU_DATA_DIR at the output of this script (use a *separate* folder + checkpoint
dir from the cities run so they don't collide).

Optionally, `--unseen_devices` writes a separate `unseen_test.txt` manifest
(e.g. the 2020 simulated devices S4-S6, which are test-only). Those clips are
NOT used for training; they are held out for the unseen-domain routing /
rejection analysis (the on-benchmark analogue of the ESC-50 probe).

Input  : extracted TAU Mobile development dir with
             audio/ , meta.csv , evaluation_setup/fold1_train.csv ,
             evaluation_setup/fold1_evaluate.csv
Output : <out>/  (default <repo>/tau_mobile_data)
             D1-train/ D1-test/ ... D3-test/            (32 kHz mono wav)
             UNSEEN-test/                               (if --unseen_devices)
             evaluation_setup/development_train.txt
             evaluation_setup/development_test.txt
             evaluation_setup/unseen_test.txt           (if --unseen_devices)

Example (2020 Mobile: A/B/C as domains, S4-S6 held out as unseen)
-----------------------------------------------------------------
    python prepare_tau_mobile.py \
        --tau_root /data/TAU-urban-acoustic-scenes-2020-mobile-development \
        --devices a,b,c \
        --unseen_devices s4,s5,s6 \
        --out ./tau_mobile_data
"""
import argparse
import csv
import os

import librosa
import soundfile as sf
from tqdm import tqdm

# Must match config_task7.py (tau mode) TAU_SCENES order exactly.
TAU_SCENES = [
    'airport', 'bus', 'metro', 'metro_station', 'park', 'public_square',
    'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram',
]
SCENE_TO_IDX = {s: i for i, s in enumerate(TAU_SCENES)}
TARGET_SR = 32000


def device_of(filename):
    """`airport-lisbon-1000-40-a.wav` -> `a` ; `...-s4.wav` -> `s4`."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return stem.split('-')[-1].lower()


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
            next(reader, None)  # header
            for row in reader:
                if row:
                    split[row[0].strip()] = tag

    _read('fold1_train.csv', 'train')
    _read('fold1_evaluate.csv', 'test')
    return split


def _write(path, rows):
    with open(path, 'w', newline='') as f:
        csv.writer(f, delimiter='\t').writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tau_root', required=True,
                    help='Extracted TAU Mobile development directory.')
    ap.add_argument('--devices', default='a,b,c',
                    help='Comma-separated devices for D1,D2,D3,... (in order).')
    ap.add_argument('--unseen_devices', default=None,
                    help='Comma-separated held-out devices for unseen_test.txt '
                         '(e.g. s4,s5,s6). Not used for training.')
    ap.add_argument('--out', default=None,
                    help='Output dir (default <repo>/tau_mobile_data).')
    ap.add_argument('--limit_per_device', type=int, default=None,
                    help='Debug: cap files per device.')
    args = ap.parse_args()

    domain_devices = [d.strip().lower() for d in args.devices.split(',') if d.strip()]
    unseen_devices = [d.strip().lower() for d in args.unseen_devices.split(',')] \
        if args.unseen_devices else []
    assert len(domain_devices) >= 2, 'need at least 2 devices for D1,D2,...'

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = args.out or os.path.join(repo, 'tau_mobile_data')
    os.makedirs(os.path.join(out, 'evaluation_setup'), exist_ok=True)

    split = load_split(args.tau_root)

    # Bucket meta rows by device for the devices we care about.
    wanted = set(domain_devices) | set(unseen_devices)
    per_device = {d: [] for d in wanted}
    with open(os.path.join(args.tau_root, 'meta.csv')) as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            rel = row[0].strip()
            scene = row[1].strip() if len(row) > 1 else scene_of(rel)
            dev = device_of(rel)
            if dev in wanted and scene in SCENE_TO_IDX:
                per_device[dev].append((rel, scene))

    def _copy(rel, scene, sub, idx):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
        out_rel = f'{sub}/{idx:05d}.wav'
        audio, _ = librosa.load(os.path.join(args.tau_root, rel), sr=TARGET_SR, mono=True)
        sf.write(os.path.join(out, out_rel), audio, TARGET_SR)
        return [out_rel, scene, None, str(SCENE_TO_IDX[scene])]

    # --- domain devices -> D1, D2, D3 ... (train/test via official split) ---
    train_rows, test_rows = [], []
    for di, dev in enumerate(domain_devices, start=1):
        dom = f'D{di}'
        files = sorted(per_device[dev])
        if args.limit_per_device:
            files = files[:args.limit_per_device]
        if not files:
            raise RuntimeError(f'no files for device "{dev}" — check --tau_root/--devices')
        n_tr = n_te = 0
        for rel, scene in tqdm(files, desc=f'{dom}=dev-{dev}'):
            tag = split.get(rel, 'train')
            sub = f'{dom}-{tag}'
            idx = n_tr if tag == 'train' else n_te
            r = _copy(rel, scene, sub, idx)
            r[2] = dom
            (train_rows if tag == 'train' else test_rows).append(r)
            if tag == 'train':
                n_tr += 1
            else:
                n_te += 1
        print(f'  {dom} (device {dev}): {n_tr} train, {n_te} test')

    _write(os.path.join(out, 'evaluation_setup', 'development_train.txt'), train_rows)
    _write(os.path.join(out, 'evaluation_setup', 'development_test.txt'), test_rows)

    # --- unseen devices -> a single held-out test manifest (no training) ----
    if unseen_devices:
        unseen_rows = []
        n = 0
        for dev in unseen_devices:
            files = sorted(per_device.get(dev, []))
            if args.limit_per_device:
                files = files[:args.limit_per_device]
            for rel, scene in tqdm(files, desc=f'UNSEEN=dev-{dev}'):
                r = _copy(rel, scene, 'UNSEEN-test', n)
                r[2] = f'UNSEEN_{dev}'
                unseen_rows.append(r)
                n += 1
        _write(os.path.join(out, 'evaluation_setup', 'unseen_test.txt'), unseen_rows)
        print(f'  UNSEEN ({",".join(unseen_devices)}): {len(unseen_rows)} clips '
              f'(held out for routing/rejection analysis)')

    print(f'\nWrote {len(train_rows)} train / {len(test_rows)} test rows to {out}')
    print('Domains: ' + ', '.join(f'D{i+1}=device_{d}'
                                   for i, d in enumerate(domain_devices)))


if __name__ == '__main__':
    main()
