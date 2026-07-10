"""
Diagnose the D3 airport/park collapse in the TAU cities DIL run.

For a chosen domain (default D3) it evaluates the FINAL trained model in two
modes and prints them side by side, per class:

  * domain-AWARE   : force that domain's own head (task=1 for D2, task=2 for
                     D3). No routing. This is "how good is the D3 head itself".
  * domain-AGNOSTIC: the ADIL entropy router (softmax -> per-task entropy ->
                     min-entropy head), exactly as baseline_DIL_task7.py. This
                     reproduces the number you already saw (D3 = 67.39).

It also reports, per class, the fraction of samples the router sent to
D1 / D2 / D3. That tells us WHERE airport/park go.

Interpretation
--------------
  * If airport/park recover under domain-AWARE (e.g. 60-80%) but are near-zero
    under domain-AGNOSTIC -> it is a ROUTING failure (those clips are routed to
    the wrong head). Great result: concrete second-dataset evidence for the
    paper's routing-failure thesis.
  * If airport/park stay near-zero even domain-AWARE -> the D3 head/data for
    those classes is the problem (undertrained / imbalance / label issue).
    A bug to fix, not a result.

Run on the server (data + checkpoints live there):
    DCASE_DATASET=tau TAU_DATA_DIR=$REPO/tau_data \
    python workshop_paper/tau2019_experiment/diagnose_routing.py \
        --checkpoint $REPO/checkpoints/TAU_run/checkpoint_D3.pth \
        --domain D3 --adapter_depths 1,3 --cuda
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
os.environ.setdefault('DCASE_DATASET', 'tau')
sys.path.insert(0, os.path.join(_REPO, 'code', 'utils'))
sys.path.insert(0, os.path.join(_REPO, 'code', 'baseline'))

import config_task7 as config                       # tau mode via DCASE_DATASET
from domain_net_arch import MCnn14ResDeepClf, set_adapter_depths
from datasetfactory_task7 import DILDatasetInc as DILDataset

DOMAIN_TO_TASK = {'D1': 0, 'D2': 1, 'D3': 2, 'D4': 3}
EPS = torch.finfo(torch.float32).tiny


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True,
                    help='Final model state_dict, e.g. checkpoints/TAU_run/checkpoint_D4.pth')
    ap.add_argument('--domain', default='D3', choices=['D2', 'D3', 'D4'])
    ap.add_argument('--nb_tasks', type=int, default=None,
                    help='Total domains incl. D1 (default: derive from manifest, '
                         'e.g. 4 for the ADIL D1..D4 protocol).')
    ap.add_argument('--adapter_depths', default='1,3')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--cuda', action='store_true')
    args = ap.parse_args()

    device = ('cuda' if (args.cuda and torch.cuda.is_available())
              else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print('device:', device, '| domain:', args.domain)

    depths = [int(d) for d in args.adapter_depths.split(',') if d.strip()]
    set_adapter_depths({t + 1: depths[t] for t in range(len(depths))})

    nb_tasks = args.nb_tasks or int(config.df_DIL_dev_train['domain'].nunique())

    model = MCnn14ResDeepClf(
        config.sample_rate, config.window_size, config.hop_size,
        config.mel_bins, config.fmin, config.fmax,
        config.classes_num_DIL, nb_tasks=nb_tasks)
    inc = model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'),
                                strict=False)
    if inc.missing_keys:
        print(f'  [warn] {len(inc.missing_keys)} missing keys on load')
    if inc.unexpected_keys:
        print(f'  [warn] {len(inc.unexpected_keys)} unexpected keys on load')
    model.eval().to(device)

    forced_task = DOMAIN_TO_TASK[args.domain]
    n_tasks = nb_tasks

    df = config.df_DIL_dev_test[config.df_DIL_dev_test['domain'] == args.domain]
    if len(df) == 0:
        raise RuntimeError(f'no {args.domain} rows in test manifest — check TAU_DATA_DIR')
    ds = DILDataset(df, config.audio_folder_DIL, training=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False,
                                         num_workers=args.num_workers)

    C = config.classes_num_DIL
    names = [None] * C
    for name, idx in config.dict_class_labels.items():
        names[idx] = name

    # per-class tallies
    n = np.zeros(C, int)
    aware_correct = np.zeros(C, int)     # this domain's own head (forced)
    agn_correct = np.zeros(C, int)       # entropy router (reproduces reported)
    unadapt_correct = np.zeros(C, int)   # D1 head (no adaptation) reference
    route_counts = np.zeros((C, n_tasks), int)   # [class, routed_task]

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.float().to(device)
            y = int(torch.argmax(targets, dim=-1))
            probs = torch.stack(
                [F.softmax(model(inputs, task=t), dim=1) for t in range(n_tasks)],
                dim=0)                                    # (T, 1, C)
            ent = -torch.sum(probs * torch.log(probs + EPS), dim=-1)  # (T, 1)
            routed = int(torch.argmin(ent, dim=0))
            agn_pred = int(torch.argmax(probs[routed, 0]))
            aware_pred = int(torch.argmax(probs[forced_task, 0]))
            unadapt_pred = int(torch.argmax(probs[0, 0]))   # D1 head

            n[y] += 1
            aware_correct[y] += (aware_pred == y)
            agn_correct[y] += (agn_pred == y)
            unadapt_correct[y] += (unadapt_pred == y)
            route_counts[y, routed] += 1

    # ---- report ----
    print(f'\n{args.domain} test: {int(n.sum())} clips  '
          f'(aware = {args.domain} head forced, agnos = entropy router, '
          f'unadapt = D1 head)\n')
    hdr = (f'{"class":18} {"n":>4} {"aware%":>7} {"agnos%":>7} {"unadapt%":>8}'
           f'   routed D1/D2/D3 (%)')
    print(hdr)
    print('-' * len(hdr))
    active = [c for c in range(C) if n[c] > 0]
    for c in active:
        aw = 100 * aware_correct[c] / n[c]
        ag = 100 * agn_correct[c] / n[c]
        un = 100 * unadapt_correct[c] / n[c]
        rf = 100 * route_counts[c] / n[c]
        flag = '   <== collapse' if (ag < 25 and aw - ag > 25) else ''
        print(f'{names[c]:18} {n[c]:4d} {aw:7.1f} {ag:7.1f} {un:8.1f}   '
              f'{rf[0]:4.0f}/{rf[1]:4.0f}/{rf[2]:4.0f}{flag}')
    tot = n[active].sum()
    print('-' * len(hdr))
    print(f'{"OVERALL":18} {tot:4d} '
          f'{100*aware_correct[active].sum()/tot:7.1f} '
          f'{100*agn_correct[active].sum()/tot:7.1f} '
          f'{100*unadapt_correct[active].sum()/tot:8.1f}')
    print('\nReading: if a class is high under aware% but near-zero under '
          'agnos%, it is a ROUTING failure (see where it routes). If low under '
          'both, the head/data for that class is the problem.')


if __name__ == '__main__':
    main()
