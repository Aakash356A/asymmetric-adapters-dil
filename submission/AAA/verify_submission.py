"""
Submission self-check — run this AS THE EVALUATOR would, to confirm the
standalone model definition (ResAdapt.py) reconstructs the trained model and
loads the provided per-domain weights cleanly.

Run from inside submission/AAA/ :

    python verify_submission.py

It will:
  1. import the standalone model (ResAdapt.py),
  2. build the model with the submitted adapter depths,
  3. load checkpoint/checkpoint_D2.pth and checkpoint/checkpoint_D3.pth
     with strict=True (fails loudly on any key mismatch),
  4. run a forward pass for every task on a dummy 4 s / 32 kHz waveform,
  5. run the domain-agnostic (ADIL) inference helper.

Exit code 0 and "ALL CHECKS PASSED" => the .py + .pth files are self-consistent.
"""

import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ResAdapt as M  # the standalone submitted model

NB_TASKS = 3
CKPT_DIR = os.path.join(HERE, "checkpoint")
CKPTS = {
    "D2": os.path.join(CKPT_DIR, "checkpoint_D2.pth"),
    "D3": os.path.join(CKPT_DIR, "checkpoint_D3.pth"),
}


def load_and_check(model, path, tag):
    print(f"\n[{tag}] loading {path}")
    state = torch.load(path, map_location="cpu")
    # Some training pipelines wrap the state dict; unwrap if needed.
    if isinstance(state, dict) and "model" in state and all(
        not k.startswith(("bn0", "conv_block", "fc", "adapters"))
        for k in list(state.keys())[:5]
    ):
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  MISSING keys ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"  UNEXPECTED keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    if not missing and not unexpected:
        print("  OK — exact match (strict load would also succeed)")
    return missing, unexpected


def main():
    print("=" * 70)
    print("DCASE 2026 Task 7 — submission self-check (ResAdapt)")
    print("=" * 70)
    print(f"adapter depths : {M.ADAPTER_DEPTHS}")
    print(f"sample rate    : {M.SAMPLE_RATE} Hz   mel bins: {M.MEL_BINS}   classes: {M.CLASSES_NUM}")

    model = M.build_model(nb_tasks=NB_TASKS, adapter_depths=M.ADAPTER_DEPTHS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params   : {n_params/1e6:.2f} M")

    problems = 0
    for tag, path in CKPTS.items():
        if not os.path.isfile(path):
            print(f"\n[{tag}] MISSING FILE: {path}")
            problems += 1
            continue
        missing, unexpected = load_and_check(model, path, tag)
        problems += len(missing) + len(unexpected)

    # Forward pass on a dummy 4 s clip at 32 kHz.
    model.eval()
    clip = M.SAMPLE_RATE * 4
    wav = torch.randn(2, clip)
    print(f"\nforward pass on dummy input {tuple(wav.shape)}:")
    with torch.no_grad():
        for t in range(NB_TASKS):
            out = model(wav, task=t)
            print(f"  task {t}: logits {tuple(out.shape)}")
            if out.shape != (2, M.CLASSES_NUM):
                problems += 1

        pred = M.predict_domain_agnostic(model, wav, nb_tasks=NB_TASKS)
        print(f"  ADIL domain-agnostic prediction: {pred.tolist()}")
        if pred.shape != (2,):
            problems += 1

    print("\n" + "=" * 70)
    if problems == 0:
        print("ALL CHECKS PASSED — .py and .pth files are self-consistent.")
    else:
        print(f"FAILED — {problems} problem(s) found above.")
    print("=" * 70)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
