# Publishing strategy — DCASE 2026 Task 7 workshop paper

Honest, condensed guidance. Context: system ranked **5th** in the challenge.

## Bottom line
- **Worth publishing? Yes.** Rank does not gate a DCASE *workshop* paper — it's a
  community venue that welcomes challenge-system papers, most of which are not
  from the #1 team. The bar is "sound, useful, honest," not "SOTA."
- **Reimplement the winners? No** — not for this deadline. Cite and position
  against their technical reports instead.
- **The catch:** as a bare system report ("beat baseline, came 5th") this is
  unremarkable. The **analysis is the contribution** — do not cut it.

## The one thing that can sink it
**Overclaiming SOTA.** Rankings are public; reviewers know. Never claim/imply
"best." Claim only what's true. This is the main avoidable rejection cause.

## Where the actual contribution lies (lead with these)
1. **Method insight** — asymmetric per-domain adapter depth; *why* more capacity
   on the harder domain helps, and why a shared unfrozen backbone couples domains.
2. **Cautionary/negative result** — learned GAP-MLP router overfits; entropy
   routing confidently **misroutes unseen domains** (ESC-50 probe). Winners'
   reports won't have this.
3. **Generalization** — second benchmark (TAU2019 cities, device-mobile later)
   shows it's not a one-dataset artifact.

## Framing to use
> A forgetting-free, parameter-efficient DIL method + a cautionary study of
> domain routing under unseen domains, validated on two benchmarks.

- No SOTA claims. Be explicit it ranked 5th if asked; don't hide it.
- One sentence positioning vs. top systems ("unlike the top systems using [X],
  we focus on forgetting-free adaptation with a frozen backbone and analyze
  routing failure modes"). Cite their technical reports.

## "Beat the winners on clean data" idea
Genuinely interesting — hypothesis: top systems partly exploit noisy
challenge-data quirks; on clean/independent data the ranking may not hold.
**But:** high effort, high risk (might confirm they're better), needs faithful
reimplementation. **Not a July-11 task.** Park it as explicit *future work* and
pursue as a larger extended/journal paper if motivated.

## Reality check
Solid, respectable, real peer-reviewed credential — valuable especially as an
independent researcher. **Not** a landmark paper; don't frame or judge it as one.

## Action checklist (for the workshop deadline)
- [ ] Keep frame: method + routing/OOD analysis + 2 benchmarks. No SOTA claims.
- [ ] Add 1 sentence + citations positioning vs. top-N technical reports.
- [ ] Land the TAU2019 (cities) generalization result.
- [ ] Add "beat winners on clean data" as a future-work line.
- [ ] Final proofread; confirm 4+1 page limit still holds.
