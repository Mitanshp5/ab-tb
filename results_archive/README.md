# Archived results (pre-correction)

Snapshots kept so the corrected run can be diffed against everything that was
previously published. **Do not cite these numbers.**

| File | What it is |
|------|------------|
| `legacy_15ep_results.json` | The 15-epoch run committed to the repo (first machine, RTX 3050) |
| `legacy_pre_fix_results.json` | Snapshot of the live MongoDB/Vercel dashboard from the 100-epoch RTX 3060 run |

## Why these are not comparable to corrected results

1. **IoU was averaged per batch**, not accumulated over the dataset — a biased
   and unstable estimator on a 57-image test split.
2. **Dice returned 1.0 for any class absent from both prediction and target.**
   Rare classes scored "perfect" on every batch that did not contain them,
   producing impossible pairs such as `dice_RunDown = 0.90` next to
   `iou_RunDown = 0.376` (Dice is pinned to IoU by `2·IoU/(1+IoU)` = 0.546).
3. **Best-epoch selection ran on the test split.** The runner searched
   `["test", "val", "valid"]` in that order, so `test` became the validation
   set. The reported score was the maximum test score over all epochs, not a
   held-out measurement. The shipped `valid/` split was never used.
4. **A1 was not protocol-matched** — it evaluated an external 200-epoch
   production checkpoint while every other variant trained for 15 or 100 epochs.
5. **A4 was configuration-identical to the baseline**, so its row carried no
   information.
6. **A10 still applied ElasticTransform, CLAHE and CoarseDropout**, because
   those probabilities were hardcoded rather than read from config.
7. **5 of 13 variants never produced results**: A9 died on a host-RAM
   allocation, and A10–A13 failed with `name 'tqdm' is not defined`.

The corrected runner emits `legacy_mean_iou`, `legacy_mean_dice` and
`legacy_iou_*` / `legacy_dice_*` alongside the corrected values, computed with
the original formulas, so old and new can be compared directly within a single
new run.
