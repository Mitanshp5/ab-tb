# CON-SOL-E 5.0 — Ablation Study
DINOv2 ViT-B/14 + Custom Dense FPN-UNet Decoder
4-class surface defect segmentation (Background / Dust / RunDown / Scratch)

---

## Architecture
- **Encoder**: DINOv2 `dinov2_vitb14` (~86M params, frozen during ablation)
- **Decoder**: Custom Dense FPN-UNet, skip_layers=[3,7,11], channels=[256,128,64]
- **Loss**: 0.5×DiceLoss + 0.5×FocalLoss (α=0.25, γ=2.0), class_weights=[0.3, 3.0, 2.0, 2.0]
- **Optimizer**: AdamW, lr=5e-5, weight_decay=0.01

> The model definition (`models/`, `loss/`) is byte-identical to the upstream
> CON-SOL-E vision system. Nothing in this repository alters the architecture;
> the study only varies configuration, and the runner only drives training and
> records metrics.

---

## Hardware Requirements
- GPU: ≥6GB VRAM (the full study is run on an RTX 3060 12GB)
- RAM: ≥16GB
- Storage: ~3GB free (model weights + dataset), plus ~1.2GB per variant for checkpoints

---

## Setup

### 1. Install dependencies

Install the **CUDA** build of PyTorch explicitly — the default PyPI wheel can
resolve to a CPU-only build, which turns a 6-hour study into a multi-day one:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Verify the GPU is visible before starting:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. Download dataset
Dataset: **Paint_Defect v7** from Roboflow
URL: https://universe.roboflow.com/shyam-sojitra-i8tgx/paint_defect-hncxs
License: CC BY 4.0

Download in **YOLOv8 format** (polygon segmentation, not bounding box) and place at:
```
data/data/
├── train/
│   ├── images/   (1081 images)
│   └── labels/
├── valid/
│   ├── images/   (56 images)
│   └── labels/
└── test/
    ├── images/   (57 images)
    └── labels/
```

### 3. Configure cloud sync (optional)
```bash
cp .env.example .env        # then edit .env with your own Atlas credentials
python upload_to_cloud.py --status
```

---

## Ablation protocol

These are the rules the runner enforces. They matter for reading the numbers.

**Splits.** `train/` and `valid/` are pooled, and a **stratified, seeded 10%**
of that pool is held out as the validation split used to pick the best epoch.
`test/` is scored exactly once, at the end, and never influences checkpoint
selection.

```
train  1023 images   val  114 images   test  57 images     (seed 42)
```

Stratification is by the set of classes present in each image, so rare classes
land in validation proportionally. The shipped `valid/` directory contains
**zero** RunDown annotations; the stratified split contains 16 RunDown images.

**Controlled baseline.** A1 trains fresh under exactly the same epoch budget,
seed, and schedule as every other variant, so each row is a one-factor change
against it. The deployed 200-epoch production checkpoint is not comparable to a
100-epoch ablation run and is reported separately:

```bash
python run_ablation.py --mode full --epochs 100 --with-pretrained-baseline
```

**Isolation.** Each variant runs in its own subprocess. A CUDA OOM, a host-RAM
OOM, or a hard crash costs that one variant instead of the whole study, and all
of its memory returns to the OS before the next variant starts. Use
`--no-isolate` to run everything in one process.

**Schedule.** Linear warmup for `warmup_epochs` (3), then a single cosine decay
to 1% of the base LR. The previous `CosineAnnealingWarmRestarts(T_0=n//3)` reset
the LR to maximum at epochs 33, 66 and 99, which is why best epochs previously
landed anywhere between 15 and 83. Pass `--scheduler warm_restarts` to reproduce
the old behaviour.

---

## Running the study

### One-click Windows runner
```cmd
run_full_ablation.bat            :: defaults to 100 epochs, 6 dataloader workers
run_full_ablation.bat 100 8      :: epochs, workers
```

### Manual
```bash
# Full study. --num-workers is the single biggest throughput lever: with 0, all
# 1081 JPEGs are decoded on the main thread every epoch.
python run_ablation.py --mode full --epochs 100 --num-workers 6

# Specific variants only
python run_ablation.py --mode full --epochs 100 --variants A5_dice_only A8_small_decoder

# Inspect the splits without training anything
python run_ablation.py --print-splits
```

### Useful flags

| Flag | Effect |
|------|--------|
| `--num-workers N` | DataLoader workers. 4–6 is a large speedup on the RTX 3060 box |
| `--cache-dir PATH` | Cache resized image/mask arrays to skip repeated JPEG decoding |
| `--image-size N` | Input side; must be a multiple of 14 for ViT-B/14 (default 518) |
| `--resize-mode letterbox` | Preserve aspect ratio instead of squashing 1440×1080 to square |
| `--scheduler warm_restarts` | Reproduce the pre-fix LR schedule |
| `--no-isolate` | Run all variants in one process |
| `--no-resume` | Ignore existing checkpoints and results |
| `--no-sync` | Disable MongoDB Atlas sync |
| `--data-root PATH` | Point at a dataset outside `./data/data` |

Re-running the same command **resumes** from the last completed epoch of each
variant, so an interrupted study can simply be restarted.

---

## Ablation Variants

| ID | What changes |
|----|--------------|
| `A1_full_model` | Baseline — all 3 skip layers [3,7,11], combined loss, frozen encoder |
| `A2_single_scale` | `skip_layers=[11]` only |
| `A3_two_scale` | `skip_layers=[7,11]` |
| `A4_partial_unfreeze` | Last 4 ViT blocks fine-tuned from epoch 5 |
| `A5_dice_only` | Dice loss only, no Focal |
| `A6_focal_only` | Focal loss only, no Dice |
| `A7_no_class_weights` | Uniform class weights [1,1,1,1] |
| `A8_small_decoder` | Decoder channels [128,64,32] |
| `A9_vitb_unfrozen` | Whole encoder unfrozen from epoch 5 (batch 4 × accum 2) |
| `A10_no_augmentation` | Every augmentation disabled |
| `A11_high_gamma` | Focal γ=4.0 |
| `A12_lower_lr` | lr=1e-5 |
| `A13_higher_lr` | lr=1e-4 |

A1 / A4 / A9 form the encoder-freezing axis: frozen throughout → last 4 blocks →
whole encoder.

---

## Reading the metrics

Metrics are accumulated as a dataset-level confusion matrix and reduced once, at
the end (the Cityscapes/ADE20K protocol). Every variant emits ~70 metric keys.
The ones worth knowing:

| Key | Meaning |
|-----|---------|
| `mean_iou` | Mean IoU over all 4 classes |
| `mean_iou_defect` | Mean IoU over defect classes only (background excluded) |
| `mean_iou_gt_present` | Mean over classes that actually have ground truth in the split |
| `frequency_weighted_iou` | IoU weighted by each class's share of GT pixels |
| `iou_<class>`, `dice_<class>` | Per class; **NaN** when the class is absent from both prediction and GT |
| `precision_<class>`, `recall_<class>`, `f1_<class>` | Per class |
| `support_px_<class>` | Ground-truth pixel count backing that class's score |
| `tp_/fp_/fn_<class>` | Raw confusion counts |
| `val_*` | The same metrics on the validation split |
| `legacy_*` | The previous per-batch numbers, for comparison only |

### Two things to keep in mind when interpreting results

**1. RunDown is barely measurable on this test split.** It has only 4 annotated
polygons across 57 test images. Its IoU is a high-variance estimate, and since
`mean_iou` averages 4 classes, a class you cannot measure drags the headline
number down by a large margin. `mean_iou_defect` and the per-class
`support_px_*` counts are reported so this is visible rather than hidden.

**2. `legacy_*` values are not comparable to the corrected ones.** The previous
metric implementation averaged per-batch IoU and Dice, and its Dice returned
`smooth/smooth = 1.0` for any class absent from both prediction and target — so
rare classes scored "perfect" on every batch that did not contain them. That is
how the earlier dashboard published `dice_RunDown = 0.90` next to
`iou_RunDown = 0.376`, an impossible pair (Dice is pinned to IoU by
`Dice = 2·IoU/(1+IoU)`, so 0.376 → 0.546). The corrected values are lower and
correct. Run `python utils/metrics.py` to execute the self-tests that pin this.

---

## Cloud sync and dashboard

Results sync to MongoDB Atlas in real time as each variant finishes, with a
per-epoch heartbeat. If the connection drops, documents are written to
`offline_sync_queue.json` and uploaded automatically on the next successful
call — no manual re-upload needed.

```bash
python upload_to_cloud.py --status        # connection, queue depth, last heartbeat
python upload_to_cloud.py --flush         # push anything stuck in the offline queue
python upload_to_cloud.py --list          # list variant docs currently in Atlas
python upload_to_cloud.py --purge-stale   # remove docs not part of the current study
python upload_to_cloud.py --dry-run       # preview documents without network calls
```

Every document carries a `run_id`, so results from different runs and protocols
can be told apart instead of silently overwriting each other. `--purge-stale`
removes leftovers from ad-hoc reruns (e.g. `A1_fresh_local`,
`A2_single_scale_fixed`), which are upserted by variant name and would otherwise
sit on the dashboard as if they were study rows.

### Live dashboard
Deployed on Vercel. Set `MONGODB_URI` and `MONGODB_TARGET` as Environment
Variables in the Vercel project settings.

### Local monitoring
```bash
python monitor_server.py --port 8080
# open http://<device-ip>:8080 from any device on the same network
```

---

## Security

`.env` is gitignored and must never be committed. `.env.example` is a public
template — do not put real credentials in it.

> **Note:** working Atlas credentials were previously committed to
> `.env.example` and `README.md` in this public repository. That password must
> be rotated in MongoDB Atlas; removing it from the current files does not
> remove it from git history.

---

## Fixes applied to the study runner

Architecture files were not touched by any of these.

- **`utils/metrics.py`** — metrics now accumulate a dataset-level confusion
  matrix instead of averaging per-batch scores; Dice no longer reports 1.0 for
  classes absent from both prediction and target. Adds defect-only and
  GT-present means, per-class precision/recall/F1, per-class support and raw
  TP/FP/FN. Old values retained under `legacy_*`. Self-tests in `__main__`.
- **`run_ablation.py`** — validation split carved from the training pool so the
  test split no longer drives checkpoint selection; A1 trained under the study
  protocol instead of loading an external checkpoint; A4 replaced (it was
  configuration-identical to the baseline and carried no information); warmup
  implemented and cosine schedule made monotonic; per-variant subprocess
  isolation; encoder hook/reference-cycle teardown between variants; trailing
  gradient-accumulation cycle no longer dropped; `run_id` on all records.
- **`data/augmentation.py`** — every augmentation probability now read from
  config. `elastic_transform_p`, `clahe` and the CoarseDropout probability were
  hardcoded, so `A10_no_augmentation` was still applying ElasticTransform, CLAHE
  and CoarseDropout on every batch. Defaults are unchanged for all other variants.
- **`data/dataset.py`** — optional aspect-preserving `letterbox` resize and an
  optional on-disk cache of resized image/mask pairs.
- **`upload_to_cloud.py`** — one shared `MongoClient` instead of a new
  (never-closed) client per call; offline queue made atomic; added `--status`,
  `--flush`, `--list`, `--purge-stale`.
- **`api/index.py`** — client reused across warm invocations; new metric columns;
  stale documents flagged; per-class support shown next to each score.
- **`run_full_ablation.bat`** — installs the CUDA PyTorch wheel explicitly and
  verifies `torch.cuda.is_available()` before starting.

### Previously applied
- `models/encoder.py`: variable `skip_layers` assigns keys from the END of the
  list; missing keys filled with zero tensors (fixes the `'mid'` / `'deep'`
  KeyError that failed A2 and A3).
- `data/dataset.py`: Windows case-insensitive glob deduplication.
- `evaluate.py`: 4-class names, handles the `valid/` directory.

---

## Known limitations

- **RunDown support in the test split is 4 polygons.** No code change fixes
  this; it needs more annotated data. Until then, treat `iou_RunDown` as
  indicative only and prefer `mean_iou_defect` plus per-class support.
- **Images are 1440×1080 but the model input is 518×518**, so the default
  `stretch` mode distorts the aspect ratio by 1.33× and thin scratches approach
  sub-pixel width. `--resize-mode letterbox` and a larger `--image-size` both
  target this, at a VRAM and time cost.
- Test-set size (57 images) makes all differences between close variants
  statistically weak. Differences under a few IoU points should not be read as
  meaningful.
