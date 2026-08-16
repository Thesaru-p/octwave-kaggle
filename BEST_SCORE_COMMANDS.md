# Best-Score Experiments

Kaggle copy-paste for the sequential matching + training pipeline. Public 40% score from the old two-fold B0 ensemble was `0.823007`; later runs reached ~0.89. Phase 1 does not retrain — wrap current best checkpoints with train/test duplicate matching.

Install before matching:

```python
!pip install ImageHash scipy
```

Data dir:

```text
/kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02
```

---

## Phase 1 — Train/test matching (do this first)

Calibrate thresholds and print coverage. Inspect agreement before submitting. If DINOv2 / internet fails, omit `--use-dino`.

```python
!python src/cluster_duplicates.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --out-dir /kaggle/working/outputs \
  --phash-threshold 4
```

Optional DINOv2 calibration (needs internet the first time):

```python
!python src/cluster_duplicates.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --out-dir /kaggle/working/outputs \
  --phash-threshold 4 \
  --use-dino \
  --dino-threshold 0.97
```

Submit: copy the train label when MD5 / phash / DINOv2 match with a unique label; otherwise use current checkpoints.

```python
!python src/predict_match.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --out-dir /kaggle/working/outputs \
  --checkpoints \
  /kaggle/working/outputs/checkpoints/efficientnet_b0_fold0_best.pt \
  /kaggle/working/outputs/checkpoints/efficientnet_b0_fold1_best.pt \
  --tta \
  --phash-threshold 4 \
  --out /kaggle/working/submission_match.csv \
  --diagnostics-out /kaggle/working/outputs/match_diagnostics.csv
```

Replace the checkpoint paths with whatever produced ~0.89. Add `--use-dino --dino-threshold 0.97` if calibration agreement at that cosine is ≥ 99%. If phash agreement at distance 4 is below 99%, drop `--phash-threshold` to 2.

STOP AND TEST: `duplicate_stats.json` (agreement, coverage, stratified vs grouped leakage) and match_source counts. Submit only after coverage is non-trivial.

---

## Phase 2 — Grouped folds (diagnostics are in Phase 1)

`cluster_duplicates.py` already prints fold-0 near-dup leakage for `StratifiedKFold` vs `StratifiedGroupKFold`. Train with:

```text
--groups-csv /kaggle/working/outputs/duplicate_clusters.csv
```

If that file is missing, training warns and falls back to stratified folds.

---

## Phase 3 / 4 — Cheap one-fold check (new crops + single imbalance correction)

Eval now resizes the full frame (no center crop). Train crop scale is `(0.85, 1.0)`. Sampler is kept; class-weighted CE and BCE `pos_weight` are removed.

```python
!python src/train.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --groups-csv /kaggle/working/outputs/duplicate_clusters.csv \
  --arch efficientnet_b0 \
  --image-size 224 \
  --epochs 12 \
  --batch-size 32 \
  --fold 0 \
  --out-dir /kaggle/working/outputs_b0
```

STOP AND TEST: honest grouped `val_macro_f1` plus `val_f1_0`…`val_f1_3`. The number should be lower than the old leaky val F1. Do not revert just because it dropped.

---

## Phase 5 — 5-fold ConvNeXt-Tiny @ 384 + probability TTA

```python
!python src/train.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --groups-csv /kaggle/working/outputs/duplicate_clusters.csv \
  --arch convnext_tiny \
  --image-size 384 \
  --epochs 18 \
  --batch-size 12 \
  --fold 0 \
  --out-dir /kaggle/working/outputs_convnext
```

Repeat `--fold 1` through `--fold 4`. `efficientnet_v2_s` at 384 is the alternate torchvision backbone. Checkpoints write `outputs_convnext/oof_convnext_tiny.csv` (appended per fold).

Probability ensemble + hflip + multi-scale TTA, then matching on top:

```python
!python src/predict_match.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --out-dir /kaggle/working/outputs \
  --checkpoints \
  /kaggle/working/outputs_convnext/checkpoints/convnext_tiny_fold0_best.pt \
  /kaggle/working/outputs_convnext/checkpoints/convnext_tiny_fold1_best.pt \
  /kaggle/working/outputs_convnext/checkpoints/convnext_tiny_fold2_best.pt \
  /kaggle/working/outputs_convnext/checkpoints/convnext_tiny_fold3_best.pt \
  /kaggle/working/outputs_convnext/checkpoints/convnext_tiny_fold4_best.pt \
  --tta \
  --tta-multiscale \
  --phash-threshold 4 \
  --out /kaggle/working/submission_match_convnext.csv \
  --diagnostics-out /kaggle/working/outputs/match_diagnostics.csv
```

`predict_ensemble.py` now averages softmax probabilities (same as `predict_multi_ensemble.py`) and accepts mixed `image_size` checkpoints.

---

## Phase 6 — OOF decision rule

```python
!python src/optimize_thresholds.py \
  --oof /kaggle/working/outputs_convnext/oof_convnext_tiny.csv \
  --out-json /kaggle/working/outputs/decision.json
```

If calibrated macro F1 beats raw argmax by more than noise, pass the JSON into matching:

```python
!python src/predict_match.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --out-dir /kaggle/working/outputs \
  --checkpoints \
  /kaggle/working/outputs_convnext/checkpoints/convnext_tiny_fold0_best.pt \
  /kaggle/working/outputs_convnext/checkpoints/convnext_tiny_fold1_best.pt \
  /kaggle/working/outputs_convnext/checkpoints/convnext_tiny_fold2_best.pt \
  /kaggle/working/outputs_convnext/checkpoints/convnext_tiny_fold3_best.pt \
  /kaggle/working/outputs_convnext/checkpoints/convnext_tiny_fold4_best.pt \
  --decision-json /kaggle/working/outputs/decision.json \
  --tta \
  --tta-multiscale \
  --phash-threshold 4 \
  --out /kaggle/working/submission_match_calibrated.csv \
  --diagnostics-out /kaggle/working/outputs/match_diagnostics.csv
```

Do not use hardcoded multipliers. The JSON is fitted on filename-aligned grouped OOF.

---

## Phase 7 — Pseudo-labels on unmatched test only

```python
!python src/predict_probs.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --checkpoints \
  /kaggle/working/outputs_convnext/checkpoints/convnext_tiny_fold0_best.pt \
  /kaggle/working/outputs_convnext/checkpoints/convnext_tiny_fold1_best.pt \
  /kaggle/working/outputs_convnext/checkpoints/convnext_tiny_fold2_best.pt \
  /kaggle/working/outputs_convnext/checkpoints/convnext_tiny_fold3_best.pt \
  /kaggle/working/outputs_convnext/checkpoints/convnext_tiny_fold4_best.pt \
  --decision-json /kaggle/working/outputs/decision.json \
  --tta \
  --tta-multiscale \
  --out /kaggle/working/outputs/submission_probs.csv \
  --probs-out /kaggle/working/outputs/test_probs.csv
```

```python
!python src/make_pseudo_train.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --probs /kaggle/working/outputs/test_probs.csv \
  --exclude-csv /kaggle/working/outputs/match_diagnostics.csv \
  --min-confidence 0.98 \
  --out /kaggle/working/outputs/train_pseudo.csv
```

If class 3 is empty, use looser per-class thresholds (still unmatched only):

```python
!python src/make_balanced_pseudo.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --probs /kaggle/working/outputs/test_probs.csv \
  --exclude-csv /kaggle/working/outputs/match_diagnostics.csv \
  --out /kaggle/working/outputs/train_pseudo_balanced.csv
```

Retrain fold 0 (or all 5 folds) with `--train-csv` pointing at the combined CSV.

---

## Phase 8 — Manual review of the unmatched tail

```python
!python src/make_review_html.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --probs /kaggle/working/outputs/test_probs.csv \
  --match-diagnostics /kaggle/working/outputs/match_diagnostics.csv \
  --only-unmatched \
  --limit 50 \
  --out /kaggle/working/outputs/review.html
```

Fill `manual_corrections_template.csv` (`filename,appearance`) and overlay:

```python
!python src/apply_corrections.py \
  --submission /kaggle/working/submission_match_calibrated.csv \
  --corrections /kaggle/working/outputs/manual_corrections.csv \
  --out /kaggle/working/submission_reviewed.csv
```

---

## Legacy recipes (pre-matching)

Two-fold EfficientNet-B0 TTA ensemble that scored `0.823007`:

```python
!python src/train.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --arch efficientnet_b0 \
  --image-size 224 \
  --epochs 12 \
  --batch-size 32 \
  --fold 0 \
  --out-dir /kaggle/working/outputs
```

`src/predict_knn.py` is still available but is not the Phase 1 matcher. Prefer `predict_match.py`.
