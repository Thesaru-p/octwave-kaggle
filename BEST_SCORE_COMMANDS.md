# Best-Score Experiments

This file contains the Kaggle commands for the strongest experiments supported by the repo.

## Current Best Reproduction

This reproduced the best public score so far:

```text
0.823007
```

Train EfficientNet-B0 fold 0:

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

Train EfficientNet-B0 fold 1:

```python
!python src/train.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --arch efficientnet_b0 \
  --image-size 224 \
  --epochs 12 \
  --batch-size 32 \
  --fold 1 \
  --out-dir /kaggle/working/outputs
```

Create the two-fold ensemble:

```python
!python src/predict_ensemble.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --checkpoints \
  /kaggle/working/outputs/checkpoints/efficientnet_b0_fold0_best.pt \
  /kaggle/working/outputs/checkpoints/efficientnet_b0_fold1_best.pt \
  --tta \
  --out /kaggle/working/submission_ensemble.csv
```

## Next Best Attempt

Train EfficientNet-B3:

```python
!python src/train.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --arch efficientnet_b3 \
  --image-size 300 \
  --epochs 16 \
  --batch-size 16 \
  --fold 0 \
  --out-dir /kaggle/working/outputs
```

Create a multi-size weighted ensemble using B0 folds plus B3:

```python
!python src/predict_multi_ensemble.py \
  --data-dir /kaggle/input/competitions/oct-wave-3-0-kaggle-challenge-02 \
  --checkpoints \
  /kaggle/working/outputs/checkpoints/efficientnet_b0_fold0_best.pt \
  /kaggle/working/outputs/checkpoints/efficientnet_b0_fold1_best.pt \
  /kaggle/working/outputs/checkpoints/efficientnet_b3_fold0_best.pt \
  --weights 1.0 1.0 0.8 \
  --tta \
  --out /kaggle/working/submission_multi_ensemble.csv
```

If the B3 single model scores poorly, reduce its ensemble weight or remove it.
