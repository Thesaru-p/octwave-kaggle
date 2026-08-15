# Tom & Jerry Image Classification

Kaggle-style image classification project for predicting whether Tom, Jerry, both, or neither appear in a frame.

## Local Data

The dataset should be placed in:

```text
data/
  images/
  train.csv
  test.csv
  sample_submission.csv
```

The `data/` folder is ignored by Git because it contains competition files and large images.

## Why This Approach

The training labels are imbalanced:

```text
0 neither:      368
1 Tom only:    1252
2 Jerry only:   841
3 both:         219
```

The competition metric is Macro F1, so rare classes matter as much as common classes. The best practical technique here is transfer learning with imbalance handling:

- pretrained `convnext_tiny` or `efficientnet_b3`
- stratified validation split
- weighted cross-entropy
- weighted random sampler
- auxiliary Tom/Jerry presence prediction head
- checkpoint selection by validation Macro F1
- test-time augmentation for submission predictions

## Setup

Use Python 3.11, not Python 3.14.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-torch-cu128.txt --index-url https://download.pytorch.org/whl/cu128
```

The CUDA PyTorch download is large. If it fails or is too slow, run the same project in Kaggle/Colab with GPU enabled.

## Check The Data

```powershell
.\.venv\Scripts\python.exe src\eda.py
```

## Train A Strong Single Model

Start with ConvNeXt-Tiny:

```powershell
.\.venv\Scripts\python.exe src\train.py --arch convnext_tiny --image-size 384 --epochs 18 --batch-size 12 --fold 0
```

If GPU memory is low, reduce `--batch-size` to `8`. If training is too slow, use:

```powershell
.\.venv\Scripts\python.exe src\train.py --arch efficientnet_b0 --image-size 224 --epochs 12 --batch-size 32 --fold 0
```

## Create Submission

```powershell
.\.venv\Scripts\python.exe src\predict.py --checkpoint outputs\checkpoints\convnext_tiny_fold0_best.pt --tta --out outputs\submission.csv
```

## Stronger Version If You Have Time

Train multiple folds/models, then average their logits:

```powershell
.\.venv\Scripts\python.exe src\train.py --arch convnext_tiny --image-size 384 --epochs 18 --batch-size 12 --fold 0
.\.venv\Scripts\python.exe src\train.py --arch convnext_tiny --image-size 384 --epochs 18 --batch-size 12 --fold 1
.\.venv\Scripts\python.exe src\predict_ensemble.py --tta --checkpoints outputs\checkpoints\convnext_tiny_fold0_best.pt outputs\checkpoints\convnext_tiny_fold1_best.pt --out outputs\submission_ensemble.csv
```

For a one-day competition, one well-trained `convnext_tiny` at `384x384` with TTA is the best first serious attempt.
