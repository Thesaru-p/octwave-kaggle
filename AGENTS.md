# AGENTS.md

## Cursor Cloud specific instructions

This is a Python (PyTorch) Kaggle-style image-classification project. The "application" is a
set of CLI scripts under `src/` for a train → predict pipeline. There is no web UI, no test
suite, and no lint configuration.

### Environment
- Dependencies live in a virtualenv at `.venv`. Always run scripts with `.venv/bin/python`.
- The cloud VM has **no GPU**, so PyTorch is installed as the **CPU** build
  (`torch ... --index-url https://download.pytorch.org/whl/cpu`), not the CUDA wheels in
  `requirements-torch-cu128.txt` (that file is for local CUDA machines only). `torch.cuda.is_available()`
  is `False` here; scripts fall back to CPU automatically.
- Python is 3.12 in the VM (the README's 3.11 suggestion is a local guideline, not required here).

### Running scripts
- Run from the repo root, e.g. `.venv/bin/python src/train.py ...`. The scripts use
  `from common import ...`; this resolves because Python puts the script's own dir (`src/`) on
  the path when invoked as `src/train.py`. Do not `cd src` and rely on `data/` — data paths are
  relative to the current working directory (default `--data-dir data`).
- First model init downloads pretrained backbone weights from `download.pytorch.org` (needs network).

### Data (not committed)
- `data/` and `outputs/` are gitignored and absent by default. The real dataset is the Kaggle
  competition (`train.csv`/`test.csv` with columns `filename`,`appearance`; images under
  `data/images/`). Place it in `data/` to reproduce real results.
- For a quick smoke test without the Kaggle data, generate a small synthetic dataset (a handful of
  JPEGs per class under `data/images/` plus `train.csv`/`test.csv`) and run a tiny config on CPU:
  `.venv/bin/python src/train.py --arch efficientnet_b0 --image-size 64 --epochs 2 --batch-size 8 --fold 0`
  then `.venv/bin/python src/predict.py --checkpoint outputs/checkpoints/efficientnet_b0_fold0_best.pt --tta`.
  Note: `StratifiedKFold` uses `--n-folds` (default 5), so each class needs at least that many rows.
- Full-size training (`convnext_tiny` at 384px, 18 epochs) is impractical on this CPU-only VM; use
  small arch / image-size / epochs for verification.

### Lint / test / build
- No linter or tests are configured. Use `.venv/bin/python -m py_compile src/*.py` as a basic
  validity check. There is no build step (pure Python scripts).
