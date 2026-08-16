from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/decision.json"))
    return parser.parse_args()


def load_oof(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    oof = pd.concat(frames, ignore_index=True)
    required = ["filename", "appearance", "logit_0", "logit_1", "logit_2", "logit_3", "prob_0", "prob_1", "prob_2", "prob_3"]
    missing = [col for col in required if col not in oof.columns]
    if missing:
        raise ValueError(f"OOF files missing columns: {missing}")
    oof = oof.drop_duplicates("filename", keep="last").reset_index(drop=True)
    return oof


def macro_f1(y_true: np.ndarray, preds: np.ndarray) -> float:
    return float(f1_score(y_true, preds, average="macro", zero_division=0))


def coordinate_search(values: np.ndarray, y_true: np.ndarray, kind: str) -> tuple[np.ndarray, float]:
    best = np.ones(4, dtype=np.float64) if kind == "multipliers" else np.zeros(4, dtype=np.float64)
    best_f1 = -1.0
    grid = np.linspace(0.2, 3.0, 29) if kind == "multipliers" else np.linspace(-2.0, 2.0, 41)

    def score(vec: np.ndarray) -> float:
        if kind == "multipliers":
            scaled = values * np.maximum(vec, 1e-6)
            preds = scaled.argmax(axis=1)
        else:
            preds = (values + vec).argmax(axis=1)
        return macro_f1(y_true, preds)

    best_f1 = score(best)
    for _ in range(3):
        for cls in range(4):
            current = best.copy()
            local_best = current[cls]
            local_f1 = best_f1
            for candidate in grid:
                current[cls] = candidate
                f1 = score(current)
                if f1 > local_f1:
                    local_f1 = f1
                    local_best = candidate
            best[cls] = local_best
            best_f1 = local_f1
    if kind == "multipliers":
        best = best / best.sum()
        best_f1 = score(best)
    return best, best_f1


def binary_thresholds(probs: np.ndarray, y_true: np.ndarray) -> tuple[tuple[float, float], float]:
    p_tom = probs[:, 1] + probs[:, 3]
    p_jerry = probs[:, 2] + probs[:, 3]
    best_f1 = -1.0
    best = (0.5, 0.5)
    for t_tom in np.linspace(0.20, 0.80, 61):
        is_tom = p_tom >= t_tom
        for t_jerry in np.linspace(0.20, 0.80, 61):
            is_jerry = p_jerry >= t_jerry
            preds = (2 * is_jerry.astype(int) + is_tom.astype(int))
            f1 = macro_f1(y_true, preds)
            if f1 > best_f1:
                best_f1 = f1
                best = (float(t_tom), float(t_jerry))
    return best, best_f1


def main():
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    oof = load_oof(args.oof)
    y_true = oof["appearance"].to_numpy()
    logits = oof[["logit_0", "logit_1", "logit_2", "logit_3"]].to_numpy()
    probs = oof[["prob_0", "prob_1", "prob_2", "prob_3"]].to_numpy()

    raw_preds = logits.argmax(axis=1)
    raw_f1 = macro_f1(y_true, raw_preds)
    print(f"raw_argmax_macro_f1={raw_f1:.5f} rows={len(oof)}")
    print(classification_report(y_true, raw_preds, digits=4, zero_division=0))

    bias, bias_f1 = coordinate_search(logits, y_true, kind="bias")
    multipliers, mult_f1 = coordinate_search(probs, y_true, kind="multipliers")
    bin_th, bin_f1 = binary_thresholds(probs, y_true)

    print(f"bias={np.round(bias, 4).tolist()} macro_f1={bias_f1:.5f} gain={bias_f1 - raw_f1:+.5f}")
    print(f"multipliers={np.round(multipliers, 4).tolist()} macro_f1={mult_f1:.5f} gain={mult_f1 - raw_f1:+.5f}")
    print(f"binary_thresholds tom={bin_th[0]:.3f} jerry={bin_th[1]:.3f} macro_f1={bin_f1:.5f} gain={bin_f1 - raw_f1:+.5f}")

    candidates = [
        ("bias", bias_f1, {"method": "bias", "bias": bias.tolist()}),
        ("multipliers", mult_f1, {"method": "multipliers", "multipliers": multipliers.tolist()}),
        (
            "binary_thresholds",
            bin_f1,
            {"method": "binary_thresholds", "binary_thresholds": {"tom": bin_th[0], "jerry": bin_th[1]}},
        ),
    ]
    best_name, best_f1, best_config = max(candidates, key=lambda item: item[1])
    if best_f1 <= raw_f1 + 1e-6:
        best_name = "argmax"
        best_f1 = raw_f1
        best_config = {"method": "argmax"}
        print("no_decision_gain keeping argmax")

    best_config.update(
        {
            "raw_macro_f1": raw_f1,
            "calibrated_macro_f1": best_f1,
            "chosen": best_name,
            "bias": bias.tolist(),
            "multipliers": multipliers.tolist(),
            "binary_thresholds": {"tom": bin_th[0], "jerry": bin_th[1]},
            "n_oof": int(len(oof)),
        }
    )
    args.out_json.write_text(json.dumps(best_config, indent=2), encoding="utf-8")
    print(f"chose {best_name} calibrated_macro_f1={best_f1:.5f}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
