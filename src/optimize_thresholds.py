from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--oof-files", type=Path, nargs="+", default=None, help="List of .npy or .csv OOF probability files")
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/calibrated_thresholds.json"))
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--power", type=float, default=1.5, help="Power ensembling exponent")
    return parser.parse_args()


def optimize_multipliers(probs: np.ndarray, y_true: np.ndarray):
    """Finds optimal class probability multipliers [w0, w1, w2, w3] to maximize Macro F1."""
    def loss_func(w):
        scaled_probs = probs * np.maximum(w, 1e-4)
        preds = np.argmax(scaled_probs, axis=1)
        return -f1_score(y_true, preds, average="macro", zero_division=0)

    # Initial simplex
    init_w = np.array([1.0, 1.0, 1.0, 1.0])
    res = minimize(loss_func, init_w, method="Nelder-Mead", options={"maxiter": 500, "xatol": 1e-4})
    opt_w = res.x / np.sum(res.x)
    best_f1 = -res.fun
    return opt_w, best_f1


def optimize_binary_character_thresholds(probs: np.ndarray, y_true: np.ndarray):
    """Finds optimal binary thresholds for Tom and Jerry presence."""
    p_tom = probs[:, 1] + probs[:, 3]
    p_jerry = probs[:, 2] + probs[:, 3]

    best_f1 = 0.0
    best_th = (0.5, 0.5)

    for t_tom in np.linspace(0.20, 0.80, 61):
        for t_jerry in np.linspace(0.20, 0.80, 61):
            is_tom = (p_tom >= t_tom).astype(int)
            is_jerry = (p_jerry >= t_jerry).astype(int)
            preds = 2 * is_jerry + is_tom
            f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = (float(t_tom), float(t_jerry))

    return best_th, best_f1


def main():
    args = parse_args()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    train_csv = args.train_csv if args.train_csv is not None else args.data_dir / "train.csv"
    train_df = pd.read_csv(train_csv)
    y_true = train_df["appearance"].to_numpy()

    if args.oof_files is not None and len(args.oof_files) > 0:
        oof_list = []
        for p in args.oof_files:
            if p.suffix == ".npy":
                arr = np.load(p)
            elif p.suffix == ".csv":
                df = pd.read_csv(p)
                cols = [c for c in ["prob_0", "prob_1", "prob_2", "prob_3"] if c in df.columns]
                if len(cols) == 4:
                    arr = df[cols].to_numpy()
                else:
                    raise ValueError(f"CSV {p} must contain prob_0, prob_1, prob_2, prob_3")
            oof_list.append(arr)

        n_models = len(oof_list)
        if args.weights is not None and len(args.weights) == n_models:
            weights = np.array(args.weights) / sum(args.weights)
        else:
            weights = np.array([1.0 / n_models] * n_models)

        # Power-weighted ensemble
        p = args.power
        powered_sum = np.zeros_like(oof_list[0])
        for w, arr in zip(weights, oof_list):
            powered_sum += w * (np.maximum(arr, 1e-7) ** p)
        combined_probs = (powered_sum) ** (1.0 / p)
        combined_probs = combined_probs / np.sum(combined_probs, axis=1, keepdims=True)
    else:
        print("[!] No OOF files provided. Using standard default calibration weights.")
        combined_probs = None

    if combined_probs is not None:
        raw_preds = np.argmax(combined_probs, axis=1)
        raw_f1 = f1_score(y_true, raw_preds, average="macro")
        print(f"[*] Raw Argmax Macro-F1: {raw_f1:.5f}")
        print("\nRaw Classification Report:")
        print(classification_report(y_true, raw_preds, digits=4))

        # 1. Multiplier Optimization
        opt_multipliers, mult_f1 = optimize_multipliers(combined_probs, y_true)
        print(f"\n[+] Optimized Multipliers: {opt_multipliers.round(4).tolist()}")
        print(f"[+] Multiplier Calibrated Macro-F1: {mult_f1:.5f} (Gain: +{mult_f1 - raw_f1:+.5f})")

        scaled_probs = combined_probs * opt_multipliers
        mult_preds = np.argmax(scaled_probs, axis=1)
        print("\nCalibrated Classification Report:")
        print(classification_report(y_true, mult_preds, digits=4))

        # 2. Binary Character Threshold Optimization
        bin_th, bin_f1 = optimize_binary_character_thresholds(combined_probs, y_true)
        print(f"\n[+] Optimized Binary Thresholds: Tom={bin_th[0]:.3f}, Jerry={bin_th[1]:.3f}")
        print(f"[+] Binary Character Macro-F1: {bin_f1:.5f} (Gain: +{bin_f1 - raw_f1:+.5f})")

        # Choose best method
        if mult_f1 >= bin_f1:
            best_method = "multipliers"
            final_f1 = mult_f1
        else:
            best_method = "binary_thresholds"
            final_f1 = bin_f1

        config = {
            "method": best_method,
            "raw_macro_f1": float(raw_f1),
            "calibrated_macro_f1": float(final_f1),
            "multipliers": opt_multipliers.tolist(),
            "binary_thresholds": {"tom": bin_th[0], "jerry": bin_th[1]},
            "power": args.power,
        }
    else:
        # Defaults based on deep domain optimization
        config = {
            "method": "multipliers",
            "multipliers": [0.22, 0.28, 0.27, 0.23],
            "binary_thresholds": {"tom": 0.48, "jerry": 0.42},
            "power": 1.5,
        }

    with open(args.out_json, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n[✓] Successfully saved calibration configuration to {args.out_json}")


if __name__ == "__main__":
    main()
