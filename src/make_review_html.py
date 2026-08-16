from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


LABELS = {
    0: "0 neither",
    1: "1 Tom only",
    2: "2 Jerry only",
    3: "3 both",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--probs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/review.html"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--sort-by", choices=["confidence", "margin"], default="margin")
    parser.add_argument("--match-diagnostics", type=Path, default=None)
    parser.add_argument("--only-unmatched", action="store_true")
    return parser.parse_args()


def image_path(data_dir: Path, filename: str) -> str:
    candidates = [
        data_dir / "images" / filename,
        data_dir / "images" / "images" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve().as_uri()
    return candidates[-1].resolve().as_uri()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    probs = pd.read_csv(args.probs)
    if args.only_unmatched:
        if args.match_diagnostics is None:
            raise ValueError("--only-unmatched requires --match-diagnostics")
        diagnostics = pd.read_csv(args.match_diagnostics)
        unmatched = set(diagnostics.loc[diagnostics["match_source"] == "model", "filename"])
        probs = probs[probs["filename"].isin(unmatched)].copy()
        print(f"unmatched_rows={len(probs)}")
    review = probs.sort_values(args.sort_by, ascending=True).head(args.limit).copy()

    rows = []
    for _, row in review.iterrows():
        label = LABELS[int(row["appearance"])]
        img = image_path(args.data_dir, row["filename"])
        rows.append(
            f"""
            <article class="card">
              <img src="{img}" loading="lazy" />
              <div class="meta">
                <strong>{row['filename']}</strong>
                <span>Pred: {label}</span>
                <span>Conf: {row['confidence']:.4f}</span>
                <span>Margin: {row['margin']:.4f}</span>
                <span>p0={row['prob_0']:.3f} p1={row['prob_1']:.3f} p2={row['prob_2']:.3f} p3={row['prob_3']:.3f}</span>
              </div>
            </article>
            """
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Low Confidence Review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #111; color: #eee; }}
    h1 {{ font-size: 22px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; }}
    .card {{ border: 1px solid #333; background: #1b1b1b; padding: 10px; border-radius: 6px; }}
    img {{ width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #000; }}
    .meta {{ display: grid; gap: 4px; margin-top: 8px; font-size: 13px; }}
    strong {{ color: #fff; }}
  </style>
</head>
<body>
  <h1>Lowest {args.sort_by} Predictions</h1>
  <p>Use this page to manually inspect uncertain predictions. Labels: 0 neither, 1 Tom only, 2 Jerry only, 3 both.</p>
  <section class="grid">
    {''.join(rows)}
  </section>
</body>
</html>
"""

    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} images={len(review)}")


if __name__ == "__main__":
    main()
