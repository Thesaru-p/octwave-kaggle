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
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/submission_review.html"))
    parser.add_argument("--limit", type=int, default=300)
    return parser.parse_args()


def find_image(data_dir: Path, filename: str) -> Path:
    candidates = [
        data_dir / "images" / filename,
        data_dir / "images" / "images" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def file_url(path: Path) -> str:
    return path.as_uri()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.read_csv(args.submission).head(args.limit)

    cards = []
    for _, row in submission.iterrows():
        filename = row["filename"]
        label = LABELS[int(row["appearance"])]
        image_url = file_url(find_image(args.data_dir, filename))
        cards.append(
            f"""
            <article class="card">
              <img src="{image_url}" loading="lazy" />
              <div class="meta">
                <strong>{filename}</strong>
                <span>Pred: {label}</span>
              </div>
            </article>
            """
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Submission Review</title>
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
  <h1>Submission Review</h1>
  <p>Labels: 0 neither, 1 Tom only, 2 Jerry only, 3 both.</p>
  <section class="grid">
    {''.join(cards)}
  </section>
</body>
</html>
"""
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} rows={len(submission)}")


if __name__ == "__main__":
    main()
