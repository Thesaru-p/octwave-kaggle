from pathlib import Path

import pandas as pd


def main():
    data_dir = Path("data")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    print("train rows:", len(train))
    print("test rows:", len(test))
    print("\nclass distribution:")
    print(train["appearance"].value_counts().sort_index())
    print("\nclass percentages:")
    print((train["appearance"].value_counts(normalize=True).sort_index() * 100).round(2))


if __name__ == "__main__":
    main()
