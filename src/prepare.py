import os
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

import config
from utils.string_norm import strip_accents_and_lowercase


def calculate_category_score(cat1: int, cat2: int):
    return 1.0 if cat1 == cat2 else 0.0


def calculate_gender_score(gender1: str, gender2: str):
    if gender1 == "u" and gender2 == "u":
        return 0.0
    elif gender1 != gender2:
        return 0.5
    else:
        return 1.0


def generate_sentence_pairs(
    df: pd.DataFrame,
    n_pairs: int,
    allow_same_sentence: bool = False,
    random_seed: int = 42,
    text_transform: Optional[Callable[[str], str]] = None,
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    n_sentences = len(df)
    pairs = []

    if allow_same_sentence:
        max_pairs = n_sentences * n_sentences
    else:
        max_pairs = n_sentences * (n_sentences - 1)

    if n_pairs > max_pairs:
        print(
            f"Warning: Requested {n_pairs} pairs, but max possible is {max_pairs}. Using {max_pairs}."
        )
        n_pairs = max_pairs

    seen_pairs = set()
    attempts = 0
    max_attempts = n_pairs * 10

    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        idx1 = random.randint(0, n_sentences - 1)
        idx2 = random.randint(0, n_sentences - 1)

        if not allow_same_sentence and idx1 == idx2:
            continue

        pair_key = (min(idx1, idx2), max(idx1, idx2)) if idx1 != idx2 else (idx1, idx2)

        if pair_key in seen_pairs:
            continue

        seen_pairs.add(pair_key)

        row1 = df.iloc[idx1]
        row2 = df.iloc[idx2]

        text1 = row1["text"]
        text2 = row2["text"]
        if text_transform is not None:
            text1 = text_transform(text1)
            text2 = text_transform(text2)

        category_score = calculate_category_score(row1["nkv_group"], row2["nkv_group"])
        gender_score = calculate_gender_score(
            row1["sentence_gender"], row2["sentence_gender"]
        )

        pairs.append(
            {
                "sentence1": text1,
                "sentence2": text2,
                "category_score": category_score,
                "gender_score": gender_score,
            }
        )

    return pd.DataFrame(pairs)


def split_train_dev_eval(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    eval_ratio: float = 0.1,
    random_seed: int = 42,
):
    assert abs(train_ratio + dev_ratio + eval_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    n = len(df_shuffled)
    train_end = int(n * train_ratio)
    dev_end = train_end + int(n * dev_ratio)

    train_df = df_shuffled.iloc[:train_end].reset_index(drop=True)
    dev_df = df_shuffled.iloc[train_end:dev_end].reset_index(drop=True)
    eval_df = df_shuffled.iloc[dev_end:].reset_index(drop=True)

    return train_df, dev_df, eval_df


def save_splits(
    train_df: pd.DataFrame, dev_df: pd.DataFrame, eval_df: pd.DataFrame, output_dir: str
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_path / "train_pairs.csv", index=False)
    dev_df.to_csv(output_path / "dev_pairs.csv", index=False)
    eval_df.to_csv(output_path / "eval_pairs.csv", index=False)

    print(f"Saved splits to {output_path}:")
    print(f"  - train_pairs.csv: {len(train_df)} pairs")
    print(f"  - dev_pairs.csv: {len(dev_df)} pairs")
    print(f"  - eval_pairs.csv: {len(eval_df)} pairs")


def main():
    csv_path = "data/verses.csv"
    n_pairs = 1_500_000
    allow_same_sentence = False
    random_seed = 42

    train_ratio = 0.8
    dev_ratio = 0.1
    eval_ratio = 0.1

    if os.path.exists(config.PREPRO_PATH):
        print(f"Output directory {config.PREPRO_PATH} already exists.")
        return

    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} sentences")

    df = df.dropna(subset=["text"])

    print(f"\nGenerating {n_pairs} sentence pairs...")
    print(f"  - Allow same sentence: {allow_same_sentence}")

    pairs_df = generate_sentence_pairs(
        df,
        n_pairs=n_pairs,
        allow_same_sentence=allow_same_sentence,
        random_seed=random_seed,
        text_transform=strip_accents_and_lowercase,
    )

    print("\nDone...\n")

    print("\nCategory score distribution:")
    print(pairs_df["category_score"].value_counts().sort_index())

    print("\nGender score distribution:")
    print(pairs_df["gender_score"].value_counts().sort_index())

    print(
        f"\nSplitting into train/dev/eval ({train_ratio}/{dev_ratio}/{eval_ratio})..."
    )

    train_df, dev_df, eval_df = split_train_dev_eval(
        pairs_df,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        eval_ratio=eval_ratio,
        random_seed=random_seed,
    )

    print("\nSaving splits...")
    save_splits(train_df, dev_df, eval_df, config.PREPRO_PATH)

    print("\nFinished!")


if __name__ == "__main__":
    main()
