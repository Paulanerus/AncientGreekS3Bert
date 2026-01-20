from pathlib import Path

import numpy as np
import pandas as pd

import config
from utils.string_norm import strip_accents_and_lowercase

SEED = 42
TARGET_TOTAL = 1_500_000
GENDERS = {"m", "f", "u"}


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text", "sentence_gender"]).copy()
    df["sentence_gender"] = df["sentence_gender"].astype(str).str.strip().str.lower()
    df = df[df["sentence_gender"].isin(GENDERS)]
    df["text"] = df["text"].astype(str).apply(strip_accents_and_lowercase)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"])
    return df[["text", "sentence_gender"]].reset_index(drop=True)


def split_df(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    indices = rng.permutation(len(df))
    train_end = int(len(df) * 0.9)
    dev_end = train_end + int(len(df) * 0.05)
    return {
        "train": df.iloc[indices[:train_end]].reset_index(drop=True),
        "dev": df.iloc[indices[train_end:dev_end]].reset_index(drop=True),
        "eval": df.iloc[indices[dev_end:]].reset_index(drop=True),
    }


def pool_size(texts_a: list, texts_b: list, same_group: bool) -> int:
    if same_group:
        n = len(texts_a)
        return n * (n - 1) // 2
    return len(texts_a) * len(texts_b)


def allocate_counts(target: int, sizes: list) -> list:
    total = sum(sizes)
    if total == 0:
        return [0 for _ in sizes]
    if target >= total:
        return list(sizes)

    raw = [(size / total) * target for size in sizes]
    counts = [int(value) for value in raw]
    remainder = target - sum(counts)
    if remainder > 0:
        fracs = [value - int(value) for value in raw]
        for idx in sorted(range(len(sizes)), key=lambda i: fracs[i], reverse=True):
            if remainder == 0:
                break
            if counts[idx] < sizes[idx]:
                counts[idx] += 1
                remainder -= 1
    if remainder > 0:
        available = [i for i in range(len(sizes)) if counts[i] < sizes[i]]
        while remainder > 0 and available:
            for idx in list(available):
                if remainder == 0:
                    break
                if counts[idx] < sizes[idx]:
                    counts[idx] += 1
                    remainder -= 1
            available = [i for i in available if counts[i] < sizes[i]]
    return counts


def sample_pairs(
    texts_a: list,
    texts_b: list,
    same_group: bool,
    count: int,
    rng: np.random.Generator,
) -> list:
    if count <= 0:
        return []

    total = pool_size(texts_a, texts_b, same_group)
    if total == 0:
        return []

    count = min(count, total)

    if same_group:
        n_a = len(texts_a)
        if count == total:
            return [
                (texts_a[i], texts_a[j]) for i in range(n_a) for j in range(i + 1, n_a)
            ]

        pairs_idx = set()
        while len(pairs_idx) < count:
            i = int(rng.integers(0, n_a))
            j = int(rng.integers(0, n_a - 1))
            if j >= i:
                j += 1
            if i > j:
                i, j = j, i
            pairs_idx.add((i, j))
        return [(texts_a[i], texts_a[j]) for i, j in pairs_idx]

    n_a = len(texts_a)
    n_b = len(texts_b)
    if count == total:
        return [(a, b) for a in texts_a for b in texts_b]

    pairs_idx = set()
    while len(pairs_idx) < count:
        i = int(rng.integers(0, n_a))
        j = int(rng.integers(0, n_b))
        pairs_idx.add((i, j))
    return [(texts_a[i], texts_b[j]) for i, j in pairs_idx]


def sample_label_pairs(pools: list, target: int, rng: np.random.Generator) -> list:
    sizes = [
        pool_size(texts_a, texts_b, same_group)
        for texts_a, texts_b, same_group in pools
    ]
    counts = allocate_counts(target, sizes)
    pairs = []
    for (texts_a, texts_b, same_group), count in zip(pools, counts):
        pairs.extend(sample_pairs(texts_a, texts_b, same_group, count, rng))
    return pairs


def label_targets(split_total: int) -> dict:
    base = split_total // 4
    remainder = split_total - base * 4
    targets = {"same": base, "mixed": base, "undef": base, "mixed_undef": base}
    order = ["same", "mixed", "undef", "mixed_undef"]
    for idx in range(remainder):
        targets[order[idx]] += 1
    return targets


def split_targets(total: int) -> dict:
    train = int(total * 0.9)
    dev = int(total * 0.05)
    eval_total = total - train - dev
    return {"train": train, "dev": dev, "eval": eval_total}


def build_pairs(
    df: pd.DataFrame, targets: dict, rng: np.random.Generator, name: str
) -> pd.DataFrame:
    texts_m = df[df["sentence_gender"] == "m"]["text"].tolist()
    texts_f = df[df["sentence_gender"] == "f"]["text"].tolist()
    texts_u = df[df["sentence_gender"] == "u"]["text"].tolist()

    same_pools = [
        (texts_m, texts_m, True),
        (texts_f, texts_f, True),
    ]
    mixed_pools = [
        (texts_m, texts_f, False),
    ]
    mixed_undef_pools = [
        (texts_m, texts_u, False),
        (texts_f, texts_u, False),
    ]
    undef_pools = [(texts_u, texts_u, True)]

    same_sizes = [pool_size(texts_m, texts_m, True), pool_size(texts_f, texts_f, True)]
    mixed_sizes = [
        pool_size(texts_m, texts_f, False),
    ]
    mixed_undef_sizes = [
        pool_size(texts_m, texts_u, False),
        pool_size(texts_f, texts_u, False),
    ]
    undef_sizes = [pool_size(texts_u, texts_u, True)]

    print(
        f"{name} pools: same={sum(same_sizes)} mixed={sum(mixed_sizes)} mixed_undef={sum(mixed_undef_sizes)} undef={sum(undef_sizes)}"
    )
    print(
        f"{name} targets: same={targets['same']} mixed={targets['mixed']} mixed_undef={targets['mixed_undef']} undef={targets['undef']}"
    )

    same_pairs = sample_label_pairs(same_pools, targets["same"], rng)
    mixed_pairs = sample_label_pairs(mixed_pools, targets["mixed"], rng)
    mixed_undef_pairs = sample_label_pairs(
        mixed_undef_pools, targets["mixed_undef"], rng
    )
    undef_pairs = sample_label_pairs(undef_pools, targets["undef"], rng)

    data = []
    data.extend([(a, b, 1.0) for a, b in same_pairs])
    data.extend([(a, b, 0.75) for a, b in mixed_pairs])
    data.extend([(a, b, 0.25) for a, b in mixed_undef_pairs])
    data.extend([(a, b, 0.0) for a, b in undef_pairs])

    pairs_df = pd.DataFrame(data, columns=["sentence1", "sentence2", "gender_score"])
    shuffle_seed = int(rng.integers(0, 2**31))
    pairs_df = pairs_df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    return pairs_df


def print_split_report(name: str, df: pd.DataFrame):
    total = len(df)
    counts = df["gender_score"].value_counts().sort_index()
    print(f"{name} pairs: {total}")
    for score, count in counts.items():
        percent = (count / total) * 100 if total else 0
        print(f"  gender_score={score}: {count} ({percent:.2f}%)")


def main():
    rng = np.random.default_rng(SEED)
    data_path = Path(config.DATA_PATH) / "verses.csv"
    pairs_dir = Path(config.PREPRO_PATH)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = normalize_df(df)

    print(f"Total normalized unique verses: {len(df)}")

    splits = split_df(df, rng)
    targets = split_targets(TARGET_TOTAL)

    print(f"Splits: {targets}")

    for split_name, split_frame in splits.items():
        split_target = targets[split_name]
        split_label_targets = label_targets(split_target)
        print(
            f"\nBuilding {split_name} pairs for {len(split_frame)} verses (target={split_target})"
        )
        pairs_df = build_pairs(split_frame, split_label_targets, rng, split_name)
        output_path = pairs_dir / f"{split_name}_pairs.csv"
        pairs_df.to_csv(output_path, index=False)
        print_split_report(split_name.capitalize(), pairs_df)


if __name__ == "__main__":
    main()
