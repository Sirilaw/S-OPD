"""Download the raw Geometry3K dataset to local parquet files."""

import argparse
from pathlib import Path

from datasets import concatenate_datasets, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-proc", type=int, default=8)
    args = parser.parse_args()

    dataset = load_dataset("hiyouga/geometry3k", num_proc=args.num_proc)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Geometry3K publishes train/validation/test. Keep the official test set for
    # evaluation and fold validation into training so none of the 300 examples
    # are silently dropped by the two-file training convention used here.
    output_splits = {
        "train": concatenate_datasets([dataset["train"], dataset["validation"]]),
        "test": dataset["test"],
    }
    for split, split_dataset in output_splits.items():
        destination = args.output_dir / f"{split}.parquet"
        split_dataset.to_parquet(destination)
        print(f"Wrote {len(split_dataset)} rows to {destination}")


if __name__ == "__main__":
    main()
