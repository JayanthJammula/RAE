import argparse
import json
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create deterministic slices from a JSON dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source dataset (JSON list).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the slice files will be written.",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=300,
        help="Number of entries per slice.",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=5,
        help="How many slices to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--prefix",
        help="Optional prefix for the slice file names (defaults to input stem).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing slice files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Expected the input JSON to contain a list of records.")

    total_records = len(dataset)
    required = args.slice_size * args.num_slices
    if total_records < required:
        raise ValueError(
            f"Dataset has {total_records} items but needs {required} "
            "for the requested slices."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or input_path.stem
    rng = random.Random(args.seed)
    indices = list(range(total_records))
    rng.shuffle(indices)

    for slice_idx in range(args.num_slices):
        start = slice_idx * args.slice_size
        end = start + args.slice_size
        chunk_indices = indices[start:end]
        slice_data = [dataset[i] for i in chunk_indices]

        out_path = output_dir / f"{prefix}_slice{slice_idx + 1:02d}_{args.slice_size}.json"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{out_path} exists. Use --overwrite to replace existing slices."
            )

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(slice_data, f, ensure_ascii=False, indent=2)

        print(f"Wrote {len(slice_data)} records to {out_path}")


if __name__ == "__main__":
    main()
