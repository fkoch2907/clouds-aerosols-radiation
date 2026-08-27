#!/usr/bin/env python3
"""Combine daily cloud-fraction time series for one camera into one CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.resolve()
DEFAULT_INPUT_DIR = REPO_ROOT / "Data" / "CloudCamProcessed" / "CAM1"
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_DIR / "combined_cloud_fraction_timeseries.csv"
INPUT_FILENAME = "cloud_fraction_timeseries.csv"


def combine_timeseries(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    camera: str = "CAM1",
) -> Path:
    """Combine daily time-series CSVs below *input_dir*.

    Daily files are expected at ``input_dir/<day>/<INPUT_FILENAME>``. The output
    contains a new ``camera`` column as its first column and keeps the remaining
    columns and row order from each daily file. Day directories are processed in
    sorted order.
    """
    input_root = Path(input_dir).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    csv_paths = sorted(input_root.glob(f"*/{INPUT_FILENAME}"))
    if not csv_paths:
        raise FileNotFoundError(f"No {INPUT_FILENAME} files found in {input_root}")

    fieldnames: list[str] | None = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)

        for csv_path in csv_paths:
            with csv_path.open(newline="", encoding="utf-8") as input_file:
                reader = csv.DictReader(input_file)
                current_fields = reader.fieldnames
                if current_fields is None:
                    raise ValueError(f"CSV file has no header: {csv_path}")
                if fieldnames is None:
                    fieldnames = current_fields
                    writer.writerow(["camera", *fieldnames])
                elif current_fields != fieldnames:
                    raise ValueError(
                        f"CSV headers differ in {csv_path}: expected {fieldnames}, "
                        f"found {current_fields}"
                    )

                for row in reader:
                    writer.writerow([camera, *(row[field] for field in fieldnames)])

    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine daily cloud_fraction_timeseries.csv files for one camera."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Camera processed-data directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Combined CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument("--camera", default="CAM1", help="Camera label for each row (default: CAM1)")
    args = parser.parse_args()

    try:
        output_path = combine_timeseries(args.input_dir, args.output, args.camera)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Combined time series written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
