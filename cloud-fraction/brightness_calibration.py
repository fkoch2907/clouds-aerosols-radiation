#!/usr/bin/env python3
"""Derive brightness-quality thresholds from a manually selected image subset.

The subset should contain JPG/JPEG files in one directory. The script applies
the same colour classification, fisheye geometry, outer-edge mask, custom mask,
sun mask, and solid-angle weights as ``cloud_fraction.py``. It writes raw
per-image diagnostics and a JSON configuration fragment with data-derived
thresholds for the normal processing workflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from camera_geometry import CameraGeometry, resolve_center_radius
from cloud_fraction import (
    DEFAULT_TS_REGEX,
    HO_MASK,
    HO_SUN,
    build_classification_table,
    calculate_weighted_cloud_brightness_statistics,
    circular_mask,
    load_custom_mask,
    parse_timestamp,
    solid_angle_weight_map,
    sun_pixel_position,
    sun_position,
)


def nan_brightness_diagnostics() -> dict[str, float]:
    return {
        "cloud_pixel_count": 0.0,
        "effective_cloud_pixels": np.nan,
        "mean_r": np.nan,
        "mean_g": np.nan,
        "mean_b": np.nan,
        "std_r": np.nan,
        "std_g": np.nan,
        "std_b": np.nan,
    }


def derive_thresholds(
    diagnostics: list[dict[str, float]],
    target_error_rgb: float = 10.0,
    confidence_z: float = 1.96,
    reference_quantile: float = 0.75,
    mad_multiplier: float = 3.0,
) -> dict[str, float]:
    """Derive minimum effective pixels and a robust RGB std threshold.

    The upper ``reference_quantile`` of effective cloud-pixel counts provides
    a well-sampled reference population. Its typical RGB variability estimates
    the sample size needed for ``target_error_rgb`` at ``confidence_z``. The
    standard-deviation threshold is the pooled median plus a MAD-based robust
    allowance over that same reference population.
    """
    if not diagnostics:
        raise ValueError("No brightness diagnostics available.")
    if target_error_rgb <= 0 or confidence_z <= 0:
        raise ValueError("target_error_rgb and confidence_z must be positive.")
    if not 0.0 <= reference_quantile < 1.0:
        raise ValueError("reference_quantile must be in the range [0, 1).")
    if mad_multiplier < 0:
        raise ValueError("mad_multiplier must not be negative.")

    effective = np.array([d["effective_cloud_pixels"] for d in diagnostics], dtype=float)
    valid = np.isfinite(effective)
    if not np.any(valid):
        raise ValueError("No images contain classified cloud pixels.")

    reference_cutoff = float(np.quantile(effective[valid], reference_quantile))
    reference = [
        d for d in diagnostics
        if np.isfinite(d["effective_cloud_pixels"])
        and d["effective_cloud_pixels"] >= reference_cutoff
    ]
    std_values = np.array(
        [d[key] for d in reference for key in ("std_r", "std_g", "std_b")],
        dtype=float,
    )
    std_values = std_values[np.isfinite(std_values)]
    if std_values.size == 0:
        raise ValueError("Reference images contain no finite RGB standard deviations.")

    typical_std = float(np.median(std_values))
    min_effective_pixels = max(
        1.0,
        math.ceil((confidence_z * typical_std / target_error_rgb) ** 2),
    )
    median_std = float(np.median(std_values))
    mad = float(np.median(np.abs(std_values - median_std)))
    std_threshold = median_std + mad_multiplier * 1.4826 * mad

    return {
        "brightness_min_effective_pixels": float(min_effective_pixels),
        "brightness_std_threshold": float(std_threshold),
        "calibration_target_error_rgb": float(target_error_rgb),
        "calibration_confidence_z": float(confidence_z),
        "calibration_reference_quantile": float(reference_quantile),
        "calibration_reference_cutoff_effective_pixels": reference_cutoff,
        "calibration_reference_image_count": float(len(reference)),
        "calibration_typical_rgb_std": typical_std,
    }


def classify_for_calibration(
    img_path: Path,
    class_table: np.ndarray,
    geo: CameraGeometry,
    edge_margin_px: float,
    sun_radius_px: float,
    utc_offset_hours: float,
    ts_regex: re.Pattern[str],
    custom_mask: np.ndarray | None,
) -> tuple[datetime, np.ndarray, np.ndarray, np.ndarray]:
    """Return timestamp, RGB array, masked classes, and solid-angle weights."""
    dt_utc = parse_timestamp(img_path.name, ts_regex, utc_offset_hours)
    arr = np.array(Image.open(img_path).convert("RGB"))
    height, width, _ = arr.shape
    classes = class_table[arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]]

    cx, cy, r_max = resolve_center_radius(geo, width, height)
    fov_circle = circular_mask(height, width, cx, cy, r_max - edge_margin_px)
    mask_outer_edge = ~fov_circle
    if custom_mask is not None:
        mask_outer_edge |= custom_mask
    classes[mask_outer_edge] = HO_MASK

    azimuth_deg, elevation_deg = sun_position(dt_utc, geo)
    sun_xy = sun_pixel_position(azimuth_deg, elevation_deg, width, height, geo)
    if sun_xy is not None:
        classes[circular_mask(height, width, sun_xy[0], sun_xy[1], sun_radius_px)] = HO_SUN
    if custom_mask is not None:
        classes[custom_mask] = HO_MASK

    weights = solid_angle_weight_map(height, width, cx, cy, r_max, geo)
    return dt_utc, arr, classes, weights


def write_diagnostics(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "filename", "timestamp_utc", "cloud_pixel_count", "effective_cloud_pixels",
        "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True, help="Manually selected JPG subset")
    parser.add_argument("--output_dir", type=Path, required=True, help="Calibration output directory")
    parser.add_argument("--himmel_file", type=Path, required=True)
    parser.add_argument("--wolken_file", type=Path, required=True)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--alt", type=float, default=0.0)
    parser.add_argument("--fov_deg", type=float, default=180.0)
    parser.add_argument("--center_x", type=float, default=None)
    parser.add_argument("--center_y", type=float, default=None)
    parser.add_argument("--circle_radius_px", type=float, default=None)
    parser.add_argument("--edge_margin_px", type=float, default=5.0)
    parser.add_argument("--sun_radius_px", type=float, default=25.0)
    parser.add_argument("--custom_mask_file", type=Path, default=None)
    parser.add_argument("--timestamp_regex", type=str, default=None)
    parser.add_argument("--utc_offset_hours", type=float, default=0.0)
    parser.add_argument("--target_error_rgb", type=float, default=10.0)
    parser.add_argument("--confidence_z", type=float, default=1.96)
    parser.add_argument("--reference_quantile", type=float, default=0.75)
    parser.add_argument("--mad_multiplier", type=float, default=3.0)
    args = parser.parse_args()

    image_paths = sorted(
        p for p in args.input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg")
    )
    if not image_paths:
        raise FileNotFoundError(f"No JPG image found in {args.input_dir}.")

    class_table = build_classification_table(args.himmel_file, args.wolken_file)
    geo = CameraGeometry(
        lat=args.lat, lon=args.lon, alt=args.alt, fov_deg=args.fov_deg,
        center_x=args.center_x, center_y=args.center_y, radius_px=args.circle_radius_px,
    )
    custom_mask = None
    if args.custom_mask_file is not None:
        with Image.open(image_paths[0]) as first_image:
            width, height = first_image.size
        custom_mask = load_custom_mask(args.custom_mask_file, width, height)

    ts_regex = re.compile(args.timestamp_regex) if args.timestamp_regex else DEFAULT_TS_REGEX
    diagnostics: list[dict[str, float]] = []
    rows: list[dict[str, object]] = []
    for image_path in image_paths:
        dt_utc, arr, classes, weights = classify_for_calibration(
            image_path, class_table, geo, args.edge_margin_px, args.sun_radius_px,
            args.utc_offset_hours, ts_regex, custom_mask,
        )
        stats = calculate_weighted_cloud_brightness_statistics(arr, classes, weights)
        diagnostics.append(stats)
        rows.append({"filename": image_path.name, "timestamp_utc": dt_utc.isoformat(), **stats})
        print(f"  {image_path.name}: n_eff={stats['effective_cloud_pixels']:.1f}, "
              f"std=({stats['std_r']:.1f}, {stats['std_g']:.1f}, {stats['std_b']:.1f})")

    thresholds = derive_thresholds(
        diagnostics, args.target_error_rgb, args.confidence_z,
        args.reference_quantile, args.mad_multiplier,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_diagnostics(args.output_dir / "brightness_calibration_diagnostics.csv", rows)
    (args.output_dir / "brightness_thresholds.json").write_text(
        json.dumps(thresholds, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(thresholds, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())