#!/usr/bin/env python3
"""Batch wrapper around cloud_fraction.py for all daily folders listed in img-for-analysis.txt.

The script reads a JSON configuration file, filters the selected camera from the
analysis list, locates the corresponding tilt matrix in Data/CloudCam/CAMx/tilt.json,
and executes cloud_fraction.py once per day directory with a project-standard output
folder under Data/CloudCamProcessed/CAMx/<day-name>.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.resolve()
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "processing_config.json"
DEFAULT_CAMERA_COORDINATES = SCRIPT_DIR / "camera_coordinates.json"


def resolve_path(value: str | Path | None, base_dir: Path) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.expanduser().resolve()


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the batch-processing configuration.

    Relative paths in the config are resolved against the repository root unless they
    explicitly refer to the config file location. This keeps the config portable.
    """
    config_file = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not config_file.is_absolute():
        candidates = [
            (Path.cwd() / config_file).resolve(),
            (SCRIPT_DIR / config_file).resolve(),
            (REPO_ROOT / config_file).resolve(),
        ]
        config_file = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    payload = json.loads(config_file.read_text(encoding="utf-8"))
    base_dir = Path(payload.get("base_dir", str(REPO_ROOT))).expanduser()
    if not base_dir.is_absolute():
        base_dir = (config_file.parent / base_dir).resolve()
    base_dir = base_dir.resolve()

    resolved: dict[str, Any] = dict(payload)
    resolved["base_dir"] = str(base_dir)

    for key in ("input_list", "output_root", "cloud_fraction_script", "himmel_file", "wolken_file", "custom_mask_file", "tilt_matrix_file", "camera_coordinates_file"):
        if key in resolved and resolved[key] not in (None, ""):
            resolved[key] = str(resolve_path(resolved[key], base_dir) or "")

    return resolved


def load_camera_coordinates(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load a mapping of camera ID -> lat/lon/alt. Empty strings act as placeholders."""
    coord_path = Path(path).expanduser() if path is not None else DEFAULT_CAMERA_COORDINATES
    if not coord_path.is_absolute():
        candidates = [
            (Path.cwd() / coord_path).resolve(),
            (SCRIPT_DIR / coord_path).resolve(),
            (REPO_ROOT / coord_path).resolve(),
        ]
        coord_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not coord_path.exists():
        return {}

    payload = json.loads(coord_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def resolve_camera_settings(cfg: dict[str, Any], camera_coords: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Fill in lat/lon/alt and land/sea mask settings from camera_coordinates.json unless explicitly set in config."""
    camera_name = str(cfg.get("camera", "")).strip().upper()
    defaults = camera_coords.get(camera_name, {}) if camera_name else {}
    resolved = dict(cfg)

    # Resolve location coordinates and colorfiles
    for key in ("lat", "lon", "alt", "himmel_file", "wolken_file"):
        if resolved.get(key) in (None, "") and key in defaults and str(defaults[key]).strip() != "":
            resolved[key] = defaults[key]

    # Resolve land/sea mask settings from camera metadata
    mask_settings = defaults.get("land_sea_mask")
    if mask_settings:
        mask_type = mask_settings.get("type")
        if mask_type == "boundaries":
            boundaries = mask_settings.get("boundaries", [])
            resolved["_mask_boundaries"] = boundaries
        elif mask_type == "coastline_bearing":
            bearing = mask_settings.get("coastline_bearing_deg")
            sea_side = mask_settings.get("sea_side", "cw")
            if bearing not in (None, ""):
                resolved["coastline_bearing_deg"] = bearing
                resolved["sea_side"] = sea_side
    else:
        # No mask specified for this camera
        resolved.pop("coastline_bearing_deg", None)
        resolved.pop("sea_side", None)
        resolved.pop("_mask_boundaries", None)

    return resolved


def filter_input_dirs(input_dirs: list[Path | str], camera: str | None) -> list[Path]:
    """Keep only the paths belonging to the chosen camera."""
    camera_name = str(camera or "").strip().upper()
    selected: list[Path] = []
    for item in input_dirs:
        path = Path(item).expanduser()
        if not camera_name:
            selected.append(path)
            continue
        try:
            path_resolved = path.resolve()
        except FileNotFoundError:
            path_resolved = path
        if camera_name in {part.upper() for part in path_resolved.parts}:
            selected.append(path_resolved)
    return selected


def build_output_dir(input_dir: Path | str, output_root: Path | str) -> Path:
    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    camera = input_path.parent.parent.name
    day_name = input_path.parent.name
    return output_path / camera / day_name


def find_tilt_file(camera_root: Path | str, day_dir: Path | str | None = None) -> Path | None:
    camera_path = Path(camera_root).expanduser().resolve()
    candidates = [camera_path / "tilt.json"]
    if day_dir is not None:
        day_path = Path(day_dir).expanduser().resolve()
        candidates.extend([day_path / "tilt.json", day_path.parent / "tilt.json"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def output_exists(output_dir: Path) -> bool:
    """Check if output directory contains any processed files (indicating prior processing)."""
    if not output_dir.exists():
        return False
    # Check if any files exist in the output directory
    return any(output_dir.iterdir())


def build_cloud_fraction_command(cfg: dict[str, Any], input_dir: Path, output_dir: Path, tilt_file: Path | None) -> list[str]:
    script_path = resolve_path(cfg.get("cloud_fraction_script", "cloud-fraction/cloud_fraction.py"), Path(cfg["base_dir"]))
    if script_path is None:
        raise ValueError("cloud_fraction_script is not configured")

    cmd = [sys.executable, str(script_path), "--input_dir", str(input_dir), "--output_dir", str(output_dir)]

    for option_name in (
        "himmel_file",
        "wolken_file",
        "lat",
        "lon",
        "alt",
        "fov_deg",
        "center_x",
        "center_y",
        "circle_radius_px",
        "edge_margin_px",
        "sun_radius_px",
        "custom_mask_file",
        "coastline_bearing_deg",
        "sea_side",
        "timestamp_regex",
        "utc_offset_hours",
        "calculate_brightness",
    ):
        value = cfg.get(option_name)
        if value in (None, ""):
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{option_name.replace('_', '_')}")
            continue
        cmd.extend([f"--{option_name}", str(value)])

    if cfg.get("no_height_correction"):
        cmd.append("--no_height_correction")

    # Add boundary specifications if present (from land_sea_mask type="boundaries")
    boundaries = cfg.get("_mask_boundaries")
    if boundaries and isinstance(boundaries, list):
        for boundary in boundaries:
            cmd.extend(["--boundary", str(boundary)])

    if tilt_file is not None:
        cmd.extend(["--tilt_matrix_file", str(tilt_file)])

    return cmd


def run_batch(config_path: str | Path | None = None, dry_run: bool = False) -> list[Path]:
    cfg = load_config(config_path)
    base_dir = Path(cfg["base_dir"]).resolve()

    camera_coords = load_camera_coordinates(cfg.get("camera_coordinates_file") or DEFAULT_CAMERA_COORDINATES)
    cfg = resolve_camera_settings(cfg, camera_coords)

    input_list = resolve_path(cfg.get("input_list"), base_dir)
    if input_list is None or not input_list.exists():
        raise FileNotFoundError(f"Input list not found: {cfg.get('input_list')}")

    input_dirs = []
    for line in input_list.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        input_dirs.append(resolve_path(cleaned, base_dir) or Path(cleaned))

    selected_days = filter_input_dirs(input_dirs, cfg.get("camera"))
    if not selected_days:
        raise FileNotFoundError(f"No entries for camera {cfg.get('camera')} found in {input_list}.")

    output_root = resolve_path(cfg.get("output_root", "Data/CloudCamProcessed"), base_dir) or (base_dir / "Data/CloudCamProcessed")
    force_recalculate = cfg.get("force_recalculate", False)
    processed: list[Path] = []
    for day_dir in selected_days:
        if not day_dir.exists():
            print(f"Data for day {day_dir.parent.name} not available.")
            continue

        output_dir = build_output_dir(day_dir, output_root)

        # Check if already processed and force_recalculate is false
        if output_exists(output_dir) and not force_recalculate:
            print(f"Data for day {day_dir.parent.name} already processed (skipping; set force_recalculate=true to reprocess).")
            continue

        camera_root = day_dir.parent.parent
        tilt_file = find_tilt_file(camera_root, day_dir)
        if tilt_file is None:
            print(f"No tilt matrix found for {camera_root.name}; proceeding without tilt matrix (assumes perfect leveling and northing).")

        output_dir.mkdir(parents=True, exist_ok=True)
        command = build_cloud_fraction_command(cfg, day_dir, output_dir, tilt_file)
        processed.append(output_dir)

        print(f"Running: {' '.join(command)}")
        if dry_run:
            continue
        subprocess.run(command, check=True)

    return processed


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-run cloud_fraction.py for the selected camera and all days in the analysis list.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="JSON config file to load (default: cloud-fraction/processing_config.json)")
    parser.add_argument("--dry-run", action="store_true", help="Print the generated cloud_fraction.py commands without executing them.")
    args = parser.parse_args()

    try:
        run_batch(args.config, dry_run=args.dry_run)
        return 0
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"Processing failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
