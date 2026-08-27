from pathlib import Path

from processing import (
    build_cloud_fraction_command,
    build_output_dir,
    filter_input_dirs,
    load_camera_coordinates,
    load_config,
    resolve_camera_settings,
)
from cloud_fraction import calculate_weighted_cloud_brightness
from brightness_calibration import derive_thresholds
import numpy as np


def test_load_config_defaults(tmp_path):
    config_path = tmp_path / "processing_config.json"
    config_path.write_text(
        '{"camera": "CAM1", "input_list": "../Data/CloudCam/img-for-analysis.txt", "base_dir": ".."}',
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg["camera"] == "CAM1"
    assert cfg["base_dir"] == str((tmp_path / "..").resolve())
    assert cfg["input_list"].endswith("Data/CloudCam/img-for-analysis.txt")


def test_filter_input_dirs_keeps_selected_camera():
    files = [
        Path("/repo/Data/CloudCam/CAM1/WMD_CAM1_JPG_20260804/day"),
        Path("/repo/Data/CloudCam/CAM2/WMD_CAM2_JPG_20260810/day"),
        Path("/repo/Data/CloudCam/CAM4/WMD_CAM4_JPG_20260807/day"),
    ]

    selected = filter_input_dirs(files, "CAM2")

    assert selected == [Path("/repo/Data/CloudCam/CAM2/WMD_CAM2_JPG_20260810/day")]


def test_build_output_dir_uses_camera_and_day_name():
    src_dir = Path("/repo/Data/CloudCam/CAM1/WMD_CAM1_JPG_20260808/day")
    output_root = Path("/repo/Data/CloudCamProcessed")

    assert build_output_dir(src_dir, output_root) == Path(
        "/repo/Data/CloudCamProcessed/CAM1/WMD_CAM1_JPG_20260808"
    )


def test_load_camera_coordinates_reads_placeholders(tmp_path):
    coord_file = tmp_path / "camera_coordinates.json"
    coord_file.write_text(
        '{"CAM1": {"lat": "", "lon": "", "alt": ""}, "CAM2": {"lat": 54.1, "lon": 11.2, "alt": 0.0}}',
        encoding="utf-8",
    )

    assert load_camera_coordinates(coord_file)["CAM1"]["lat"] == ""
    assert load_camera_coordinates(coord_file)["CAM2"]["lon"] == 11.2


def test_resolve_camera_settings_prefers_config_over_file():
    cfg = {
        "camera": "CAM2",
        "lat": 10.0,
        "lon": 20.0,
        "alt": 30.0,
    }

    coords = {"CAM2": {"lat": 54.1, "lon": 11.2, "alt": 0.0}}
    resolved = resolve_camera_settings(cfg, coords)

    assert resolved["lat"] == 10.0
    assert resolved["lon"] == 20.0
    assert resolved["alt"] == 30.0


def test_calculate_weighted_cloud_brightness_uses_solid_angle_weights():
    arr = np.array([
        [[10, 20, 30], [100, 110, 120]],
        [[200, 210, 220], [40, 50, 60]],
    ], dtype=np.uint8)
    classes = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    weights = np.array([[1.0, 3.0], [10.0, 2.0]])

    result = calculate_weighted_cloud_brightness(
        arr, classes, weights, min_effective_pixels=1
    )

    expected = np.average(arr[classes == 1], axis=0, weights=weights[classes == 1])
    assert np.allclose([result["mean_r"], result["mean_g"], result["mean_b"]], expected)
    assert result["total_mean_brightness"] == np.mean(expected)


def test_calculate_weighted_cloud_brightness_returns_nan_for_insufficient_pixels():
    arr = np.array([[[10, 20, 30], [100, 110, 120]]], dtype=np.uint8)
    classes = np.array([[1, 0]], dtype=np.uint8)
    weights = np.array([[1.0, 1.0]])

    result = calculate_weighted_cloud_brightness(
        arr, classes, weights, min_effective_pixels=2
    )

    assert all(np.isnan(value) for value in result.values())


def test_calculate_weighted_cloud_brightness_returns_nan_for_high_variability():
    arr = np.array([[[0, 0, 0], [100, 100, 100]]], dtype=np.uint8)
    classes = np.array([[1, 1]], dtype=np.uint8)
    weights = np.array([[1.0, 1.0]])

    result = calculate_weighted_cloud_brightness(
        arr, classes, weights, min_effective_pixels=1, std_threshold=10
    )

    assert all(np.isnan(value) for value in result.values())


def test_build_cloud_fraction_command_adds_brightness_flag():
    cfg = {
        "base_dir": "/repo",
        "cloud_fraction_script": "cloud-fraction/cloud_fraction.py",
        "calculate_brightness": True,
        "brightness_min_effective_pixels": 12,
        "brightness_std_threshold": 40,
    }

    command = build_cloud_fraction_command(cfg, Path("/repo/input"), Path("/repo/output"), None)

    assert "--calculate_brightness" in command
    assert command[command.index("--brightness_min_effective_pixels") + 1] == "12"
    assert command[command.index("--brightness_std_threshold") + 1] == "40"


def test_derive_thresholds_uses_reference_images_and_robust_variability():
    diagnostics = [
        {
            "effective_cloud_pixels": 100.0,
            "std_r": 10.0,
            "std_g": 12.0,
            "std_b": 14.0,
        },
        {
            "effective_cloud_pixels": 200.0,
            "std_r": 20.0,
            "std_g": 22.0,
            "std_b": 24.0,
        },
        {
            "effective_cloud_pixels": 300.0,
            "std_r": 30.0,
            "std_g": 32.0,
            "std_b": 34.0,
        },
        {
            "effective_cloud_pixels": 400.0,
            "std_r": 40.0,
            "std_g": 42.0,
            "std_b": 44.0,
        },
    ]

    thresholds = derive_thresholds(
        diagnostics,
        target_error_rgb=10.0,
        reference_quantile=0.75,
        mad_multiplier=3.0,
    )

    assert thresholds["calibration_reference_image_count"] == 1.0
    assert thresholds["brightness_min_effective_pixels"] == 68.0
    assert thresholds["brightness_std_threshold"] == 50.8956
