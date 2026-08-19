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

    result = calculate_weighted_cloud_brightness(arr, classes, weights)

    expected = np.average(arr[classes == 1], axis=0, weights=weights[classes == 1])
    assert np.allclose([result["mean_r"], result["mean_g"], result["mean_b"]], expected)
    assert result["total_mean_brightness"] == np.mean(expected)


def test_build_cloud_fraction_command_adds_brightness_flag():
    cfg = {
        "base_dir": "/repo",
        "cloud_fraction_script": "cloud-fraction/cloud_fraction.py",
        "calculate_brightness": True,
    }

    command = build_cloud_fraction_command(cfg, Path("/repo/input"), Path("/repo/output"), None)

    assert "--calculate_brightness" in command
