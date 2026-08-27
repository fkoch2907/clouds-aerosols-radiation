from combine_cloud_fraction_timeseries import combine_timeseries


def test_combine_timeseries_adds_camera_column_and_sorts_days(tmp_path):
    input_dir = tmp_path / "CAM1"
    header = "timestamp_utc,cloud_fraction_total\n"
    (input_dir / "day_b").mkdir(parents=True)
    (input_dir / "day_a").mkdir()
    (input_dir / "day_b" / "cloud_fraction_timeseries.csv").write_text(
        header + "2026-08-02T00:00:00+00:00,0.2\n", encoding="utf-8"
    )
    (input_dir / "day_a" / "cloud_fraction_timeseries.csv").write_text(
        header + "2026-08-01T00:00:00+00:00,0.1\n", encoding="utf-8"
    )

    output_path = combine_timeseries(input_dir, tmp_path / "combined.csv")

    assert output_path.read_text(encoding="utf-8") == (
        "camera,timestamp_utc,cloud_fraction_total\n"
        "CAM1,2026-08-01T00:00:00+00:00,0.1\n"
        "CAM1,2026-08-02T00:00:00+00:00,0.2\n"
    )
