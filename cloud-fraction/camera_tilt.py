#!/usr/bin/env python3
"""
camera_tilt.py
===============

Bestimmt die Verdrehung (Nivellierung + Einnordung) der Wolkenkamera für
einen Tag, indem die tatsächliche Bildposition der Sonne über mehrere
Aufnahmen mit ihrer theoretisch berechneten Position verglichen wird
(PDF "Verwendung der Kamera VIVOTEK FE8174V als Wolkenkamera", Abschnitt 5).

Auch eine sorgfältig aufgestellte Kamera ist in der Praxis nie exakt
horizontal und nach Norden ausgerichtet, und Wind/Wetter können die
Ausrichtung von Tag zu Tag leicht verändern. Dieses Tool soll daher vor
der eigentlichen Bildauswertung (`cloud_fraction.py`) für jeden Tag separat
laufen und liefert eine Rotationsmatrix M, mit der gemessene
Kamerakoordinaten in echte Himmelskoordinaten umgerechnet werden können
(Swahr = M * Smess, Gl. 4).

Vorgehen
--------
1. Für jedes Bild eines Tagesordners:
   a) Theoretische Sonnenposition (Azimuth/Elevation) berechnen (pvlib).
   b) Falls die Sonne über dem Horizont steht: in der (unverdrehten)
      Kameraprojektion die erwartete Pixelposition bestimmen.
   c) In einem kleinen Suchfenster um diese erwartete Position die
      tatsächliche Sonnenposition im Bild finden (Schwerpunkt der  
      übersättigten/weißen Pixel, vgl. PDF Abschnitt 7.1, Fall 1).
2. Aus mindestens 3 solcher Beobachtungspaare (gemessen, wahr) wird die
   Rotationsmatrix M nach Gl. 4 bestimmt (bei mehr als 3 Beobachtungen über
   eine Ausgleichsrechnung / Pseudoinverse, was robuster gegen einzelne
   Ablesefehler ist als die Verwendung von genau drei Punkten).
3. Es werden Kennzahlen ausgegeben, mit denen sich die Verdrehung intuitiv
   interpretieren lässt (vgl. das Beispiel in PDF Abschnitt 5.3):
     - Wohin zeigt die Kamera-Zenitachse wirklich (Soll: Elevation 90°)?
       -> Gesamt-Kippwinkel der Kamera.
     - Wohin zeigt die Kamera-Nordmarkierung wirklich (Soll: Azimut 0°,
       Elevation 0°)?  -> Einnordungsfehler + Nord-Süd-Kippwinkel.
     - Wohin zeigt die Kamera-Ostachse wirklich (Soll: Azimut 90°,
       Elevation 0°)? -> Ost-West-Kippwinkel.
     - Determinante und Orthogonalität von M (Gütemaß, vgl. Abschnitt 5.1).
4. Die Matrix M wird zusammen mit den Kennzahlen in eine JSON-Datei
   geschrieben (z. B. tilt_20260620.json), die anschließend mit
   `cloud_fraction.py --tilt_matrix_file tilt_20260620.json` für die
   Bildauswertung dieses Tages verwendet werden kann.

Beispiel
--------
python camera_tilt.py \\
    --input_dir /data/2026-06-20/cam1 \\
    --lat 52.45 --lon 13.30 --alt 50 \\
    --fov_deg 180 \\
    --output_json /data/2026-06-20/tilt_20260620.json \\
    --plot /data/2026-06-20/tilt_20260620.png
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from camera_geometry import (
    CameraGeometry,
    az_el_to_cartesian,
    cartesian_to_az_el,
    height_angle_true_from_ideal,
    pixel_to_ideal_hemisphere,
    resolve_center_radius,
    save_tilt_matrix,
    sky_angles_to_pixel,
)

DEFAULT_TS_REGEX = re.compile(r"(\d{8})_(\d{6})")


def parse_timestamp(filename: str, regex: re.Pattern, utc_offset_hours: float) -> datetime:
    m = regex.search(filename)
    if not m:
        raise ValueError(f"No timestamp found in filename: {filename}")
    date_str, time_str = m.group(1), m.group(2)
    local_dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    return local_dt.replace(tzinfo=timezone(timedelta(hours=utc_offset_hours))).astimezone(timezone.utc)


def sun_position(dt_utc: datetime, geo: CameraGeometry) -> tuple[float, float]:
    """Returns (azimuth_deg, elevation_deg) of the sun at time dt_utc."""
    import pvlib

    times = pd.DatetimeIndex([dt_utc])
    solpos = pvlib.solarposition.get_solarposition(times, geo.lat, geo.lon, altitude=geo.alt)
    azimuth = float(solpos["azimuth"].iloc[0])
    elevation = float(solpos["apparent_elevation"].iloc[0])
    return azimuth, elevation


# ----------------------------------------------------------------------
# 1) Detect the sun's actual pixel position in an image
# ----------------------------------------------------------------------
def detect_sun_centroid(
    arr: np.ndarray,
    roi_center_xy: tuple[float, float],
    roi_radius_px: float,
    fisheye_center_xy: tuple[float, float],
    fisheye_radius_px: float,
    saturation_threshold: int = 253,
    min_sun_pixels: int = 5,
    max_sun_pixel_fraction: float = 0.4,
) -> tuple[float, float] | None:
    """Looks for the sun as a cluster of (near-)saturated white pixels
    (PDF Sec. 7.1, condition 1: R,G,B >= 250) inside a search window
    around the theoretically expected position, and returns the
    brightness-weighted centroid. Returns None if no plausible sun blob
    was found (e.g. cloud cover, or the bright area is too large to be
    just the sun -> overcast/glare).
    """
    height, width = arr.shape[:2]
    rx, ry = roi_center_xy
    x0 = max(0, int(rx - roi_radius_px))
    x1 = min(width, int(rx + roi_radius_px) + 1)
    y0 = max(0, int(ry - roi_radius_px))
    y1 = min(height, int(ry + roi_radius_px) + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    roi = arr[y0:y1, x0:x1]
    r, g, b = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    sat = (r >= saturation_threshold) & (g >= saturation_threshold) & (b >= saturation_threshold)

    # also stay within the actual fisheye image circle
    yy, xx = np.mgrid[y0:y1, x0:x1]
    fcx, fcy = fisheye_center_xy
    inside = (xx - fcx) ** 2 + (yy - fcy) ** 2 <= fisheye_radius_px ** 2
    sat &= inside

    n = int(sat.sum())
    roi_area = (x1 - x0) * (y1 - y0)
    if n < min_sun_pixels or n > max_sun_pixel_fraction * roi_area:
        return None

    ys, xs = np.nonzero(sat)
    return float(xs.mean() + x0), float(ys.mean() + y0)


# ----------------------------------------------------------------------
# 2) Collect (measured, true) sun-direction pairs over the day
# ----------------------------------------------------------------------
def collect_sun_observations(
    input_dir: Path,
    geo: CameraGeometry,
    ts_regex: re.Pattern,
    utc_offset_hours: float,
    edge_margin_px: float,
    roi_radius_px: float,
    min_sun_pixels: int,
    min_elevation_deg: float,
) -> list[dict]:
    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg")
    )
    if not image_paths:
        raise FileNotFoundError(f"No JPG image found in {input_dir}.")

    observations = []
    for img_path in image_paths:
        try:
            dt_utc = parse_timestamp(img_path.name, ts_regex, utc_offset_hours)
        except ValueError as e:
            print(f"  Skipped (timestamp): {e}", file=sys.stderr)
            continue

        az_true, el_true = sun_position(dt_utc, geo)
        if el_true < min_elevation_deg:
            continue  # sun too low / below horizon -> unreliable or unusable

        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        height, width = arr.shape[:2]
        cx, cy, r_max = resolve_center_radius(geo, width, height)

        # expected pixel position assuming a perfectly levelled & northed
        # camera (M = identity) -- just used as the search-window center
        nominal_geo = CameraGeometry(
            lat=geo.lat, lon=geo.lon, alt=geo.alt, fov_deg=geo.fov_deg,
            center_x=geo.center_x, center_y=geo.center_y, radius_px=geo.radius_px,
            apply_height_correction=geo.apply_height_correction, tilt_matrix=None,
        )
        nominal_xy = sky_angles_to_pixel(az_true, el_true, cx, cy, r_max, nominal_geo)

        detected = detect_sun_centroid(
            arr, nominal_xy, roi_radius_px,
            (cx, cy), r_max - edge_margin_px,
            min_sun_pixels=min_sun_pixels,
        )
        if detected is None:
            continue

        px, py = detected
        az_ideal_cam, eps_ideal_cam = pixel_to_ideal_hemisphere(px, py, cx, cy, r_max)
        if geo.apply_height_correction:
            eps_true_cam = float(height_angle_true_from_ideal(eps_ideal_cam))
        else:
            eps_true_cam = float(eps_ideal_cam)

        observations.append({
            "file": img_path.name,
            "time_utc": dt_utc,
            "pixel_xy": (px, py),
            "az_true_sun": az_true,
            "el_true_sun": el_true,
            "az_cam_sun": float(az_ideal_cam),
            "el_cam_sun": eps_true_cam,
        })
        print(
            f"  {img_path.name}: sun at px=({px:.1f},{py:.1f})  "
            f"true(az={az_true:6.2f}°,el={el_true:5.2f}°)  "
            f"cam(az={az_ideal_cam:6.2f}°,el={eps_true_cam:5.2f}°)"
        )

    return observations


# ----------------------------------------------------------------------
# 3) Determine rotation matrix M (PDF Sec. 5.1, Eq. 4) and tilt diagnostics
# ----------------------------------------------------------------------
def compute_rotation_matrix(observations: list[dict]) -> np.ndarray:
    """M is found from  Swahr = M * Smess  (Eq. 4). With exactly 3
    observations this is the exact PDF recipe (M = Swahr @ inv(Smess));
    with more than 3 we use the least-squares / pseudo-inverse solution,
    which is more robust against individual read-off errors."""
    s_mess = np.array([
        az_el_to_cartesian(o["az_cam_sun"], o["el_cam_sun"]) for o in observations
    ]).T  # 3 x N
    s_wahr = np.array([
        az_el_to_cartesian(o["az_true_sun"], o["el_true_sun"]) for o in observations
    ]).T  # 3 x N

    if s_mess.shape[1] == 3:
        M = s_wahr @ np.linalg.inv(s_mess)
    else:
        M = s_wahr @ np.linalg.pinv(s_mess)
    return M


def rotation_quality(M: np.ndarray) -> dict:
    """Orthogonality diagnostics as described in PDF Sec. 5.1/5.3: column
    lengths, pairwise angles between columns, and determinant. A perfect
    rotation matrix has unit-length, mutually orthogonal columns and
    det(M) = 1."""
    cols = [M[:, i] for i in range(3)]
    lengths = [float(np.linalg.norm(c)) for c in cols]

    def angle(a, b):
        cos_ang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))

    angles = {
        "angle_col1_col2_deg": angle(cols[0], cols[1]),
        "angle_col2_col3_deg": angle(cols[1], cols[2]),
        "angle_col3_col1_deg": angle(cols[2], cols[0]),
    }
    return {
        "column_lengths": lengths,
        **angles,
        "determinant": float(np.linalg.det(M)),
    }


def tilt_summary(M: np.ndarray) -> dict:
    """Maps the camera's East/North/Zenith basis vectors through M and
    reads off azimuth/elevation, exactly as in the PDF Sec. 5.3 example.
    For a perfectly levelled & northed camera:
      East   -> azimuth  90°, elevation 0°
      North  -> azimuth   0°, elevation 0°
      Zenith -> elevation 90° (azimuth undefined)
    Deviations from these directly give the tilt / mis-alignment angles.
    """
    east_true = M[:, 0]
    north_true = M[:, 1]
    zenith_true = M[:, 2]

    az_e, el_e = cartesian_to_az_el(*east_true)
    az_n, el_n = cartesian_to_az_el(*north_true)
    az_z, el_z = cartesian_to_az_el(*zenith_true)

    return {
        "east_axis_true_azimuth_deg": float(az_e),
        "east_axis_true_elevation_deg": float(el_e),       # ideally 0 -> east-west tilt
        "north_axis_true_azimuth_deg": float(az_n),         # ideally 0 -> einnordung error
        "north_axis_true_elevation_deg": float(el_n),       # ideally 0 -> north-south tilt
        "zenith_axis_true_azimuth_deg": float(az_z),
        "zenith_axis_true_elevation_deg": float(el_z),      # ideally 90
        "total_tilt_from_vertical_deg": float(90.0 - el_z),
        "northing_error_deg": float(az_n),
        "eastwest_tilt_deg": float(el_e),
        "northsouth_tilt_deg": float(el_n),
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Determine the daily tilt (levelling + northing) of the Wolkenkamera "
                    "from the discrepancy between the sun's actual and theoretical image position."
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder with JPG images of one day")
    parser.add_argument("--output_json", type=Path, required=True, help="Where to write the tilt matrix (JSON)")

    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--alt", type=float, default=0.0)

    parser.add_argument("--fov_deg", type=float, default=180.0)
    parser.add_argument("--center_x", type=float, default=None)
    parser.add_argument("--center_y", type=float, default=None)
    parser.add_argument("--circle_radius_px", type=float, default=None)
    parser.add_argument("--edge_margin_px", type=float, default=5.0)
    parser.add_argument(
        "--no_height_correction", action="store_true",
        help="Disable the lens height-angle correction (PDF Sec. 4.1) when "
             "deriving the camera-frame sun position.",
    )

    parser.add_argument("--timestamp_regex", type=str, default=None)
    parser.add_argument("--utc_offset_hours", type=float, default=0.0)

    parser.add_argument(
        "--min_elevation_deg", type=float, default=15.0,
        help="Ignore observations where the theoretical sun elevation is below "
             "this (low sun -> unreliable detection / strong refraction near horizon).",
    )
    parser.add_argument(
        "--roi_radius_px", type=float, default=80.0,
        help="Radius of the search window (px) around the theoretical sun position "
             "used to look for the actual sun blob.",
    )
    parser.add_argument("--min_sun_pixels", type=int, default=5, help="Minimum number of saturated pixels to count as a sun detection")
    parser.add_argument("--min_observations", type=int, default=3, help="Minimum number of usable sun detections required")
    parser.add_argument("--plot", type=Path, default=None, help="Optional PNG with measured vs. theoretical sun track")

    args = parser.parse_args()
    ts_regex = re.compile(args.timestamp_regex) if args.timestamp_regex else DEFAULT_TS_REGEX

    geo = CameraGeometry(
        lat=args.lat, lon=args.lon, alt=args.alt, fov_deg=args.fov_deg,
        center_x=args.center_x, center_y=args.center_y, radius_px=args.circle_radius_px,
        apply_height_correction=not args.no_height_correction,
    )

    print(f"Looking for sun positions in {args.input_dir} ...")
    observations = collect_sun_observations(
        args.input_dir, geo, ts_regex, args.utc_offset_hours,
        args.edge_margin_px, args.roi_radius_px, args.min_sun_pixels,
        args.min_elevation_deg,
    )

    if len(observations) < args.min_observations:
        print(
            f"\nOnly {len(observations)} usable sun detection(s) found "
            f"(need >= {args.min_observations}). Cannot determine a tilt matrix "
            f"for this day -- likely overcast. Falling back to the identity "
            f"matrix (no tilt correction) for {args.input_dir}.",
            file=sys.stderr,
        )
        M = np.eye(3)
        quality = rotation_quality(M)
        tilt = tilt_summary(M)
        save_tilt_matrix(
            args.output_json, M,
            meta={
                "status": "fallback_identity",
                "n_observations": len(observations),
                "quality": quality,
                "tilt_summary": tilt,
            },
        )
        print(f"Wrote fallback tilt matrix to {args.output_json}")
        return

    M = compute_rotation_matrix(observations)
    quality = rotation_quality(M)
    tilt = tilt_summary(M)

    print("\nRotation matrix M (camera -> true sky):")
    print(np.array2string(M, precision=6, suppress_small=True))
    print(f"\ndet(M) = {quality['determinant']:.5f}  (1.0 = perfect rotation)")
    print(
        f"Column lengths: {', '.join(f'{l:.4f}' for l in quality['column_lengths'])}  "
        "(1.0 each = perfect rotation)"
    )
    print(
        "Pairwise column angles: "
        f"{quality['angle_col1_col2_deg']:.3f}°, "
        f"{quality['angle_col2_col3_deg']:.3f}°, "
        f"{quality['angle_col3_col1_deg']:.3f}°  (90° each = perfect rotation)"
    )

    print("\nTilt summary:")
    print(f"  Camera zenith axis really points to elevation {tilt['zenith_axis_true_elevation_deg']:.3f}° "
          f"(ideal: 90°)  ->  total tilt from vertical: {tilt['total_tilt_from_vertical_deg']:.3f}°")
    print(f"  Camera north marking really points to azimuth {tilt['north_axis_true_azimuth_deg']:.3f}° "
          f"(ideal: 0°)  ->  northing error: {tilt['northing_error_deg']:.3f}°")
    print(f"  Camera north axis elevation: {tilt['north_axis_true_elevation_deg']:.3f}° "
          f"(ideal: 0°)  ->  north-south tilt: {tilt['northsouth_tilt_deg']:.3f}°")
    print(f"  Camera east axis elevation:  {tilt['east_axis_true_elevation_deg']:.3f}° "
          f"(ideal: 0°)  ->  east-west tilt:  {tilt['eastwest_tilt_deg']:.3f}°")

    save_tilt_matrix(
        args.output_json, M,
        meta={
            "status": "ok",
            "n_observations": len(observations),
            "observation_times_utc": [o["time_utc"].isoformat() for o in observations],
            "quality": quality,
            "tilt_summary": tilt,
        },
    )
    print(f"\nWrote tilt matrix to {args.output_json}")

    if args.plot is not None:
        plot_observations(observations, args.plot)
        print(f"Wrote diagnostic plot to {args.plot}")


def plot_observations(observations: list[dict], out_png: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    px = [o["pixel_xy"][0] for o in observations]
    py = [o["pixel_xy"][1] for o in observations]
    ax.scatter(px, py, c="orange", label="detected sun position", zorder=3)
    for o in observations:
        ax.annotate(o["time_utc"].strftime("%H:%M"), o["pixel_xy"], fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title("Detected sun positions used for tilt determination")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
