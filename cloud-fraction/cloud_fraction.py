#!/usr/bin/env python3
"""
cloud_fraction.py
==================

Bestimmt die Cloud Fraction aus Fisheye-Himmelskamerabildern.

Pipeline
--------
1. Liest die zwei Farb-Häufigkeitsdateien (Himmel + Wolken, je 256^3 Byte,
   Reihenfolge R-G-B wie in Colorfiles.txt beschrieben) ein.
2. Berechnet für jede der 16.777.216 Farben die Anzahl "gesetzter" Nachbarn
   im 3x3x3-Würfel (das Pascal-Äquivalent von GesetzteInUmgebung), separat
   für Himmel- und Wolken-Datei -- vektorisiert mit einer 3D-Faltung statt
   Tripel-Loop.
3. Baut daraus eine Klassifikationstabelle (256x256x256 uint8) analog zu
   LoadFromFiles: hoUnbekannt/hoWolke/hoHimmel/hoBeides, plus Sonderfälle
   hoMaske (schwarz) und hoSonne (weiß).
4. Für jedes Bild eines Tagesordners:
   a) Sonnenposition (Azimuth/Elevation) zur Aufnahmezeit über die
      Kamerakoordinaten berechnen (pvlib).
   b) Sonnenposition per Fisheye-Projektion in Bildkoordinaten umrechnen
      und einen Kreis um die Sonne ausmaskieren.
   c) Äußeren Rand (alles außerhalb des Fisheye-Kreises) ausmaskieren.
   d) Jeden verbleibenden Pixel über die Klassifikationstabelle einfärben
      und ein Ausgabebild (gleiche Größe wie Original) schreiben.
   e) Cloud Fraction = Wolken-Pixel / (Wolken+Himmel+Beides)-Pixel.
5. Schreibt eine CSV mit Zeitstempel + Cloud Fraction pro Bild und plottet
   die Tageszeitreihe.

1. Reads the two color frequency files (sky + clouds, 256^3 bytes each, in R-G-B order as described in Colorfiles.txt).
2. For each of the 16,777,216 colors, calculates the number of “set” neighbors in the 3x3x3 cube (the Pascal equivalent of “SetsInNeighborhood”), separately for the sky and cloud files—vectorized using a 3D convolution instead of a triple loop.
3. Construct a classification table (256x256x256 uint8) from this, analogous to LoadFromFiles: hoUnknown/hoCloud/hoSky/hoBoth, plus special cases hoMask (black) and hoSun (white).
4. For each image in a daily folder:
a) Calculate the sun’s position (azimuth/elevation) at the time of capture using the
      camera coordinates (pvlib).
b) Convert the sun’s position to image coordinates via fisheye projection
 and mask a circle around the sun.
c) Mask the outer edge (everything outside the fisheye circle).
d) Color-code each remaining pixel using the classification table
and write an output image (same size as the original).
e) Cloud Fraction = cloud pixels / (clouds + sky + both) pixels.
5. Write a CSV file with timestamps + cloud fraction per image and plot the time-of-day series.

Abhängigkeiten: numpy, scipy, pillow, pvlib, matplotlib
    pip install numpy scipy pillow pvlib matplotlib

Beispiel
--------
python cloud_fraction.py \\
    --input_dir /data/2026-06-20/cam1 \\
    --output_dir /data/2026-06-20/cam1_out \\
    --himmel_file FE1_Himmel.dat \\
    --wolken_file FE1_Wolken.dat \\
    --lat 52.45 --lon 13.30 --alt 50 \\
    --fov_deg 180 \\
    --edge_margin_px 5 \\
    --sun_radius_px 25

Annahmen, die ggf. an die eigene Kamera angepasst werden müssen
-----------------------------------------------------------------
- Die Fisheye-Optik wird als equidistante Projektion angenommen:
  r(zenith) = R_max * zenith_rad / (FOV_rad/2)
  Falls die Kamera eine andere Projektion (z.B. equisolid) hat, einfach
  die Funktion `zenith_to_radius` anpassen.
- Bildmitte = Kreismittelpunkt des Fisheye-Bilds. Falls die Optik nicht
  exakt zentriert ist, --center_x/--center_y setzen.
- Der Zeitstempel wird per Default aus dem Dateinamen als
  YYYYMMDD_HHMMSS extrahiert (--timestamp_regex anpassbar) und als
  lokale Zeit in UTC angenommen (--utc_offset_hours anpassbar).

The fisheye lens is assumed to use an equidistant projection:
  r(zenith) = R_max * zenith_rad / (FOV_rad/2)
  If the camera uses a different projection (e.g., equisolid), simply
  adjust the `zenith_to_radius` function.
- Image center = center of the fisheye image. If the lens is not
  exactly centered, set --center_x/--center_y.
- By default, the timestamp is extracted from the filename as
  YYYYMMDD_HHMMSS (--timestamp_regex is customizable) and assumed to be
  local time in UTC (--utc_offset_hours is customizable).

Translated with DeepL.com (free version)
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
from scipy.ndimage import convolve

from camera_geometry import (
    CameraGeometry,
    height_angle_true_from_ideal,
    pixel_to_sky_angles,
    resolve_center_radius,
    sky_angles_to_pixel,
)
from land_sea_mask import build_land_sea_mask, build_sector_mask

# ----------------------------------------------------------------------
# classification-codes (analog to Pascal-constants in Colorfiles.txt)
# ----------------------------------------------------------------------
HO_UNKNOWN = 0
HO_CLOUD = 1
HO_SKY = 2
HO_BOTH = 3
HO_SUN = 4
HO_MASK = 5

# colors representing classes in output images (R,G,B)
DISPLAY_COLORS = {
    HO_UNKNOWN: (128, 128, 128),  # grey
    HO_CLOUD:     (255, 255, 255),  # white
    HO_SKY:    (0, 100, 255),    # blue
    HO_BOTH:    (255, 165, 0),    # orange
    HO_SUN:     (255, 255, 0),    # yellow
    HO_MASK:     (0, 0, 0),        # black
}

N = 256  # edge length of color cube


# ----------------------------------------------------------------------
# 1) read color files + build classification table
# ----------------------------------------------------------------------
def load_color_cube(path: Path) -> np.ndarray:
    """Reads a 256^3-Byte file in the order described in Colorfiles.txt:
    outer loop R, then G, then B."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size != N ** 3:
        raise ValueError(
            f"{path} has {data.size} Bytes, expected were {N**3} (256^3)."
        )
    return data.reshape(N, N, N)  # order of axes: [R, G, B]


def count_set_neighbors_in_neighborhood(cube: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of `SetsInNeighborhood`.

    For every cell, this returns the number of “set” values (> 0) in the
    3x3x3 neighborhood, including the cell itself. The edges are handled by
    clamping at the borders, which is reproduced exactly by convolution with
    `mode="nearest"`.
    """
    set_mask = (cube > 0).astype(np.uint8)
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    # mode='nearest' corresponds to Clamping (Max(0,R-1), Min(255,R+1)) in Pascal-code
    counts = convolve(set_mask, kernel, mode="nearest")
    return counts


def build_classification_table(sky_path: Path, cloud_path: Path) -> np.ndarray:
    """Builds recognition table [R,G,B] -> class code."""
    sky = load_color_cube(sky_path)
    cloud = load_color_cube(cloud_path)

    h = count_set_neighbors_in_neighborhood(sky)
    w = count_set_neighbors_in_neighborhood(cloud)

    table = np.full((N, N, N), HO_UNKNOWN, dtype=np.uint8)
    table[(w > 0) & (w == h)] = HO_BOTH
    table[w > h] = HO_CLOUD
    table[h > w] = HO_SKY   # other cases(w == h == 0) remain HO_UNKNOWN

    table[0, 0, 0] = HO_MASK   # black
    table[255, 255, 255] = HO_SUN  # white
    return table


# ----------------------------------------------------------------------
# 2) Sun position -> image coordinates (Fisheye-geometry)
# ----------------------------------------------------------------------
# CameraGeometry, the height-angle correction (Sec. 4.1) and the pixel
# <-> sky-angle transforms (Sec. 4.2/4.3, Sec. 5) now live in
# camera_geometry.py so they can be shared with camera_tilt.py and reused
# by future image-analysis tools. `CameraGeometry.tilt_matrix` carries the
# day's rotation matrix M as determined by `camera_tilt.py`; if it is not
# set, M defaults to the identity (perfectly levelled & northed camera).

def sun_position(dt_utc: datetime, geo: CameraGeometry) -> tuple[float, float]:
    """Returns (azimuth_deg, elevation_deg) of the sun at time dt_utc."""
    import pvlib

    times = pd.DatetimeIndex([dt_utc])
    solpos = pvlib.solarposition.get_solarposition(
        times, geo.lat, geo.lon, altitude=geo.alt
    )
    azimuth = float(solpos["azimuth"].iloc[0])      # 0=north, clockwise
    elevation = float(solpos["apparent_elevation"].iloc[0])
    return azimuth, elevation


def sun_pixel_position(
    azimuth_deg: float,
    elevation_deg: float,
    width: int,
    height: int,
    geo: CameraGeometry,
) -> tuple[int, int] | None:
    """Converts the sun's true azimuth/elevation to image coordinates,
    including the lens height-angle correction (Sec. 4.1, Eq. 3) and the
    day's camera-tilt rotation M (Sec. 5), if set on `geo`. Returns None
    if the sun is below the horizon (night image)."""
    if elevation_deg <= 0:
        return None
    cx, cy, r_max = resolve_center_radius(geo, width, height)
    px, py = sky_angles_to_pixel(azimuth_deg, elevation_deg, cx, cy, r_max, geo)
    return int(round(px)), int(round(py))


def circular_mask(height: int, width: int, cx: float, cy: float, radius: float) -> np.ndarray:
    """Boolean mask, True = within the circle."""
    yy, xx = np.mgrid[0:height, 0:width]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def load_custom_mask(mask_path: Path, width: int, height: int) -> np.ndarray:
    """Load an optional custom mask from .npy or .png.

    The returned array is boolean with True for pixels that should be masked.
    For PNG files, non-zero pixels are treated as masked.
    """
    suffix = mask_path.suffix.lower()
    if suffix == ".npy":
        mask = np.load(mask_path)
    elif suffix == ".png":
        mask = np.array(Image.open(mask_path).convert("L"))
    else:
        raise ValueError(f"Unsupported mask format '{mask_path.suffix}'. Use .npy or .png.")

    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2)
    else:
        mask = mask > 0

    if mask.shape != (height, width):
        raise ValueError(
            f"Custom mask {mask_path} has shape {mask.shape}, expected {(height, width)}."
        )

    return mask


def solid_angle_weight_map(
    height: int,
    width: int,
    cx: float,
    cy: float,
    a: float,
    geo: CameraGeometry,
) -> np.ndarray:
    """Per-pixel weight proportional to the solid angle that one pixel
    covers on the sky (PDF Sec. 6.2):

        (1 px)^2 ≙ (cos ε / z) * 0.1704e-5 sr ,

    with z = 1 - eps_ideal/90 (Eq. 1) the zenith-distance fraction read
    directly off the pixel's distance to the image centre, and ε the
    *actual* height angle of that pixel on the real sky -- i.e. after
    both the lens height-angle correction (Eq. 2) and, if known, the
    day's camera-tilt rotation M (Sec. 5) have been applied via
    `pixel_to_sky_angles`. We drop the constant factor since only the
    ratio of summed weights is needed for the cloud fraction; that
    constant cancels out.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    x0 = xx - cx
    y0 = yy - cy
    r = np.sqrt(x0 ** 2 + y0 ** 2)
    z = np.clip(r / a, 0.0, 1.0)  # = 1 - eps_ideal/90 (Eq. 1), unaffected by tilt

    _az_true, eps_true = pixel_to_sky_angles(xx, yy, cx, cy, a, geo)
    cos_eps = np.cos(np.deg2rad(eps_true))
    z_safe = np.maximum(z, 1e-6)  # avoid 0/0 right at the zenith pixel(s)
    weight = cos_eps / z_safe
    return weight


# ----------------------------------------------------------------------
# 3) Timestamp from filename
# ----------------------------------------------------------------------
DEFAULT_TS_REGEX = re.compile(r"(\d{8})_(\d{6})")


def parse_timestamp(filename: str, regex: re.Pattern, utc_offset_hours: float) -> datetime:
    m = regex.search(filename)
    if not m:
        raise ValueError(f"No timestamp found in filename: {filename}")
    date_str, time_str = m.group(1), m.group(2)
    local_dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    return local_dt.replace(tzinfo=timezone(timedelta(hours=utc_offset_hours))).astimezone(timezone.utc)


# ----------------------------------------------------------------------
# 4) Process an image
# ----------------------------------------------------------------------
def process_image(
    img_path: Path,
    out_path: Path,
    class_table: np.ndarray,
    geo: CameraGeometry,
    edge_margin_px: float,
    sun_radius_px: float,
    dt_utc: datetime,
    custom_mask: np.ndarray | None = None,
    land_mask: np.ndarray | None = None,
    sea_mask: np.ndarray | None = None,
) -> dict:
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)  # (H, W, 3) uint8
    height, width, _ = arr.shape

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    classes = class_table[r, g, b]  # (H, W) Klassen-Code je Pixel

    # --- mask outer edge ---
    cx, cy, r_max = resolve_center_radius(geo, width, height)
    fov_circle = circular_mask(height, width, cx, cy, r_max - edge_margin_px)
    mask_outer_edge = ~fov_circle
    if custom_mask is not None:
        mask_outer_edge = mask_outer_edge | custom_mask
    classes[mask_outer_edge] = HO_MASK

    # --- mask sun circle ---
    azimuth_deg, elevation_deg = sun_position(dt_utc, geo)
    sun_xy = sun_pixel_position(azimuth_deg, elevation_deg, width, height, geo)
    if sun_xy is not None:
        sx, sy = sun_xy
        sun_circle = circular_mask(height, width, sx, sy, sun_radius_px)
        classes[sun_circle] = HO_SUN

    if custom_mask is not None:
        classes[custom_mask] = HO_MASK

    # --- calculate cloud fraction using solid angles, not raw pixel counts ---
    # PDF Sec. 6.2/7.5: a pixel near the zenith covers a much smaller solid
    # angle than a pixel near the horizon, so pixels must be weighted by the
    # solid angle they represent before forming the cloud/sky ratio
    # (b1 = Ω(Wolke) / (Ω(Himmel)+Ω(Wolke))).
    weights = solid_angle_weight_map(height, width, cx, cy, r_max, geo)

    def region_fraction(region_mask: np.ndarray | None) -> float:
        # region_mask=None -> whole image (total). Cloud/sky classes already
        # exclude obstructed/masked/sun pixels (HO_MASK, HO_SUN, HO_UNKNOWN,
        # HO_BOTH), so no separate validity mask is needed here.
        cloud_sel = classes == HO_CLOUD
        sky_sel = classes == HO_SKY
        if region_mask is not None:
            cloud_sel = cloud_sel & region_mask
            sky_sel = sky_sel & region_mask
        cloud_omega = weights[cloud_sel].sum()
        sky_omega = weights[sky_sel].sum()
        valid_omega = cloud_omega + sky_omega
        return cloud_omega / valid_omega if valid_omega > 0 else np.nan

    cloud_fraction = {
        "total": region_fraction(None),
        "land": region_fraction(land_mask) if land_mask is not None else np.nan,
        "sea": region_fraction(sea_mask) if sea_mask is not None else np.nan,
    }

    # --- create and save output image ---
    out_arr = np.zeros_like(arr)
    for code, color in DISPLAY_COLORS.items():
        out_arr[classes == code] = color
    Image.fromarray(out_arr, mode="RGB").save(out_path)

    return cloud_fraction


# ----------------------------------------------------------------------
# 5) directory workflow + plot
# ----------------------------------------------------------------------
def process_folder(
    input_dir: Path,
    output_dir: Path,
    class_table: np.ndarray,
    geo: CameraGeometry,
    edge_margin_px: float,
    sun_radius_px: float,
    ts_regex: re.Pattern,
    utc_offset_hours: float,
    custom_mask_path: Path | None = None,
    coastline_bearing_deg: float | None = None,
    sea_side: str = "cw",
    boundaries: list[tuple[float, str]] | None = None,
) -> Path:
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg")
    )
    if not image_paths:
        raise FileNotFoundError(f"No JPG image found in {input_dir}.")

    custom_mask = None
    if custom_mask_path is not None:
        with Image.open(image_paths[0]) as first_img:
            width, height = first_img.size
        custom_mask = load_custom_mask(custom_mask_path, width, height)

    # Land/sea mask is static for a fixed camera -- build it once from the
    # first image's dimensions and reuse for every frame in the folder,
    # rather than recomputing it per image.
    land_mask = sea_mask = None
    sector_masks = None
    if boundaries is not None:
        # Use sector-based boundary masking
        first_img = np.array(Image.open(image_paths[0]))
        h0, w0 = first_img.shape[:2]
        sector_masks = build_sector_mask(geo, w0, h0, boundaries)
        print(f"Built sector mask from boundaries: "
              f"{', '.join(f'{label}={mask.sum()} px' for label, mask in sector_masks.items())}")
        # Set land_mask and sea_mask if they exist in the sectors
        if "land" in sector_masks:
            land_mask = sector_masks["land"]
        if "sea" in sector_masks:
            sea_mask = sector_masks["sea"]
    elif coastline_bearing_deg is not None:
        # Use bearing-based half-plane masking
        first_img = np.array(Image.open(image_paths[0]))
        h0, w0 = first_img.shape[:2]
        land_mask, sea_mask = build_land_sea_mask(geo, w0, h0, coastline_bearing_deg, sea_side)
        print(f"Built land/sea mask (bearing={coastline_bearing_deg}, sea_side={sea_side}): "
              f"{land_mask.sum()} land px, {sea_mask.sum()} sea px")

    results = []
    for img_path in image_paths:
        try:
            dt_utc = parse_timestamp(img_path.name, ts_regex, utc_offset_hours)
        except ValueError as e:
            print(f"  Skipped: {e}", file=sys.stderr)
            continue

        out_path = output_dir / f"{img_path.stem}_classified.jpg"
        cf = process_image(
            img_path, out_path, class_table, geo,
            edge_margin_px, sun_radius_px, dt_utc,
            custom_mask=custom_mask,
            land_mask=land_mask, sea_mask=sea_mask,
        )
        results.append((dt_utc, cf))
        print(
            f"  {img_path.name}: total = {cf['total']:.3f}, "
            f"land = {cf['land']:.3f}, sea = {cf['sea']:.3f}"
        )

    csv_path = output_dir / "cloud_fraction_timeseries.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "cloud_fraction_total", "cloud_fraction_land", "cloud_fraction_sea"])
        for dt_utc, cf in results:
            writer.writerow([dt_utc.isoformat(), f"{cf['total']:.4f}", f"{cf['land']:.4f}", f"{cf['sea']:.4f}"])

    plot_timeseries(results, output_dir / "cloud_fraction_timeseries.png", input_dir.name)
    return csv_path


def plot_timeseries(results: list[tuple[datetime, dict]], out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if not results:
        return
    times = [r[0] for r in results]
    totals = [r[1]["total"] for r in results]
    lands = [r[1]["land"] for r in results]
    seas = [r[1]["sea"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, totals, marker="o", linestyle="-", markersize=3, label="Total", color="black")
    if not all(np.isnan(v) for v in lands):
        ax.plot(times, lands, marker="o", linestyle="--", markersize=3, label="Land", color="tab:green")
    if not all(np.isnan(v) for v in seas):
        ax.plot(times, seas, marker="o", linestyle="--", markersize=3, label="Sea", color="tab:blue")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Cloud fraction")
    ax.set_xlabel("Time (UTC)")
    ax.set_title(f"Cloud fraction time series – {title}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cloud fraction from Fisheye cloud camera images")
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder with JPG images of one day")
    parser.add_argument("--output_dir", type=Path, required=True, help="Target folder for output")
    parser.add_argument("--himmel_file", type=Path, required=True, help="Color file 'Sky' (FE1/FE3)")
    parser.add_argument("--wolken_file", type=Path, required=True, help="Color file 'Clouds' (FE1/FE3)")

    parser.add_argument("--lat", type=float, required=True, help="Latitude of the camera")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of the camera")
    parser.add_argument("--alt", type=float, default=0.0, help="Height of the camera above sea level (m)")

    parser.add_argument("--fov_deg", type=float, default=180.0, help="Full field of view of the fisheye lens")
    parser.add_argument("--center_x", type=float, default=None, help="Image x-coordinate of the fisheye center (default: image center)")
    parser.add_argument("--center_y", type=float, default=None, help="Image y-coordinate of the fisheye center (default: image center)")
    parser.add_argument("--circle_radius_px", type=float, default=None, help="Radius of the fisheye image circle in pixels (default: min(w,h)/2)")
    parser.add_argument("--edge_margin_px", type=float, default=5.0, help="Additional margin to be subtracted from the fisheye image circle")
    parser.add_argument("--sun_radius_px", type=float, default=25.0, help="Radius of the sun circle to be cropped in pixels")
    parser.add_argument(
        "--custom_mask_file", type=Path, default=None,
        help="Optional additional mask file (.npy or .png) with the same size as the input images.",
    )
    parser.add_argument(
        "--no_height_correction", action="store_true",
        help="Disable the camera-specific height-angle correction polynomials "
             "(PDF Sec. 4.1, Eq. 2/3) and use the plain ideal equidistant "
             "projection instead.",
    )
    parser.add_argument(
        "--tilt_matrix_file", type=Path, default=None,
        help="JSON file with the day's camera-tilt rotation matrix M, as "
             "written by camera_tilt.py. If omitted, the camera is assumed "
             "to be perfectly levelled and northed (M = identity).",
    )
    parser.add_argument(
        "--coastline_bearing_deg", type=float, default=None,
        help="Real compass bearing (0=N, clockwise) of the coastline, as "
             "read off a map. If set, land/sea cloud fraction is computed "
             "in addition to the total, using a straight line through the "
             "image center (valid given camera-to-coast distance < 200 m).",
    )
    parser.add_argument(
        "--sea_side", choices=["cw", "ccw"], default="cw",
        help="Which side of the coastline bearing line (going clockwise "
             "from the bearing) is the sea. Flip if land/sea come out "
             "swapped -- check the preview from land_sea_mask.py first.",
    )
    parser.add_argument(
        "--boundary", action="append", default=None, metavar="BEARING:LABEL",
        help="Boundary line as 'bearing:label' pair, e.g. --boundary 220:land --boundary 90:sea. "
             "Angles are 0=North, increasing clockwise. Can be used multiple times to define "
             "multiple angular sectors. Overrides --coastline_bearing_deg if specified.",
    )

    parser.add_argument("--timestamp_regex", type=str, default=None, help=r"Regex for timestamp in filename, default (\d{8})_(\d{6})")
    parser.add_argument("--utc_offset_hours", type=float, default=0.0, help="UTC-Offset of the timestamp from filename (local time)")

    args = parser.parse_args()

    ts_regex = re.compile(args.timestamp_regex) if args.timestamp_regex else DEFAULT_TS_REGEX

    # Parse boundaries if provided
    boundaries = None
    if args.boundary:
        boundaries = []
        for entry in args.boundary:
            parts = entry.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid boundary format '{entry}': expected 'bearing:label'")
            try:
                bearing_deg = float(parts[0])
                label = parts[1]
                boundaries.append((bearing_deg, label))
            except ValueError as e:
                raise ValueError(f"Invalid boundary '{entry}': bearing must be a number. {e}")

    print("Building classification table from color files …")
    class_table = build_classification_table(args.himmel_file, args.wolken_file)

    geo = CameraGeometry(
        lat=args.lat, lon=args.lon, alt=args.alt,
        fov_deg=args.fov_deg,
        center_x=args.center_x, center_y=args.center_y,
        radius_px=args.circle_radius_px,
        apply_height_correction=not args.no_height_correction,
    )
    if args.tilt_matrix_file is not None:
        geo.load_tilt_matrix(args.tilt_matrix_file)
        print(f"Loaded camera-tilt matrix from {args.tilt_matrix_file}")

    print(f"Processing images from {args.input_dir} …")
    csv_path = process_folder(
        args.input_dir, args.output_dir, class_table, geo,
        args.edge_margin_px, args.sun_radius_px,
        ts_regex, args.utc_offset_hours,
        custom_mask_path=args.custom_mask_file,
        coastline_bearing_deg=args.coastline_bearing_deg,
        sea_side=args.sea_side,
        boundaries=boundaries,
    )
    print(f"Finished. Results: {csv_path}")


if __name__ == "__main__":
    main()
