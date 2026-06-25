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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import convolve

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
@dataclass
class CameraGeometry:
    lat: float
    lon: float
    alt: float
    fov_deg: float           # full field of view of the fisheye lens (typ. 180°)
    center_x: float | None   # None => image center
    center_y: float | None
    radius_px: float | None  # None => min(width,height)/2 - edge_margin


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


def zenith_to_radius(zenith_deg: float, fov_deg: float, r_max: float) -> float:
    """Equidistante Fisheye-Projektion: r ~ proportional zum Zenitwinkel."""
    zenith_deg = max(0.0, min(zenith_deg, fov_deg / 2))
    return r_max * (zenith_deg / (fov_deg / 2))


def sun_pixel_position(
    azimuth_deg: float,
    elevation_deg: float,
    width: int,
    height: int,
    geo: CameraGeometry,
) -> tuple[int, int] | None:
    """Rechnet Sonnen-Azimuth/Elevation in Bildpixel um. Gibt None zurück,
    wenn die Sonne unter dem Horizont steht (Nachtbild)."""
    if elevation_deg <= 0:
        return None

    cx = geo.center_x if geo.center_x is not None else width / 2
    cy = geo.center_y if geo.center_y is not None else height / 2
    r_max = geo.radius_px if geo.radius_px is not None else min(width, height) / 2

    zenith_deg = 90.0 - elevation_deg
    r = zenith_to_radius(zenith_deg, geo.fov_deg, r_max)

    # Azimuth: 0°=Nord, Uhrzeigersinn über Ost. Bildkonvention: oben=Nord.
    az_rad = np.deg2rad(azimuth_deg)
    px = cx + r * np.sin(az_rad)
    py = cy - r * np.cos(az_rad)
    return int(round(px)), int(round(py))


def circular_mask(height: int, width: int, cx: float, cy: float, radius: float) -> np.ndarray:
    """Boolesche Maske, True = innerhalb des Kreises."""
    yy, xx = np.mgrid[0:height, 0:width]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


# ----------------------------------------------------------------------
# 3) Zeitstempel aus Dateinamen
# ----------------------------------------------------------------------
DEFAULT_TS_REGEX = re.compile(r"(\d{8})_(\d{6})")


def parse_timestamp(filename: str, regex: re.Pattern, utc_offset_hours: float) -> datetime:
    m = regex.search(filename)
    if not m:
        raise ValueError(f"Kein Zeitstempel im Dateinamen gefunden: {filename}")
    date_str, time_str = m.group(1), m.group(2)
    local_dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
    return local_dt.replace(tzinfo=timezone(timedelta(hours=utc_offset_hours))).astimezone(timezone.utc)


# ----------------------------------------------------------------------
# 4) Ein Bild verarbeiten
# ----------------------------------------------------------------------
def process_image(
    img_path: Path,
    out_path: Path,
    class_table: np.ndarray,
    geo: CameraGeometry,
    edge_margin_px: float,
    sun_radius_px: float,
    dt_utc: datetime,
) -> float:
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)  # (H, W, 3) uint8
    height, width, _ = arr.shape

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    classes = class_table[r, g, b]  # (H, W) Klassen-Code je Pixel

    # --- äußeren Rand maskieren ---
    cx = geo.center_x if geo.center_x is not None else width / 2
    cy = geo.center_y if geo.center_y is not None else height / 2
    r_max = geo.radius_px if geo.radius_px is not None else min(width, height) / 2
    fov_circle = circular_mask(height, width, cx, cy, r_max - edge_margin_px)
    classes[~fov_circle] = HO_MASK

    # --- Sonnenkreis maskieren ---
    azimuth_deg, elevation_deg = sun_position(dt_utc, geo)
    sun_xy = sun_pixel_position(azimuth_deg, elevation_deg, width, height, geo)
    if sun_xy is not None:
        sx, sy = sun_xy
        sun_circle = circular_mask(height, width, sx, sy, sun_radius_px)
        classes[sun_circle] = HO_SUN

    # --- Cloud Fraction berechnen (Maske/Sonne/Unbekannt ausgeschlossen) ---
    cloud_px = np.count_nonzero(classes == HO_CLOUD)
    sky_px = np.count_nonzero(classes == HO_SKY)
    both_px = np.count_nonzero(classes == HO_BOTH)
    valid_px = cloud_px + sky_px + both_px
    if valid_px == 0:
        cloud_fraction = np.nan
    else:
        # "Beides"-Pixel werden mit Gewicht 0.5 als Wolke gezählt
        cloud_fraction = (cloud_px + 0.5 * both_px) / valid_px

    # --- Ausgabebild einfärben und speichern ---
    out_arr = np.zeros_like(arr)
    for code, color in DISPLAY_COLORS.items():
        out_arr[classes == code] = color
    Image.fromarray(out_arr, mode="RGB").save(out_path)

    return cloud_fraction


# ----------------------------------------------------------------------
# 5) Ordner-Workflow + Plot
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
) -> Path:
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg")
    )
    if not image_paths:
        raise FileNotFoundError(f"Keine JPG-Bilder in {input_dir} gefunden.")

    results = []
    for img_path in image_paths:
        try:
            dt_utc = parse_timestamp(img_path.name, ts_regex, utc_offset_hours)
        except ValueError as e:
            print(f"  Übersprungen: {e}", file=sys.stderr)
            continue

        out_path = output_dir / f"{img_path.stem}_classified.jpg"
        cf = process_image(
            img_path, out_path, class_table, geo,
            edge_margin_px, sun_radius_px, dt_utc,
        )
        results.append((dt_utc, cf))
        print(f"  {img_path.name}: cloud fraction = {cf:.3f}")

    csv_path = output_dir / "cloud_fraction_timeseries.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "cloud_fraction"])
        for dt_utc, cf in results:
            writer.writerow([dt_utc.isoformat(), f"{cf:.4f}"])

    plot_timeseries(results, output_dir / "cloud_fraction_timeseries.png", input_dir.name)
    return csv_path


def plot_timeseries(results: list[tuple[datetime, float]], out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if not results:
        return
    times = [r[0] for r in results]
    cfs = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, cfs, marker="o", linestyle="-", markersize=3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Cloud fraction")
    ax.set_xlabel("Zeit (UTC)")
    ax.set_title(f"Cloud fraction Zeitreihe – {title}")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cloud fraction aus Fisheye-Wolkenkamerabildern")
    parser.add_argument("--input_dir", type=Path, required=True, help="Ordner mit JPG-Bildern eines Tages")
    parser.add_argument("--output_dir", type=Path, required=True, help="Ziel-Ordner für Output")
    parser.add_argument("--himmel_file", type=Path, required=True, help="Farbdatei 'Himmel' (FE1/FE3)")
    parser.add_argument("--wolken_file", type=Path, required=True, help="Farbdatei 'Wolken' (FE1/FE3)")

    parser.add_argument("--lat", type=float, required=True, help="Breitengrad der Kamera")
    parser.add_argument("--lon", type=float, required=True, help="Längengrad der Kamera")
    parser.add_argument("--alt", type=float, default=0.0, help="Höhe der Kamera über NN (m)")

    parser.add_argument("--fov_deg", type=float, default=180.0, help="Voller Öffnungswinkel der Fisheye-Optik")
    parser.add_argument("--center_x", type=float, default=None, help="Bild-x des Fisheye-Zentrums (default: Bildmitte)")
    parser.add_argument("--center_y", type=float, default=None, help="Bild-y des Fisheye-Zentrums (default: Bildmitte)")
    parser.add_argument("--circle_radius_px", type=float, default=None, help="Radius des Fisheye-Kreises in Pixel (default: min(w,h)/2)")
    parser.add_argument("--edge_margin_px", type=float, default=5.0, help="Zusätzlicher Rand, der vom Fisheye-Kreis abgezogen wird")
    parser.add_argument("--sun_radius_px", type=float, default=25.0, help="Radius des auszuschneidenden Sonnenkreises in Pixel")

    parser.add_argument("--timestamp_regex", type=str, default=None, help=r"Regex für Zeitstempel im Dateinamen, default (\d{8})_(\d{6})")
    parser.add_argument("--utc_offset_hours", type=float, default=0.0, help="UTC-Offset der Dateinamen-Zeitstempel (lokale Zeit)")

    args = parser.parse_args()

    ts_regex = re.compile(args.timestamp_regex) if args.timestamp_regex else DEFAULT_TS_REGEX

    print("Baue Klassifikationstabelle aus Farbdateien …")
    class_table = build_classification_table(args.himmel_file, args.wolken_file)

    geo = CameraGeometry(
        lat=args.lat, lon=args.lon, alt=args.alt,
        fov_deg=args.fov_deg,
        center_x=args.center_x, center_y=args.center_y,
        radius_px=args.circle_radius_px,
    )

    print(f"Verarbeite Bilder aus {args.input_dir} …")
    csv_path = process_folder(
        args.input_dir, args.output_dir, class_table, geo,
        args.edge_margin_px, args.sun_radius_px,
        ts_regex, args.utc_offset_hours,
    )
    print(f"Fertig. Ergebnisse: {csv_path}")


if __name__ == "__main__":
    main()
