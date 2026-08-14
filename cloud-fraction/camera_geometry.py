#!/usr/bin/env python3
"""
camera_geometry.py
===================

Shared fisheye-camera geometry for the Wolkenkamera tool family
(`cloud_fraction.py`, `camera_tilt.py`, and any future image-analysis tool).

Implements the coordinate transforms described in the technical
documentation "Verwendung der Kamera VIVOTEK FE8174V als Wolkenkamera"
(I. Lange, Universität Hamburg, Meteorologisches Institut):

  - Sec. 4.1            Höhenwinkelkorrektur (Polynome P / Q, Eq. 2 + 3)
  - Sec. 4.2 / 4.3       Bild <-> Hemisphärenkoordinaten (Azimut, Höhenwinkel)
  - Sec. 5.1             Hemisphärenkoordinaten <-> kartesische Koordinaten
                         (Eq. 5-9) and the camera-tilt rotation matrix M
                         (Smess, Swahr, Eq. 4)

The central entry point for other tools is `pixel_to_sky_angles`: it turns
an (x, y) image position into the *actual* (azimuth, elevation) position on
the sky, including both the lens's height-angle calibration (Sec. 4.1) and,
if known, the day's camera-tilt correction (Sec. 5, matrix M). The inverse,
`sky_angles_to_pixel`, is used e.g. to find where a known sky position
(such as the sun) appears in the image.

If no tilt matrix is supplied, M defaults to the identity matrix, i.e. a
perfectly levelled and northed camera -- behaviour is then identical to
just using the Sec. 4 transforms on their own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Höhenwinkelkorrektur (PDF Sec. 4.1, Eq. 2 + 3)
# ----------------------------------------------------------------------
# The camera optics deviate from the ideal equidistant projection. The PDF
# gives two interpolation polynomials, fitted to a measurement of the real
# camera (Kamera Nr. 2):
#   P: eps_ideal -> eps_wahr   (Eq. 2, used when going image -> sky, Sec. 4.2)
#   Q: eps_wahr  -> eps_ideal  (Eq. 3, used when going sky -> image, Sec. 4.3)
# Both take/return degrees. Inputs are clipped to the calibrated domain
# (roughly -1.2..90 / 0.9..90) because the polynomials are only valid there.
def height_angle_true_from_ideal(eps_ideal_deg):
    """Eq. (2): eps_wahr = P(eps_ideal)."""
    e = np.clip(eps_ideal_deg, -1.19, 90.0)
    return (
        -6.380024219e-7 * e**4
        + 1.384399783e-4 * e**3
        - 1.122405179e-2 * e**2
        + 1.326190211 * e
        + 2.494295303
    )


def height_angle_ideal_from_true(eps_wahr_deg):
    """Eq. (3): eps_ideal = Q(eps_wahr)."""
    e = np.clip(eps_wahr_deg, 0.9, 90.0)
    return (
        4.407488186e-7 * e**4
        - 9.844546547e-5 * e**3
        + 8.969454015e-3 * e**2
        + 6.890441144e-1 * e
        - 1.817333483
    )


# ----------------------------------------------------------------------
# Hemisphärenkoordinaten <-> kartesische Koordinaten (PDF Sec. 5.1, Eq. 5-9)
# ----------------------------------------------------------------------
def az_el_to_cartesian(az_deg, el_deg):
    """Eq. (5)-(7): unit-sphere cartesian coords from azimuth/elevation.
    x -> East, y -> North, z -> up. Works for scalars or numpy arrays."""
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    x = np.cos(el) * np.sin(az)
    y = np.cos(el) * np.cos(az)
    z = np.sin(el)
    return x, y, z


def cartesian_to_az_el(vx, vy, vz):
    """Eq. (8)-(9): azimuth/elevation from cartesian coords (need not be
    unit length). Returns azimuth in [0, 360), elevation in [-90, 90].
    The degenerate case x=y=0 (PDF special case) maps to azimuth=0."""
    r = np.sqrt(vx**2 + vy**2 + vz**2)
    r_safe = np.where(r < 1e-12, 1.0, r)
    pole = (np.abs(vx) < 1e-9) & (np.abs(vy) < 1e-9)
    az = np.where(pole, 0.0, np.degrees(np.arctan2(vx, vy)) % 360.0)
    el = np.degrees(np.arcsin(np.clip(vz / r_safe, -1.0, 1.0)))
    return az, el


# ----------------------------------------------------------------------
# Bild <-> ideale Hemisphärenkoordinaten (PDF Sec. 4.2 / 4.3)
# ----------------------------------------------------------------------
def pixel_to_ideal_hemisphere(x, y, cx, cy, a):
    """PDF Sec. 4.2: (x, y) image coords -> (azimuth_ideal, eps_ideal),
    *before* the height-angle correction. Works for scalars or arrays."""
    x0 = np.asarray(x, dtype=np.float64) - cx
    y0 = np.asarray(y, dtype=np.float64) - cy
    r = np.hypot(x0, y0)
    az_ideal = np.degrees(np.arctan2(x0, y0)) % 360.0
    eps_ideal = 90.0 * (1.0 - r / a)
    return az_ideal, eps_ideal


def ideal_hemisphere_to_pixel(az_ideal_deg, eps_ideal_deg, cx, cy, a, fov_deg=180.0):
    """PDF Sec. 4.3 (x0 = r sin α, y0 = r cos α): inverse of
    `pixel_to_ideal_hemisphere`. Note the '+' sign on the y-term -- per PDF
    Sec. 2, image-north is at the *bottom* of the image, unlike on normal
    maps."""
    zenith_ideal = np.clip(90.0 - eps_ideal_deg, 0.0, fov_deg / 2)
    r = a * (zenith_ideal / (fov_deg / 2))
    az_rad = np.deg2rad(az_ideal_deg)
    x = cx + r * np.sin(az_rad)
    y = cy + r * np.cos(az_rad)
    return x, y


# ----------------------------------------------------------------------
# Camera geometry, including tilt matrix M (PDF Sec. 5)
# ----------------------------------------------------------------------
@dataclass
class CameraGeometry:
    lat: float
    lon: float
    alt: float = 0.0
    fov_deg: float = 180.0           # full field of view of the fisheye lens
    center_x: float | None = None    # None => image center
    center_y: float | None = None
    radius_px: float | None = None   # None => min(width,height)/2
    apply_height_correction: bool = True   # apply Sec. 4.1 polynomials P/Q
    tilt_matrix: list | None = None  # 3x3 rotation matrix M (cam->true sky), Sec. 5.1

    def get_M(self) -> np.ndarray:
        """Returns the camera->true-sky rotation matrix M, or the identity
        matrix (perfectly levelled & northed camera) if none was set."""
        if self.tilt_matrix is None:
            return np.eye(3)
        return np.asarray(self.tilt_matrix, dtype=np.float64)

    def load_tilt_matrix(self, path: Path) -> None:
        """Loads M from a JSON file written by `camera_tilt.py`."""
        payload = json.loads(Path(path).read_text())
        self.tilt_matrix = payload["tilt_matrix"]


def resolve_center_radius(geo: CameraGeometry, width: int, height: int):
    cx = geo.center_x if geo.center_x is not None else width / 2
    cy = geo.center_y if geo.center_y is not None else height / 2
    r_max = geo.radius_px if geo.radius_px is not None else min(width, height) / 2
    return cx, cy, r_max


# ----------------------------------------------------------------------
# Full chain: image position <-> real position in the sky
# ----------------------------------------------------------------------
def pixel_to_sky_angles(x, y, cx, cy, a, geo: CameraGeometry):
    """Image position -> *actual* (azimuth, elevation) on the sky.

    Chain: pixel -> ideal camera hemisphere coords (Sec. 4.2)
                  -> true camera hemisphere coords (Eq. 2 / P)
                  -> cartesian camera-frame vector (Eq. 5-7)
                  -> rotate with the day's tilt matrix M (Sec. 5.1)
                  -> cartesian true-sky vector
                  -> true azimuth/elevation (Eq. 8-9).

    This is the function meant to be reused by future image-analysis
    tools whenever a pixel position needs to be turned into the actual
    direction on the sky it corresponds to. Works for scalars or
    same-shaped numpy arrays (e.g. a full pixel grid).
    """
    az_ideal, eps_ideal = pixel_to_ideal_hemisphere(x, y, cx, cy, a)
    if geo.apply_height_correction:
        eps_cam = height_angle_true_from_ideal(eps_ideal)
    else:
        eps_cam = eps_ideal

    vx, vy, vz = az_el_to_cartesian(az_ideal, eps_cam)
    M = geo.get_M()
    tx = M[0, 0] * vx + M[0, 1] * vy + M[0, 2] * vz
    ty = M[1, 0] * vx + M[1, 1] * vy + M[1, 2] * vz
    tz = M[2, 0] * vx + M[2, 1] * vy + M[2, 2] * vz

    return cartesian_to_az_el(tx, ty, tz)


def sky_angles_to_pixel(az_true_deg, eps_true_deg, cx, cy, a, geo: CameraGeometry):
    """Inverse of `pixel_to_sky_angles`: a real sky direction -> image
    position. Used e.g. to find where the (theoretical) sun should appear
    in the image so it can be masked out."""
    vx, vy, vz = az_el_to_cartesian(az_true_deg, eps_true_deg)
    M = geo.get_M()
    v_true = np.array([vx, vy, vz], dtype=np.float64)
    # v_cam = M^-1 @ v_true  (Sec. 5.1: Swahr = M Smess  =>  Smess = M^-1 Swahr)
    v_cam = np.linalg.solve(M, v_true)
    az_cam, eps_cam = cartesian_to_az_el(v_cam[0], v_cam[1], v_cam[2])

    if geo.apply_height_correction:
        eps_ideal = height_angle_ideal_from_true(eps_cam)
    else:
        eps_ideal = eps_cam

    return ideal_hemisphere_to_pixel(az_cam, eps_ideal, cx, cy, a, geo.fov_deg)


# ----------------------------------------------------------------------
# Persistence of the daily tilt matrix
# ----------------------------------------------------------------------
def save_tilt_matrix(path: Path, M: np.ndarray, meta: dict | None = None) -> None:
    payload = {"tilt_matrix": np.asarray(M).tolist()}
    if meta:
        payload.update(meta)
    Path(path).write_text(json.dumps(payload, indent=2, default=str))


def load_tilt_matrix(path: Path) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    return np.asarray(payload["tilt_matrix"], dtype=np.float64)
