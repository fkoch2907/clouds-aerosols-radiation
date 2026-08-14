#!/usr/bin/env python3
"""
land_sea_mask.py
=================

Builds a static land/sea mask for a Wolkenkamera image, using a single
straight coastline bearing line through the image center (cx, cy).

Since both cameras are < 200 m from the coast, the true arctan(h/r)
sky-projection of the coastline is well approximated by a straight bearing
line through the center -- the resulting few-degree error near the center
is negligible given clouds don't have a sharp classification boundary
anyway (see prior discussion).

This reuses `pixel_to_ideal_hemisphere` from camera_geometry.py directly,
rather than reimplementing azimuth from scratch. That matters for two
reasons:
  1. It guarantees the same north-at-bottom convention already handled
     internally by that function (PDF Sec. 2) -- so the coastline bearing
     you pass in is just the real, map-read compass bearing (0=N,
     clockwise). No manual +180 flip needed.
  2. It uses the same (cx, cy) reference point as the rest of the
     pipeline (via `resolve_center_radius`), so this mask can't drift out
     of sync with the sun mask / solid-angle weight map's notion of
     "image center".

Deliberately NOT used here: height-angle correction (P/Q) and the tilt
matrix M. Both only affect elevation, not azimuth, and are irrelevant to
a pure bearing split -- and the whole point of this approach is to skip
tilt correction, since its effect on the mask is judged negligible at
this camera-to-coast distance.

Since each camera is fixed, generate this mask once per camera (unless
remounted) and cache it; there is no need to recompute it per frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from camera_geometry import CameraGeometry, pixel_to_ideal_hemisphere, resolve_center_radius


def build_sector_mask(
    geo: CameraGeometry,
    width: int,
    height: int,
    boundaries: list[tuple[float, str]],
) -> dict[str, np.ndarray]:
    """
    General version: splits the image into an arbitrary number of angular
    sectors around the center, each labelled by the caller -- not limited
    to a single land/sea half-plane split.

    boundaries: list of (bearing_deg, label) pairs. Each entry means "going
    clockwise starting at bearing_deg, the sector belongs to `label`, until
    the next boundary's bearing". Order in the list doesn't matter -- they
    are sorted internally by bearing.

    Example -- a peninsula seen between bearings 220 deg and 320 deg
    (land occupies the 100 deg wedge from 220 to 320; sea occupies the
    remaining 260 deg wedge wrapping the other way around):

        boundaries = [(220, "land"), (320, "sea")]

    This reduces to the old single-line half-plane split when you give
    exactly two boundaries 180 deg apart, e.g. [(90, "sea"), (270, "land")].

    Returns a dict {label: boolean mask}, one entry per distinct label
    used in `boundaries` (masks are disjoint and cover the whole image).

    Caveat: this only works if the coastline is "star-shaped" as seen from
    the camera -- i.e. any single bearing crosses the coastline at most
    once. A concave coastline that the camera sees twice along the same
    bearing (e.g. a bay behind a headland) can't be represented as a pure
    angular sector split; that would need the full range-aware projection
    discussed earlier instead.
    """
    cx, cy, a = resolve_center_radius(geo, width, height)
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    az_ideal, _eps_ideal = pixel_to_ideal_hemisphere(xx, yy, cx, cy, a)

    boundaries = sorted(((b % 360, label) for b, label in boundaries), key=lambda t: t[0])
    bearings = [b for b, _ in boundaries]
    labels = [label for _, label in boundaries]

    masks = {label: np.zeros(az_ideal.shape, dtype=bool) for label in set(labels)}
    n = len(boundaries)
    for i in range(n):
        start = bearings[i]
        end = bearings[(i + 1) % n]
        label = labels[i]
        if start < end:
            sel = (az_ideal >= start) & (az_ideal < end)
        else:  # sector wraps through 0/360
            sel = (az_ideal >= start) | (az_ideal < end)
        masks[label] |= sel
    return masks


def build_land_sea_mask(
    geo: CameraGeometry,
    width: int,
    height: int,
    coastline_bearing_deg: float,
    sea_side: str = "cw",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple case: a single straight coastline through the center, splitting
    the image into two halves. Returns (land_mask, sea_mask).

    coastline_bearing_deg: real compass bearing (0=N, clockwise) of the
        coastline, exactly as read off a map -- no manual flip needed.
    sea_side: "cw" or "ccw" -- which side of the bearing line, going
        clockwise from the bearing, is the sea. Flip if land/sea come out
        swapped in the preview.

    For peninsulas, bays, or any coastline that isn't a single straight
    line through the camera, use `build_sector_mask` instead.
    """
    boundaries = (
        [(coastline_bearing_deg, "sea"), (coastline_bearing_deg + 180, "land")]
        if sea_side == "cw"
        else [(coastline_bearing_deg, "land"), (coastline_bearing_deg + 180, "sea")]
    )
    masks = build_sector_mask(geo, width, height, boundaries)
    return masks["land"], masks["sea"]


def preview_mask(img: np.ndarray, land_mask: np.ndarray, sea_mask: np.ndarray, title: str = "") -> None:
    """Overlay land/sea split on a sample image for a visual sanity check."""
    overlay = img.copy()
    if overlay.ndim == 2:
        overlay = np.stack([overlay] * 3, axis=-1)

    alpha = 0.35
    blue = np.array([50, 90, 220])
    green = np.array([70, 150, 70])
    overlay[sea_mask] = (1 - alpha) * overlay[sea_mask] + alpha * blue
    overlay[land_mask] = (1 - alpha) * overlay[land_mask] + alpha * green
    overlay = overlay.astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[0].axis("off")
    ax[1].imshow(overlay)
    ax[1].set_title(f"Land (green) / Sea (blue) -- {title}")
    ax[1].axis("off")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Build a land/sea mask for one Wolkenkamera.")
    parser.add_argument("--name", type=str, required=True, help="Camera name, used for output filenames")
    parser.add_argument("--sample_image", type=Path, required=True, help="A representative image (also used for width/height)")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--alt", type=float, default=0.0)
    parser.add_argument("--fov_deg", type=float, default=180.0)
    parser.add_argument("--center_x", type=float, default=None)
    parser.add_argument("--center_y", type=float, default=None)
    parser.add_argument("--circle_radius_px", type=float, default=None)
    parser.add_argument(
        "--coastline_bearing_deg", type=float, default=None,
        help="Simple case: single coastline bearing (0=N, clockwise), "
             "splitting the image into two halves. Ignored if --boundary "
             "is given.",
    )
    parser.add_argument("--sea_side", choices=["cw", "ccw"], default="cw")
    parser.add_argument(
        "--boundary", action="append", default=None, metavar="BEARING:LABEL",
        help="General case (e.g. a peninsula seen as a wedge): repeatable "
             "'bearing:label' pair, e.g. --boundary 220:land --boundary "
             "320:sea. Any number of boundaries/labels is allowed. "
             "Overrides --coastline_bearing_deg/--sea_side if given.",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("masks"))
    parser.add_argument("--no_preview", action="store_true")
    args = parser.parse_args()

    img = np.array(Image.open(args.sample_image))
    height, width = img.shape[:2]

    geo = CameraGeometry(
        lat=args.lat, lon=args.lon, alt=args.alt,
        fov_deg=args.fov_deg,
        center_x=args.center_x, center_y=args.center_y,
        radius_px=args.circle_radius_px,
    )

    if args.boundary:
        boundaries = []
        for entry in args.boundary:
            bearing_str, label = entry.split(":")
            boundaries.append((float(bearing_str), label))
        masks = build_sector_mask(geo, width, height, boundaries)
        if set(masks.keys()) != {"land", "sea"}:
            raise ValueError(
                f"Expected exactly the labels 'land' and 'sea' in --boundary, got {sorted(masks.keys())}. "
                "cloud_fraction.py's regional split assumes these two labels."
            )
        land_mask, sea_mask = masks["land"], masks["sea"]
    else:
        if args.coastline_bearing_deg is None:
            raise ValueError("Provide either --coastline_bearing_deg or one or more --boundary entries.")
        land_mask, sea_mask = build_land_sea_mask(
            geo, width, height, args.coastline_bearing_deg, args.sea_side
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / f"{args.name}_land_mask.npy", land_mask)
    np.save(args.out_dir / f"{args.name}_sea_mask.npy", sea_mask)
    print(f"[{args.name}] saved land_mask / sea_mask to {args.out_dir}")
    print(f"[{args.name}] land pixels: {land_mask.sum()}, sea pixels: {sea_mask.sum()}")

    if not args.no_preview:
        preview_mask(img, land_mask, sea_mask, title=args.name)


if __name__ == "__main__":
    main()
