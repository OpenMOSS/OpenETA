"""Shared host-side depth compatibility helpers."""

from __future__ import annotations

import math
from typing import Any

from PIL import Image


def target_depth_cutoff_factor(
    *,
    depth_path: str,
    mask_path: str,
    intrinsics: dict[str, Any],
) -> float:
    """Keep targets below a fixed 1m service cutoff without changing raw depth.

    Some deployed grasp services use a fixed one-metre truncation. The caller can
    preserve metric geometry by scaling request intrinsics and restoring returned
    lengths. The 99th percentile prevents isolated invalid depth from inflating the
    compatibility factor.
    """

    try:
        scale = float(intrinsics.get("scale"))
        if not math.isfinite(scale) or scale <= 0:
            return 1.0
        with Image.open(depth_path) as depth_image, Image.open(mask_path) as mask_image:
            if depth_image.size != mask_image.size:
                return 1.0
            mask_gray = mask_image.convert("L")
            depth_pixels = depth_image.load()
            mask_pixels = mask_gray.load()
            width, height = depth_image.size
            depths = [
                float(depth_pixels[x, y]) / scale
                for y in range(height)
                for x in range(width)
                if int(mask_pixels[x, y]) > 0 and float(depth_pixels[x, y]) > 0
            ]
    except (OSError, TypeError, ValueError):
        return 1.0
    if not depths:
        return 1.0
    depths.sort()
    p99 = depths[min(len(depths) - 1, math.floor(0.99 * len(depths)))]
    if p99 <= 0.9:
        return 1.0
    return round(min(4.0, max(1.0, p99 / 0.9)), 6)
