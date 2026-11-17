"""
SmartVine AI — Reference Card & Illumination Calibration Engine
Translates Seng et al. (IEEE Access, 2018) color normalization & scale estimation.
Compensates for outdoor lighting variation (midday sun, cloud cover, canopy shade)
and detects reference markers to compute physical metric scale (pixels-per-mm).
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


def apply_gray_world(bgr_image: np.ndarray) -> np.ndarray:
    """
    Applies classic Gray-World white balancing assuming mean scene reflectance is achromatic.
    """
    b, g, r = cv2.split(bgr_image.astype(np.float32))
    b_avg, g_avg, r_avg = np.mean(b), np.mean(g), np.mean(r)
    gray_val = (b_avg + g_avg + r_avg) / 3.0

    b = np.clip(b * (gray_val / (b_avg + 1e-6)), 0, 255)
    g = np.clip(g * (gray_val / (g_avg + 1e-6)), 0, 255)
    r = np.clip(r * (gray_val / (r_avg + 1e-6)), 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


def calibrate_image_and_scale(
    bgr_image: np.ndarray,
    known_circle_mm: float = 25.0,
    apply_clahe_flag: bool = True
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Normalizes outdoor vineyard illumination using Luminance-preserving CIELab
    white balance and CLAHE on the L* channel, and detects circular reference
    markers (e.g. 25mm red calibration circle) to compute pixels_per_mm scale.

    Parameters:
        bgr_image: Input BGR image (numpy ndarray).
        known_circle_mm: Physical diameter of reference circle in mm (default 25.0 mm).
        apply_clahe_flag: Whether to apply CLAHE to mitigate canopy shadow artifacts.

    Returns:
        calibrated_bgr: Luminance-normalized BGR image.
        pixels_per_mm: Computed scale ratio (pixels per mm).
        info: Diagnostic dictionary containing marker bounding box, center, and circularity.
    """
    if bgr_image is None or bgr_image.size == 0:
        raise ValueError("Invalid image supplied to calibrate_image_and_scale")

    # 1. Luminance-preserving White Balance in CIELab space
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab).astype(np.float32)
    avg_a = np.mean(lab[:, :, 1])
    avg_b = np.mean(lab[:, :, 2])

    # Scale chromatic shift proportionally to luminance to protect deep shadows
    lum_factor = lab[:, :, 0] / 255.0
    lab[:, :, 1] = lab[:, :, 1] - ((avg_a - 128.0) * lum_factor * 1.1)
    lab[:, :, 2] = lab[:, :, 2] - ((avg_b - 128.0) * lum_factor * 1.1)
    lab = np.clip(lab, 0, 255).astype(np.uint8)

    # Apply CLAHE strictly on L-channel to enhance contrast under dense canopy shadows
    if apply_clahe_flag:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    calibrated_bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

    # 2. Detect circular size reference marker on calibration card/strip
    hsv = cv2.cvtColor(calibrated_bgr, cv2.COLOR_BGR2HSV)

    # Red hue wraps around 0/180 in HSV
    mask_red1 = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([12, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([168, 100, 70]), np.array([180, 255, 255]))
    red_mask = cv2.morphologyEx(
        mask_red1 | mask_red2,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pixels_per_mm = 2.5  # Standard fallback scale (representative 640x640 close-up)
    marker_found = False
    marker_info: Dict[str, Any] = {
        "marker_detected": False,
        "marker_type": "fallback_scale",
        "center_xy": None,
        "diameter_px": round(pixels_per_mm * known_circle_mm, 1),
        "circularity": 0.0,
        "pixels_per_mm": pixels_per_mm
    }

    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0 and area > 180:
            circularity = 4.0 * np.pi * (area / (perimeter ** 2))
            if circularity > 0.72:
                (cx, cy), radius_px = cv2.minEnclosingCircle(cnt)
                diameter_px = radius_px * 2.0
                if diameter_px > 10.0:
                    pixels_per_mm = float(diameter_px / known_circle_mm)
                    marker_found = True
                    marker_info = {
                        "marker_detected": True,
                        "marker_type": "circular_red_reference",
                        "center_xy": (int(round(cx)), int(round(cy))),
                        "diameter_px": round(float(diameter_px), 2),
                        "circularity": round(float(circularity), 3),
                        "pixels_per_mm": round(pixels_per_mm, 4)
                    }
                    break

    return calibrated_bgr, pixels_per_mm, marker_info
