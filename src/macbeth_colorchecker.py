"""
SmartVine AI — 24-Patch Macbeth ColorChecker Matrix Calibration
Translates Section III & IV of Seng et al. (IEEE Access, 2018) for GrapeCS-ML Datasets 2, 3, and 4.
Computes a 3x3 least-squares Color Correction Matrix (CCM) mapping raw in-field RGB patches
to canonical X-Rite / BabelColor Macbeth sRGB reference values to eliminate daylight spectral drift.
"""

from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import cv2

# Canonical 24 Macbeth ColorChecker sRGB reference values (D65 illumination / BabelColor standard)
# Organized row-by-row (4 rows x 6 columns)
MACBETH_SRGB_REFERENCE = np.array([
    # Row 1: Natural colors
    [115, 82, 68],    # 1. Dark skin
    [194, 150, 130],  # 2. Light skin
    [98, 122, 157],   # 3. Blue sky
    [87, 108, 67],    # 4. Foliage
    [133, 128, 177],  # 5. Blue flower
    [103, 189, 170],  # 6. Bluish green
    # Row 2: Miscellaneous colors
    [214, 126, 44],   # 7. Orange
    [80, 91, 166],    # 8. Purplish blue
    [193, 90, 99],    # 9. Moderate red
    [94, 60, 108],    # 10. Purple
    [157, 188, 64],   # 11. Yellow green
    [224, 163, 46],   # 12. Orange yellow
    # Row 3: Primary / Secondary colors
    [56, 61, 150],    # 13. Blue
    [70, 148, 73],    # 14. Green
    [175, 54, 60],    # 15. Red
    [231, 199, 31],   # 16. Yellow
    [187, 86, 149],   # 17. Magenta
    [8, 133, 161],    # 18. Cyan
    # Row 4: Grayscale gradient
    [243, 243, 242],  # 19. White 9.5 (.05 D)
    [200, 200, 200],  # 20. Neutral 8 (.23 D)
    [160, 160, 160],  # 21. Neutral 6.5 (.44 D)
    [122, 122, 121],  # 22. Neutral 5 (.70 D)
    [85, 85, 85],     # 23. Neutral 3.5 (1.05 D)
    [52, 52, 52]      # 24. Black 2 (1.5 D)
], dtype=np.float32)


def compute_ccm(source_rgb: np.ndarray, target_rgb: np.ndarray = MACBETH_SRGB_REFERENCE) -> np.ndarray:
    """
    Computes a 3x3 least-squares Color Correction Matrix (CCM).
    Solves min || source_rgb @ CCM - target_rgb ||_2
    Returns a 3x3 transformation matrix.
    """
    if source_rgb.shape[0] < 3:
        raise ValueError("At least 3 color patches required to solve for CCM.")
    # Least squares solution: CCM = (X^T X)^-1 X^T Y
    ccm, residuals, rank, s = np.linalg.lstsq(source_rgb, target_rgb, rcond=None)
    return ccm


def apply_color_correction_matrix(rgb_image: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    """
    Applies the 3x3 CCM to an RGB image: I_corrected = np.clip(I @ CCM, 0, 255).
    """
    h, w, c = rgb_image.shape
    reshaped = rgb_image.reshape(-1, 3).astype(np.float32)
    corrected = np.dot(reshaped, ccm)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    return corrected.reshape(h, w, c)


def detect_macbeth_chart_bounding_box(bgr_image: np.ndarray) -> Optional[np.ndarray]:
    """
    Locates potential rectangular Macbeth ColorChecker chart in field imagery
    using edge detection, contour approximation, aspect ratio checks (chart ratio ~ 1.5),
    and verified multi-patch color variance to avoid confusing it with a white calibration card.
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    max_area = 0

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 10000:
                rect = cv2.minAreaRect(approx)
                (cx, cy), (width, height), angle = rect
                if min(width, height) > 0:
                    aspect = max(width, height) / min(width, height)
                    # Macbeth chart standard aspect ratio is approx 6:4 = 1.50
                    if 1.30 <= aspect <= 1.70:
                        # Ensure region has significant chromatic variance (24 distinct color patches)
                        x, y, w_box, h_box = cv2.boundingRect(approx)
                        roi = bgr_image[y:y + h_box, x:x + w_box]
                        if roi.size > 0:
                            std_per_channel = np.std(roi, axis=(0, 1))
                            # Real Macbeth chart has high color variance across patches (std > 35)
                            if np.mean(std_per_channel) > 35.0:
                                if area > max_area:
                                    max_area = area
                                    best_rect = approx.reshape(4, 2)

    return best_rect


def calibrate_with_macbeth(
    bgr_image: np.ndarray,
    chart_polygon: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Calibrates image color fidelity using detected or provided Macbeth chart.
    If no chart is cleanly segmented, falls back to identity matrix.
    """
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    if chart_polygon is None:
        chart_polygon = detect_macbeth_chart_bounding_box(bgr_image)

    if chart_polygon is not None and len(chart_polygon) == 4:
        # Perspective transform to canonical 6x4 patch grid
        target_pts = np.array([
            [0, 0],
            [300, 0],
            [300, 200],
            [0, 200]
        ], dtype=np.float32)

        # Order polygon points (tl, tr, br, bl)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = chart_polygon.sum(axis=1)
        rect[0] = chart_polygon[np.argmin(s)]
        rect[2] = chart_polygon[np.argmax(s)]
        diff = np.diff(chart_polygon, axis=1)
        rect[1] = chart_polygon[np.argmin(diff)]
        rect[3] = chart_polygon[np.argmax(diff)]

        M = cv2.getPerspectiveTransform(rect, target_pts)
        warped = cv2.warpPerspective(rgb_image, M, (300, 200))

        # Sample the 24 patches (4 rows x 6 cols) at center of each block
        sampled_patches = []
        dx = 300 / 6.0
        dy = 200 / 4.0
        for r in range(4):
            for c in range(6):
                cx = int(c * dx + dx / 2.0)
                cy = int(r * dy + dy / 2.0)
                patch = warped[max(0, cy - 8):cy + 8, max(0, cx - 8):cx + 8]
                sampled_patches.append(np.mean(patch, axis=(0, 1)))

        source_patches = np.array(sampled_patches, dtype=np.float32)
        ccm = compute_ccm(source_patches, MACBETH_SRGB_REFERENCE)
        calibrated_rgb = apply_color_correction_matrix(rgb_image, ccm)
        calibrated_bgr = cv2.cvtColor(calibrated_rgb, cv2.COLOR_RGB2BGR)

        return calibrated_bgr, ccm, {
            "macbeth_detected": True,
            "ccm_matrix": ccm.tolist(),
            "mean_rgb_drift_pre": float(np.mean(np.abs(source_patches - MACBETH_SRGB_REFERENCE))),
            "mean_rgb_drift_post": float(np.mean(np.abs(np.dot(source_patches, ccm) - MACBETH_SRGB_REFERENCE)))
        }

    # Fallback: Identity CCM
    identity_ccm = np.eye(3, dtype=np.float32)
    return bgr_image.copy(), identity_ccm, {
        "macbeth_detected": False,
        "ccm_matrix": identity_ccm.tolist(),
        "note": "No Macbeth chart detected; identity transformation applied."
    }
