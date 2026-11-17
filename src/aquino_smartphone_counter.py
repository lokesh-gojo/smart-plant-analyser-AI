"""
SmartVine AI — Aquino Smartphone CIELab Specular-Maxima Berry Counter
Translates Section II-A & Section 13-D of Seng et al. (IEEE Access, 2018).
Implements:
1. Aquino et al. (2017) regional illumination maxima detection on the CIELab L* channel.
2. Nuske et al. (2011) 5-nearest-neighbor spatial proximity clustering to eliminate leaf glare.
3. Vincent et al. (2016) 4-class day/night illumination regime classifier.
"""

from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors


def classify_illumination_regime(bgr_image: np.ndarray) -> Dict[str, Any]:
    """
    Vincent et al. (2016) 4-class lighting & cultivar classification:
    - Regime 1: Day White Cultivar (high canopy luminance, greenish hue)
    - Regime 2: Day Red Cultivar (high canopy luminance, red/blue hue)
    - Regime 3: Night Flash White Cultivar (dark background, flash specular highlights)
    - Regime 4: Night Flash Red Cultivar (dark background, high anthocyanin absorption)
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    mean_lum = float(np.mean(gray))
    background_darkness = float(np.mean(gray < 40))

    # Evaluate hue distribution of non-dark pixels
    bright_pixels = hsv[gray > 40]
    if len(bright_pixels) > 0:
        mean_hue = float(np.mean(bright_pixels[:, 0]))
    else:
        mean_hue = 60.0

    is_night = (mean_lum < 65.0) or (background_darkness > 0.60)
    is_white_cultivar = (25.0 <= mean_hue <= 85.0)

    if not is_night and is_white_cultivar:
        regime = "Daytime White Cultivar (Chardonnay / Sauvignon Blanc)"
    elif not is_night and not is_white_cultivar:
        regime = "Daytime Red Cultivar (Shiraz / Merlot / Cabernet)"
    elif is_night and is_white_cultivar:
        regime = "Night Flash White Cultivar (High contrast specular mode)"
    else:
        regime = "Night Flash Red Cultivar (Optimized for anthocyanin reflection)"

    return {
        "regime_name": regime,
        "is_night_flash": is_night,
        "is_white_cultivar": is_white_cultivar,
        "mean_luminance": round(mean_lum, 1),
        "dark_pixel_ratio": round(background_darkness, 3)
    }


def detect_specular_maxima_berries(
    bgr_image: np.ndarray,
    binary_mask: Optional[np.ndarray] = None,
    pixels_per_mm: float = 2.5,
    min_prominence: float = 25.0
) -> Dict[str, Any]:
    """
    Aquino et al. (2017) CIELab L* channel specular highlight detector:
    Spherical grape berries produce localized specular reflection peaks on L*.
    Filtering with Nuske et al. (2011) 5-nearest-neighbor distance eliminates false positives
    from isolated foliage or trellis wires.
    """
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2Lab)
    L_channel = lab[:, :, 0]

    # If mask is provided, isolate the bunch area
    if binary_mask is not None:
        L_eval = cv2.bitwise_and(L_channel, L_channel, mask=binary_mask)
    else:
        L_eval = L_channel

    # Morphological Top-Hat filter extracts bright circular spots smaller than structuring element
    kernel_size = max(5, int(10.0 * pixels_per_mm))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(L_eval, cv2.MORPH_TOPHAT, kernel)

    # Threshold to isolate candidate specular reflections
    _, candidate_mask = cv2.threshold(tophat, int(min_prominence), 255, cv2.THRESH_BINARY)

    # Find connected component centroids
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_mask)

    raw_points = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # Valid berry specular highlight size constraint
        if 2 <= area <= (np.pi * (4.5 * pixels_per_mm) ** 2):
            raw_points.append(centroids[i])

    if len(raw_points) == 0:
        return {
            "aquino_berry_count": 0,
            "berry_centers_xy": [],
            "filtered_out_noise_points": 0,
            "regime": classify_illumination_regime(bgr_image)
        }

    pts = np.array(raw_points, dtype=np.float32)

    # Nuske et al. (2011) 5-neighbor proximity filtering:
    # Grape berries naturally cluster tightly together on a rachis
    valid_berry_points = []
    max_neighbor_distance_px = 25.0 * pixels_per_mm  # within ~25mm neighborhood

    if len(pts) >= 5:
        nbrs = NearestNeighbors(n_neighbors=5, algorithm="auto").fit(pts)
        distances, indices = nbrs.kneighbors(pts)
        # Average distance to 4 nearest neighbors (excluding self)
        mean_4_dist = np.mean(distances[:, 1:], axis=1)
        valid_indices = np.where(mean_4_dist <= max_neighbor_distance_px)[0]
        valid_berry_points = pts[valid_indices].tolist()
    else:
        valid_berry_points = pts.tolist()

    regime_info = classify_illumination_regime(bgr_image)

    return {
        "aquino_berry_count": len(valid_berry_points),
        "raw_candidate_peaks": len(pts),
        "filtered_out_noise_points": len(pts) - len(valid_berry_points),
        "berry_centers_xy": [[round(p[0], 1), round(p[1], 1)] for p in valid_berry_points],
        "regime": regime_info
    }
