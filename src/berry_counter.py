"""
SmartVine AI — Berry Counting, Bunch Morphology & Compactness Engine
Translates Section II-A & II-D of Seng et al. (IEEE Access, 2018).
Implements Circular Hough Transform (Murillo-Bracamontes et al.), Distance Transform + Watershed,
3D occlusion factor (Nuske et al.), and Cubero et al. bunch compactness solidity ratio
for Botrytis bunch rot risk scoring.
"""

from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np


def detect_berries_circular_hough(
    gray_masked: np.ndarray,
    pixels_per_mm: float = 2.5
) -> Tuple[int, List[Tuple[int, int, int]]]:
    """
    Detects spherical berries using Circular Hough Transform with physical scale constraints.
    Typical grape berry diameter ranges from 8mm to 28mm (radius 4mm to 14mm).
    """
    blurred = cv2.GaussianBlur(gray_masked, (5, 5), 1.5)

    min_dist = max(5, int(8.0 * pixels_per_mm))
    min_radius = max(3, int(3.5 * pixels_per_mm))
    max_radius = max(min_radius + 5, int(15.0 * pixels_per_mm))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=50,
        param2=18,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    detected_circles: List[Tuple[int, int, int]] = []
    if circles is not None:
        # Convert circle parameters to integers
        circles_round = np.round(circles[0, :]).astype(int)
        for (x, y, r) in circles_round:
            # Verify circle center falls inside non-zero mask region
            if 0 <= y < gray_masked.shape[0] and 0 <= x < gray_masked.shape[1]:
                if gray_masked[y, x] > 0:
                    detected_circles.append((int(x), int(y), int(r)))

    return len(detected_circles), detected_circles


def detect_berries_watershed(
    binary_mask: np.ndarray,
    bgr_image: Optional[np.ndarray] = None
) -> Tuple[int, np.ndarray]:
    """
    Distance Transform + Watershed segmentation to separate tightly clumped, touching berries.
    """
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area using distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.40 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region between foreground and background
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    if bgr_image is None:
        dummy_img = cv2.merge([binary_mask, binary_mask, binary_mask])
    else:
        dummy_img = bgr_image.copy()

    cv2.watershed(dummy_img, markers)
    berry_count = max(0, ret - 1)
    return berry_count, markers


def compute_bunch_compactness(binary_mask: np.ndarray) -> Tuple[float, float, str, Dict[str, float]]:
    """
    Computes bunch compactness according to Cubero et al. and OIV descriptor 204:
    - Solidity = Bunch Area / Convex Hull Area
    - High solidity (> 0.86) indicates tight cluster vulnerable to Botrytis bunch rot.
    - Low solidity (< 0.78) indicates loose, open cluster with high air penetration.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, "Low", {"bunch_area_px": 0, "hull_area_px": 0, "aspect_ratio": 1.0}

    largest_cnt = max(contours, key=cv2.contourArea)
    bunch_area = cv2.contourArea(largest_cnt)
    hull = cv2.convexHull(largest_cnt)
    hull_area = cv2.contourArea(hull)

    solidity = float(bunch_area / (hull_area + 1e-6))
    solidity = max(0.0, min(1.0, solidity))

    x, y, w, h = cv2.boundingRect(largest_cnt)
    aspect_ratio = float(h / (w + 1e-6))

    if solidity > 0.86:
        rot_risk = "High"
    elif solidity > 0.78:
        rot_risk = "Moderate"
    else:
        rot_risk = "Low"

    descriptors = {
        "bunch_area_px": float(bunch_area),
        "hull_area_px": float(hull_area),
        "aspect_ratio": round(aspect_ratio, 2),
        "solidity": round(solidity, 3)
    }

    return round(solidity, 3), round(bunch_area, 1), rot_risk, descriptors


def analyze_bunch_morphology(
    bgr_image: np.ndarray,
    binary_mask: np.ndarray,
    pixels_per_mm: float = 2.5,
    occlusion_factor: float = 1.45,
    mean_single_berry_weight_g: float = 1.85
) -> Dict[str, Any]:
    """
    Unified bunch analysis combining Hough berry detection, watershed,
    3D occlusion multiplier, and Cubero compactness index.
    """
    # Isolate bunch region
    masked_bunch = cv2.bitwise_and(bgr_image, bgr_image, mask=binary_mask)
    gray = cv2.cvtColor(masked_bunch, cv2.COLOR_BGR2GRAY)

    # 1. Circular Hough Berry Detection
    hough_count, circles = detect_berries_circular_hough(gray, pixels_per_mm=pixels_per_mm)

    # 2. Watershed Fallback / Complement
    ws_count, _ = detect_berries_watershed(binary_mask, bgr_image)

    # Robust visible count: take max of Hough and Watershed
    visible_berries = max(hough_count, ws_count)
    if visible_berries == 0 and np.sum(binary_mask > 0) > 100:
        # Fallback estimation based on average projected berry area
        avg_berry_area_px = np.pi * ((5.5 * pixels_per_mm) ** 2)
        visible_berries = int(np.sum(binary_mask > 0) / (avg_berry_area_px + 1e-6))

    # 3. 3D Occlusion Multiplier (Nuske et al. empirical 2D-to-3D ratio: 1.45)
    estimated_total_berries = int(round(visible_berries * occlusion_factor))
    estimated_bunch_weight_g = round(estimated_total_berries * mean_single_berry_weight_g, 1)

    # 4. Bunch Compactness & Botrytis Rot Risk
    solidity, bunch_area_px, rot_risk, descriptors = compute_bunch_compactness(binary_mask)

    # Convert area to real-world mm^2
    projected_area_mm2 = round(bunch_area_px / (pixels_per_mm ** 2), 1)

    return {
        "visible_berry_count": visible_berries,
        "hough_berry_count": hough_count,
        "watershed_berry_count": ws_count,
        "estimated_total_berries": estimated_total_berries,
        "estimated_bunch_weight_g": estimated_bunch_weight_g,
        "compactness_solidity": solidity,
        "botrytis_rot_risk": rot_risk,
        "projected_area_mm2": projected_area_mm2,
        "aspect_ratio": descriptors["aspect_ratio"],
        "detected_circles": circles[:50]  # Cap list to top 50 for clean JSON serialization
    }
