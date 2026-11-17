"""
SmartVine AI — GrapeCS-ML Dataset 5 Ground-Truth & Excel Parser
Translates Section III & Step 9 of the build specification.
Parses the Charles Sturt University laboratory chemical dataset (Dyostem Excel spreadsheet),
matches hand-segmented ground-truth masks with raw captures, extracts multi-spectral features,
and includes an automated benchmark synthesizer for offline evaluation.
"""

from typing import Tuple, List, Dict, Any, Optional
import os
import glob
import cv2
import pandas as pd
import numpy as np
from src.quality_regressor import extract_ripeness_features


def build_dataset5_training_table(
    dataset5_dir: str,
    excel_path: str,
    pixels_per_mm: float = 2.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Matches Sauvignon Blanc field images and hand-segmented ground-truth masks in Dataset 5
    with laboratory chemical metrics:
    - TSS (Total Soluble Solids in degrees Brix)
    - pH
    - Berry Number
    - Bunch Weight (g)
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Laboratory ground truth spreadsheet not found: {excel_path}")

    # Read Excel sheet
    df_lab = pd.read_excel(excel_path)
    df_lab.columns = [str(c).strip() for c in df_lab.columns]

    features_list, brix_list, ph_list, berry_count_list = [], [], [], []

    for _, row in df_lab.iterrows():
        sample_id = str(row.get("Sample_ID", row.get("Bunch_ID", ""))).strip()
        if not sample_id:
            continue

        # Look for matching image and mask
        img_matches = glob.glob(os.path.join(dataset5_dir, "**", f"*{sample_id}*.jpg"), recursive=True)
        mask_matches = glob.glob(os.path.join(dataset5_dir, "**", f"*{sample_id}*mask*.png"), recursive=True)

        if img_matches and mask_matches:
            bgr = cv2.imread(img_matches[0])
            mask = cv2.imread(mask_matches[0], cv2.IMREAD_GRAYSCALE)

            if bgr is not None and mask is not None:
                feats = extract_ripeness_features(bgr, mask, scale_mm_per_px=1.0 / pixels_per_mm)
                features_list.append(feats)

                # Extract chemical targets with robust column name handling
                brix_val = float(row.get("TSS", row.get("Brix", 20.0)))
                ph_val = float(row.get("pH", 3.25))
                berry_val = int(row.get("Berry_Number", row.get("Berry_Count", 75)))

                brix_list.append(brix_val)
                ph_list.append(ph_val)
                berry_count_list.append(berry_val)

    if len(features_list) == 0:
        raise ValueError(f"No matching image-mask pairs found in {dataset5_dir}")

    return (
        np.array(features_list, dtype=np.float32),
        np.array(brix_list, dtype=np.float32),
        np.array(ph_list, dtype=np.float32),
        np.array(berry_count_list, dtype=np.int32)
    )


def generate_benchmark_sample(
    cultivar: str = "Sauvignon_Blanc",
    bunch_id: str = "SB_01",
    img_size: Tuple[int, int] = (640, 640),
    maturity_stage: str = "Ripe"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generates a high-fidelity synthetic vineyard frame adhering to GrapeCS-ML Dataset 5:
    - Grape bunch cluster with individual spherical berries
    - Foliage leaf camouflage in background
    - Red 25mm circular calibration marker on white strip
    - Ground-truth chemical metadata
    """
    h, w = img_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    # 1. Background vine leaves (green canopy texture)
    cv2.rectangle(img, (0, 0), (w, h), (35, 75, 40), -1)
    # Add random foliage patches
    for _ in range(12):
        cx, cy = np.random.randint(50, w - 50), np.random.randint(50, h - 50)
        axes = (np.random.randint(40, 100), np.random.randint(25, 60))
        angle = np.random.randint(0, 180)
        color = (
            int(np.random.randint(20, 45)),
            int(np.random.randint(85, 140)),
            int(np.random.randint(30, 60))
        )
        cv2.ellipse(img, (cx, cy), axes, angle, 0, 360, color, -1)

    # 2. Add White strip with Red 25mm Reference Circle Marker (top-left)
    cv2.rectangle(img, (20, 20), (140, 90), (240, 240, 240), -1)
    # Red circle with diameter ~ 62 px (giving ~2.5 px/mm)
    marker_center = (80, 55)
    marker_radius = 31
    cv2.circle(img, marker_center, marker_radius, (20, 20, 210), -1)

    # 3. Draw Grape Cluster (conical bunch shape)
    bunch_center_x = 340
    bunch_top_y = 180
    bunch_bottom_y = 520

    # Cultivar color profiles
    is_white = "Sauvignon" in cultivar or "Chardonnay" in cultivar
    if is_white:
        if maturity_stage == "Ripe":
            base_color = (65, 175, 190)    # Translucent yellow-gold BGR
            ground_truth_brix = 22.4
            ground_truth_ph = 3.32
        else:
            base_color = (55, 160, 110)    # Immature green BGR
            ground_truth_brix = 16.2
            ground_truth_ph = 2.98
    else:
        # Red / Black cultivar (Shiraz / Merlot)
        if maturity_stage == "Ripe":
            base_color = (75, 30, 55)      # Deep anthocyanin purple BGR
            ground_truth_brix = 24.1
            ground_truth_ph = 3.58
        else:
            base_color = (60, 120, 80)     # Veraison transition
            ground_truth_brix = 18.0
            ground_truth_ph = 3.12

    # Draw berries in conical cluster
    berry_radius = 16  # ~13mm diameter at 2.5 px/mm
    berries_placed = 0

    for y in range(bunch_top_y, bunch_bottom_y, int(berry_radius * 1.3)):
        # Cluster narrows toward the bottom
        progress = (y - bunch_top_y) / float(bunch_bottom_y - bunch_top_y)
        current_width = int((1.0 - 0.70 * progress) * 190)
        num_row_berries = max(1, int(current_width / (berry_radius * 1.5)))

        xs = np.linspace(bunch_center_x - current_width // 2, bunch_center_x + current_width // 2, num_row_berries)
        for bx in xs:
            bx = int(bx + np.random.randint(-4, 5))
            by = int(y + np.random.randint(-4, 5))

            # Berry color jitter
            b_val = int(np.clip(base_color[0] + np.random.randint(-10, 11), 0, 255))
            g_val = int(np.clip(base_color[1] + np.random.randint(-10, 11), 0, 255))
            r_val = int(np.clip(base_color[2] + np.random.randint(-10, 11), 0, 255))

            # Draw spherical gradient
            cv2.circle(img, (bx, by), berry_radius, (b_val, g_val, r_val), -1)
            cv2.circle(mask, (bx, by), berry_radius, 255, -1)

            # Specular highlight (spherical reflection)
            spec_center = (bx - int(berry_radius * 0.35), by - int(berry_radius * 0.35))
            spec_radius = max(2, int(berry_radius * 0.25))
            cv2.circle(img, spec_center, spec_radius, (245, 245, 245), -1)
            berries_placed += 1

    # Stem / Peduncle directly above cluster shoulder
    cv2.line(img, (bunch_center_x, bunch_top_y), (bunch_center_x, bunch_top_y - 45), (40, 70, 95), 5)

    metadata = {
        "bunch_id": bunch_id,
        "cultivar": cultivar,
        "maturity_stage": maturity_stage,
        "ground_truth_berry_count": berries_placed,
        "ground_truth_brix": ground_truth_brix,
        "ground_truth_ph": ground_truth_ph,
        "bunch_weight_g": round(berries_placed * 1.85, 1)
    }

    return img, mask, metadata
