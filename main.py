"""
SmartVine AI — Unified Command-Line Runner (CLI)
Translates Seng et al. (IEEE Access, 2018) into a unified terminal tool.
Executes end-to-end precision viticulture analysis on single captures, batch sequences,
or launches FastAPI & Streamlit services.
"""

import argparse
import os
import sys
import json
import cv2
import numpy as np

from src.calibration import calibrate_image_and_scale
from src.macbeth_colorchecker import calibrate_with_macbeth
from src.berry_counter import analyze_bunch_morphology
from src.quality_regressor import extract_ripeness_features, load_regressors, predict_chemical_metrics
from src.disease_classifier import predict_disease
from src.pruning_and_seeds import detect_winter_buds_and_cutting_point
from src.yield_3d_extrapolator import estimate_cluster_3d_volume, extrapolate_vineyard_block_yield
from src.data_loader import generate_benchmark_sample
from app.api import segment_clusters_fallback


def run_pipeline_on_image(
    image_path: str,
    marker_mm: float = 25.0,
    cultivar: str = "Sauvignon Blanc",
    output_dir: str = "outputs"
) -> dict:
    """
    Executes full 5-domain agronomic pipeline on a single image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise ValueError(f"Could not decode image at {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"🍇 SMARTVINE AI — PRECISION VITICULTURE PIPELINE")
    print(f"   Target Image: {image_path}")
    print(f"   Cultivar:     {cultivar}")
    print(f"=======================================================\n")

    # 1. Calibration
    print("[1/5] Calibrating illumination & scale marker...")
    calibrated_bgr, scale, calib_info = calibrate_image_and_scale(bgr_img, known_circle_mm=marker_mm)
    calibrated_bgr, ccm, macbeth_info = calibrate_with_macbeth(calibrated_bgr)
    print(f"      Scale: {scale:.3f} px/mm | Marker Detected: {calib_info['marker_detected']}")

    # 2. Bunch Instance Segmentation
    print("[2/5] Segmenting grape clusters...")
    cluster_masks = segment_clusters_fallback(calibrated_bgr)
    print(f"      Detected {len(cluster_masks)} grape cluster(s).")

    # 3. Cluster Morphology, Berry Counting & Ripeness Regressors
    print("[3/5] Extracting berry counts, bunch solidity & chemical Brix/pH...")
    brix_model, ph_model = load_regressors("models")
    clusters_report = []
    bunch_masses = []

    annotated_img = calibrated_bgr.copy()

    for idx, b_mask in enumerate(cluster_masks):
        morphology = analyze_bunch_morphology(calibrated_bgr, b_mask, pixels_per_mm=scale)
        volume_3d = estimate_cluster_3d_volume(b_mask, pixels_per_mm=scale)
        bunch_masses.append(volume_3d["estimated_bunch_mass_g"])

        feats = np.array(
            extract_ripeness_features(calibrated_bgr, b_mask, scale_mm_per_px=1.0 / scale)
        ).reshape(1, -1)

        pred_brix, pred_ph = predict_chemical_metrics(brix_model, ph_model, feats)

        bud_info = detect_winter_buds_and_cutting_point(calibrated_bgr, bunch_binary_mask=b_mask)

        # Draw detected berries on annotated image
        for (cx, cy, cr) in morphology.get("detected_circles", []):
            cv2.circle(annotated_img, (cx, cy), cr, (0, 255, 0), 2)
            cv2.circle(annotated_img, (cx, cy), 2, (0, 0, 255), -1)

        # Draw cutting point
        cp = bud_info.get("peduncle_cutting_point_xy")
        if cp:
            cv2.circle(annotated_img, cp, 8, (255, 255, 0), -1)
            cv2.putText(annotated_img, "CUT", (cp[0] + 10, cp[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        clusters_report.append({
            "cluster_id": idx + 1,
            "visible_berries": morphology["visible_berry_count"],
            "estimated_total_berries_3d": morphology["estimated_total_berries"],
            "compactness_solidity": morphology["compactness_solidity"],
            "botrytis_rot_risk": morphology["botrytis_rot_risk"],
            "estimated_bunch_mass_g": volume_3d["estimated_bunch_mass_g"],
            "predicted_brix_tss": round(pred_brix, 2),
            "predicted_ph": round(pred_ph, 2),
            "harvest_window": "Optimal Harvest Window" if pred_brix >= 21.5 else "Developing",
            "peduncle_cutting_xy": cp
        })

    # 4. Foliar Disease Screening
    print("[4/5] Evaluating canopy foliar disease & Grad-CAM...")
    disease_res = predict_disease(calibrated_bgr)
    print(f"      Condition: {disease_res['diagnosed_condition']} ({disease_res['confidence_percent']}%)")
    if disease_res["spray_alert_triggered"]:
        print(f"      [!] SPRAY ALERT TRIGGERED: {disease_res['agronomic_action']}")

    # 5. Block Yield Extrapolation
    print("[5/5] Extrapolating vineyard block yield...")
    block_yield = extrapolate_vineyard_block_yield(
        bunch_masses_g=bunch_masses,
        clusters_detected_per_frame=len(clusters_report)
    )
    print(f"      Yield Forecast: {block_yield['yield_tonnes_per_ha']} Tonnes/Hectare (95% CI: {block_yield['confidence_interval_95_t_ha']})")

    # Save outputs
    annotated_path = os.path.join(output_dir, "annotated_detection.png")
    cam_path = os.path.join(output_dir, "disease_gradcam.png")
    json_path = os.path.join(output_dir, "agronomic_report.json")

    cv2.imwrite(annotated_path, annotated_img)
    cv2.imwrite(cam_path, disease_res["gradcam_overlay_bgr"])

    full_report = {
        "source_image": image_path,
        "cultivar": cultivar,
        "calibration": {
            "scale_px_per_mm": round(scale, 4),
            "marker_detected": calib_info["marker_detected"]
        },
        "clusters": clusters_report,
        "disease_diagnosis": {
            "condition": disease_res["diagnosed_condition"],
            "confidence_percent": disease_res["confidence_percent"],
            "spray_alert": disease_res["spray_alert_triggered"],
            "action": disease_res["agronomic_action"]
        },
        "block_yield": block_yield,
        "output_artifacts": {
            "annotated_detection_image": annotated_path,
            "disease_gradcam_image": cam_path
        }
    }

    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n[✓] Analysis Complete! Reports saved to '{output_dir}/'")
    return full_report


def main():
    parser = argparse.ArgumentParser(
        description="SmartVine AI — Precision Viticulture Computer Vision & Machine Learning CLI"
    )
    parser.add_argument("--image", type=str, help="Path to input vineyard capture")
    parser.add_argument("--marker-mm", type=float, default=25.0, help="Diameter of reference marker circle in mm")
    parser.add_argument("--cultivar", type=str, default="Sauvignon Blanc", help="Grape variety name")
    parser.add_argument("--output", type=str, default="outputs", help="Directory for output reports and visual artifacts")
    parser.add_argument("--benchmark", action="store_true", help="Generate and run benchmark test on synthetic GrapeCS-ML sample")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI REST backend on port 8000")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit interactive dashboard on port 7860")

    args = parser.parse_args()

    if args.serve:
        import uvicorn
        print("Launching SmartVine AI FastAPI backend on http://0.0.0.0:8000 ...")
        uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)
        return

    if args.dashboard:
        print("Launching Streamlit Agronomic Dashboard on port 7860 ...")
        os.system("streamlit run app/dashboard.py --server.port 7860")
        return

    if args.benchmark or args.image is None:
        print("[*] Generating high-fidelity GrapeCS-ML synthetic vineyard frame...")
        os.makedirs("sample_data", exist_ok=True)
        img, mask, meta = generate_benchmark_sample(cultivar=args.cultivar, bunch_id="Bench_01")
        sample_path = "sample_data/benchmark_vineyard_sample.jpg"
        cv2.imwrite(sample_path, img)
        target_img = sample_path
    else:
        target_img = args.image

    run_pipeline_on_image(
        image_path=target_img,
        marker_mm=args.marker_mm,
        cultivar=args.cultivar,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
