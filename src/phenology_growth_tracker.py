"""
SmartVine AI — Longitudinal Berry Growth Curve & Veraison Phenology Tracker
Translates Section II-E & III (Datasets 1-4) of Seng et al. (IEEE Access, 2018).
Implements Coombe & McCarthy (2000) double-sigmoid grape growth modeling,
daily veraison hue shift tracking, and 4-phase physiological staging.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from scipy.optimize import curve_fit


def double_sigmoid_growth(
    t: np.ndarray,
    v1: float,
    k1: float,
    t1: float,
    v2: float,
    k2: float,
    t2: float,
    v0: float
) -> np.ndarray:
    """
    Models the classic double-sigmoid growth curve of grape berries (Coombe & McCarthy, 2000):
    - Stage I (v1, k1, t1): Rapid post-flowering cell division & expansion
    - Herbaceous Lag Phase: Seed hardening, slow volumetric growth
    - Stage II (v2, k2, t2): Veraison softening, sugar accumulation & rapid cell expansion
    - v0: Initial post-fruit-set berry volume
    """
    stage1 = v1 / (1.0 + np.exp(-np.clip(k1 * (t - t1), -50, 50)))
    stage2 = v2 / (1.0 + np.exp(-np.clip(k2 * (t - t2), -50, 50)))
    return v0 + stage1 + stage2


def fit_double_sigmoid(
    days_after_flowering: List[float],
    volumes: List[float]
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Fits double-sigmoid curve parameters via non-linear least squares.
    """
    t = np.array(days_after_flowering, dtype=float)
    y = np.array(volumes, dtype=float)

    if len(t) < 7:
        # Not enough data points to constrain 7 parameters stably
        return None, False

    # Sensible agronomical priors:
    # v1 ~ max(y)*0.4, k1 ~ 0.15, t1 ~ 25 DAF
    # v2 ~ max(y)*0.55, k2 ~ 0.12, t2 ~ 65 DAF, v0 ~ min(y)
    p0 = [
        max(y) * 0.40, 0.15, 25.0,
        max(y) * 0.55, 0.12, 65.0,
        min(y)
    ]
    bounds = (
        [0.0, 0.01, 10.0, 0.0, 0.01, 45.0, 0.0],
        [max(y) * 2.0, 1.0, 45.0, max(y) * 2.0, 1.0, 100.0, max(y)]
    )

    try:
        popt, _ = curve_fit(double_sigmoid_growth, t, y, p0=p0, bounds=bounds, maxfev=2000)
        return popt, True
    except Exception:
        return None, False


def analyze_bunch_time_series(
    days_after_flowering: List[float],
    bunch_volumes_mm3: List[float],
    dominant_hues_deg: List[float]
) -> Dict[str, Any]:
    """
    Analyzes multi-temporal grapevine longitudinal captures (GrapeCS-ML Datasets 1-4):
    Calculates instantaneous volumetric growth rate dV/dt and hue transition velocity dH/dt
    to classify the vine into one of four physiological growth phases.
    """
    if len(days_after_flowering) == 0:
        raise ValueError("Time series requires at least one observation.")

    t = np.array(days_after_flowering, dtype=float)
    vol = np.array(bunch_volumes_mm3, dtype=float)
    hue = np.array(dominant_hues_deg, dtype=float)

    if len(t) > 1:
        dv_dt = np.gradient(vol, t)
        dh_dt = np.gradient(hue, t)
        latest_dv = float(dv_dt[-1])
        latest_dh = float(dh_dt[-1])
    else:
        latest_dv = 0.0
        latest_dh = 0.0

    last_t = t[-1]

    # Staging rules from Coombe & McCarthy (2000) and Seng et al. (2018)
    if last_t < 35 and latest_dv > 5.0:
        phase = "Stage I: Post-Flowering Rapid Cell Division (Hard Green Berries)"
        description = "Early berry formation; high organic acid accumulation, hard green pericarp."
        harvest_readiness = "Not Ready — Early Season"
    elif 35 <= last_t < 55 and abs(latest_dv) <= 5.0:
        phase = "Stage II: Herbaceous Lag Phase (Seed Hardening)"
        description = "Embryo development and seed coat hardening; volume growth temporarily plateaus."
        harvest_readiness = "Not Ready — Mid Season"
    elif abs(latest_dh) > 1.2 or (last_t >= 55 and latest_dv > 5.0):
        phase = "Stage III: Veraison & Second Growth (Berry Softening & Color Shift)"
        description = "Rapid skin softening, anthocyanin synthesis (reds) or chlorophyll breakdown (whites), sugar import starts."
        harvest_readiness = "Pre-Harvest Ripening"
    else:
        phase = "Stage IV: Engustment (Aroma/Flavor Intensification — Harvest Ready)"
        description = "Peak sugar accumulation, acid degradation, synthesis of monoterpenes and volatile thiols."
        harvest_readiness = "Optimal Harvest Window"

    fit_params, fit_success = fit_double_sigmoid(days_after_flowering, bunch_volumes_mm3)

    return {
        "current_day": int(round(last_t)),
        "latest_volume_mm3": round(float(vol[-1]), 1),
        "latest_dominant_hue": round(float(hue[-1]), 2),
        "volume_growth_rate_mm3_per_day": round(latest_dv, 2),
        "hue_shift_rate_deg_per_day": round(latest_dh, 2),
        "phenological_stage": phase,
        "phase_description": description,
        "harvest_readiness": harvest_readiness,
        "sigmoid_fit_converged": fit_success,
        "fitted_parameters": fit_params.tolist() if fit_success else None
    }
