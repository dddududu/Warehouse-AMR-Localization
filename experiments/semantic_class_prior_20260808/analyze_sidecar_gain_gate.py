from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = (
    "semi_dynamic_point_ratio",
    "raw_icp_inlier_ratio",
    "raw_icp_rmse",
    "sidecar_icp_inlier_ratio",
    "sidecar_icp_rmse",
    "icp_inlier_ratio_gain",
    "icp_rmse_gain",
    "pose_translation_delta_m",
    "pose_yaw_delta_deg",
    "raw_deep_match_probability",
    "raw_temporal_score",
)


def load_sidecar_examples(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    features: list[list[float]] = []
    deltas: list[float] = []
    for row in report["frame_results"]:
        sidecar_features = row.get("semantic_sidecar_features")
        raw_error = row.get("raw_tracker_position_error_m")
        if not row.get("semantic_sidecar_applied") or sidecar_features is None or raw_error is None:
            continue
        features.append([float(sidecar_features[name]) for name in FEATURE_NAMES])
        deltas.append(float(row["position_error_m"]) - float(raw_error))
    if not features:
        raise ValueError(f"No comparable sidecar examples in {path}.")
    return np.asarray(features, dtype=np.float64), np.asarray(deltas, dtype=np.float64)


def evaluate_transfer(
    train_features: np.ndarray,
    train_deltas: np.ndarray,
    test_features: np.ndarray,
    test_deltas: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    train_labels = train_deltas <= -0.01
    model = make_pipeline(
        SimpleImputer(),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", C=0.5, max_iter=2000, random_state=0),
    )
    model.fit(train_features, train_labels)
    probabilities = model.predict_proba(test_features)[:, 1]
    selected = probabilities >= float(threshold)
    selected_deltas = test_deltas[selected]
    return {
        "selected_count": int(selected.sum()),
        "selected_ratio": float(selected.mean()),
        "mean_delta_m": float(selected_deltas.mean()) if selected_deltas.size else 0.0,
        "material_benefit_count": int((selected_deltas <= -0.01).sum()),
        "material_regression_count": int((selected_deltas >= 0.01).sum()),
        "material_benefit_precision": (
            float((selected_deltas <= -0.01).mean()) if selected_deltas.size else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit sidecar-gain transfer without Oct.12 labels.")
    parser.add_argument("--ccw", required=True)
    parser.add_argument("--cw", required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    ccw_features, ccw_deltas = load_sidecar_examples(args.ccw)
    cw_features, cw_deltas = load_sidecar_examples(args.cw)
    output = {
        "benefit_definition": "sidecar position error improves by at least 0.01 m",
        "threshold": float(args.threshold),
        "feature_names": list(FEATURE_NAMES),
        "datasets": {
            "ccw": {
                "count": int(ccw_deltas.size),
                "mean_delta_m": float(ccw_deltas.mean()),
                "material_benefit_count": int((ccw_deltas <= -0.01).sum()),
                "material_regression_count": int((ccw_deltas >= 0.01).sum()),
            },
            "cw": {
                "count": int(cw_deltas.size),
                "mean_delta_m": float(cw_deltas.mean()),
                "material_benefit_count": int((cw_deltas <= -0.01).sum()),
                "material_regression_count": int((cw_deltas >= 0.01).sum()),
            },
        },
        "cross_route_transfer": {
            "ccw_to_cw": evaluate_transfer(
                ccw_features, ccw_deltas, cw_features, cw_deltas, args.threshold
            ),
            "cw_to_ccw": evaluate_transfer(
                cw_features, cw_deltas, ccw_features, ccw_deltas, args.threshold
            ),
        },
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
