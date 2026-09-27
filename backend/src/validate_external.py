from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.curve_analysis import calculate_threshold_bands  # noqa: E402
from app.predictor import FEATURES, get_model_info, load_model  # noqa: E402

GEOMETRY_FEATURES = FEATURES[:-1]
TARGET = "s11"


def parse_args():
    parser = ArgumentParser(
        description="Évalue le modèle sans réentraînement sur des données externes."
    )
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--validation-kind",
        choices=["cst-unseen", "measurement"],
        default="cst-unseen",
    )
    parser.add_argument("--threshold-db", type=float, default=-10.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-mae-db", type=float)
    parser.add_argument("--max-resonance-error-ghz", type=float)
    parser.add_argument("--max-bandwidth-error-ghz", type=float)
    return parser.parse_args()


def geometry_key(row) -> tuple[float, ...]:
    return tuple(round(float(row[name]), 12) for name in GEOMETRY_FEATURES)


def summarize(values: list[float]) -> dict | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset externe introuvable : {dataset_path}")

    data = pd.read_csv(dataset_path)
    missing = [name for name in FEATURES + [TARGET] if name not in data.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    if len(data) < 2:
        raise ValueError("Le dataset externe doit contenir au moins deux points")

    selected = data[FEATURES + [TARGET]].copy()
    if selected.isna().any().any():
        raise ValueError("Le dataset externe contient des valeurs manquantes")
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError("Le dataset externe contient des valeurs non finies")

    info = get_model_info()
    training_geometries = {
        geometry_key(row)
        for row in info.get("geometry_domain", {}).get(
            "training_geometries",
            [],
        )
    }
    external_geometries = {
        geometry_key(row)
        for row in selected[GEOMETRY_FEATURES].drop_duplicates().to_dict(
            orient="records"
        )
    }
    overlap = training_geometries.intersection(external_geometries)
    if args.validation_kind == "cst-unseen" and overlap:
        raise ValueError(
            f"{len(overlap)} géométrie(s) externe(s) sont déjà présentes dans "
            "l'entraînement. Utilisez des géométries inédites."
        )

    model = load_model()
    predictions = np.asarray(model.predict(selected[FEATURES]), dtype=float)
    truth = selected[TARGET].to_numpy(dtype=float)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Le modèle a produit une valeur non finie")

    metrics = {
        "mae_db": float(mean_absolute_error(truth, predictions)),
        "rmse_db": float(np.sqrt(mean_squared_error(truth, predictions))),
        "r2": float(r2_score(truth, predictions)),
        "absolute_error_p95_db": float(
            np.quantile(np.abs(truth - predictions), 0.95)
        ),
    }

    evaluated = selected.copy()
    evaluated["_prediction"] = predictions
    if "configuration_id" in data.columns:
        evaluated["_configuration_id"] = data["configuration_id"].astype(str)
    else:
        evaluated["_configuration_id"] = [
            "|".join(str(value) for value in geometry_key(row))
            for row in selected[GEOMETRY_FEATURES].to_dict(orient="records")
        ]

    resonance_errors: list[float] = []
    minimum_errors: list[float] = []
    bandwidth_errors: list[float] = []
    curve_reports: list[dict] = []
    for configuration_id, curve in evaluated.groupby("_configuration_id"):
        curve = curve.sort_values("FREQUENCY")
        true_minimum = curve.loc[curve[TARGET].idxmin()]
        predicted_minimum = curve.loc[curve["_prediction"].idxmin()]
        resonance_error = abs(
            float(predicted_minimum["FREQUENCY"])
            - float(true_minimum["FREQUENCY"])
        )
        minimum_error = abs(
            float(predicted_minimum["_prediction"])
            - float(true_minimum[TARGET])
        )
        true_curve = [
            {"frequency": row["FREQUENCY"], "s11": row[TARGET]}
            for _, row in curve.iterrows()
        ]
        predicted_curve = [
            {"frequency": row["FREQUENCY"], "s11": row["_prediction"]}
            for _, row in curve.iterrows()
        ]
        true_bandwidth = sum(
            band["bandwidth"]
            for band in calculate_threshold_bands(
                true_curve,
                args.threshold_db,
            )
        )
        predicted_bandwidth = sum(
            band["bandwidth"]
            for band in calculate_threshold_bands(
                predicted_curve,
                args.threshold_db,
            )
        )
        bandwidth_error = abs(predicted_bandwidth - true_bandwidth)
        resonance_errors.append(resonance_error)
        minimum_errors.append(minimum_error)
        bandwidth_errors.append(bandwidth_error)
        curve_reports.append(
            {
                "configuration_id": configuration_id,
                "rows": int(len(curve)),
                "curve_mae_db": float(
                    mean_absolute_error(curve[TARGET], curve["_prediction"])
                ),
                "resonance_frequency_error_ghz": resonance_error,
                "minimum_s11_error_db": minimum_error,
                "bandwidth_error_ghz": bandwidth_error,
            }
        )

    criteria = {
        "max_mae_db": args.max_mae_db,
        "max_resonance_error_ghz": args.max_resonance_error_ghz,
        "max_bandwidth_error_ghz": args.max_bandwidth_error_ghz,
    }
    checks = []
    if args.max_mae_db is not None:
        checks.append(metrics["mae_db"] <= args.max_mae_db)
    if args.max_resonance_error_ghz is not None:
        checks.append(max(resonance_errors) <= args.max_resonance_error_ghz)
    if args.max_bandwidth_error_ghz is not None:
        checks.append(max(bandwidth_errors) <= args.max_bandwidth_error_ghz)

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": info.get("model_id"),
        "model_scope": info.get("antenna_scope"),
        "dataset": str(dataset_path),
        "validation_kind": args.validation_kind,
        "rows": int(len(data)),
        "configurations": int(evaluated["_configuration_id"].nunique()),
        "training_geometry_overlap_count": int(len(overlap)),
        "threshold_db": args.threshold_db,
        "metrics": metrics,
        "curve_metrics": {
            "resonance_frequency_absolute_error_ghz": summarize(
                resonance_errors
            ),
            "minimum_s11_absolute_error_db": summarize(minimum_errors),
            "bandwidth_absolute_error_ghz": summarize(bandwidth_errors),
        },
        "acceptance_criteria": criteria,
        "acceptance_passed": all(checks) if checks else None,
        "curves": curve_reports,
    }

    output_path = (
        args.output.resolve()
        if args.output
        else dataset_path.with_suffix(".validation.json")
    )
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_output, output_path)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Rapport sauvegardé : {output_path}")
    if checks and not all(checks):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
