from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.catalog import validate_antenna_selection  # noqa: E402

DATA_PATH = ROOT / "data" / "pik_clean_ml.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "s11_predictor.pkl"
INFO_PATH = MODEL_DIR / "model_info.json"

FEATURES = [
    "gap",
    "surface_width",
    "surface_length",
    "epsilon_r",
    "FREQUENCY",
]
GEOMETRY_FEATURES = FEATURES[:-1]
TARGET = "s11"
CV_SPLITS = 5
MODEL_PARAMETERS = {
    "objective": "reg:squarederror",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Entraîne et valide le modèle S11 par géométries complètes."
    )
    parser.add_argument(
        "--antenna-family-id",
        help="Identifiant de famille retourné par GET /api/antennas.",
    )
    parser.add_argument("--antenna-variant", help="Nom exact de la variante.")
    parser.add_argument(
        "--scope-verified",
        action="store_true",
        help=(
            "Atteste que le CSV correspond scientifiquement à la famille et à "
            "la variante fournies."
        ),
    )
    args = parser.parse_args()
    if bool(args.antenna_family_id) != bool(args.antenna_variant):
        parser.error(
            "--antenna-family-id et --antenna-variant doivent être fournis ensemble"
        )
    if args.scope_verified and not args.antenna_family_id:
        parser.error("--scope-verified exige une famille et une variante")
    return args


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model() -> XGBRegressor:
    return XGBRegressor(**MODEL_PARAMETERS)


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = [column for column in FEATURES + [TARGET] if column not in raw.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    selected = raw[FEATURES + [TARGET]].copy()
    if selected.isna().any().any():
        missing_counts = selected.isna().sum()
        raise ValueError(
            "Valeurs manquantes détectées : "
            + str(missing_counts[missing_counts > 0].to_dict())
        )
    if not np.isfinite(selected.to_numpy(dtype=float)).all():
        raise ValueError("Le dataset contient une valeur non finie")

    by_input = selected.groupby(FEATURES, dropna=False)[TARGET].agg(
        repetitions="size",
        distinct_targets="nunique",
        minimum="min",
        maximum="max",
    )
    conflicts = by_input[by_input["distinct_targets"] > 1]

    cleaned = (
        selected.groupby(FEATURES, as_index=False, dropna=False)[TARGET]
        .mean()
        .sort_values(GEOMETRY_FEATURES + ["FREQUENCY"])
        .reset_index(drop=True)
    )
    geometry_count = int(
        cleaned[GEOMETRY_FEATURES].drop_duplicates().shape[0]
    )
    if geometry_count < CV_SPLITS:
        raise ValueError(
            f"Au moins {CV_SPLITS} géométries sont requises pour la validation"
        )

    audit = {
        "raw_rows": int(len(raw)),
        "effective_rows_after_input_aggregation": int(len(cleaned)),
        "removed_repeated_input_rows": int(len(selected) - len(cleaned)),
        "fully_duplicated_rows": int(raw.duplicated().sum()),
        "conflicting_input_keys": int(len(conflicts)),
        "maximum_target_spread_db": (
            float((conflicts["maximum"] - conflicts["minimum"]).max())
            if len(conflicts)
            else 0.0
        ),
        "geometry_count": geometry_count,
        "width_always_equals_length": bool(
            np.allclose(raw["surface_width"], raw["surface_length"])
        ),
        "aggregation_policy": (
            "Les entrées identiques sont regroupées et leur cible S11 est moyennée."
        ),
    }
    return cleaned, audit


def grouped_validation(data: pd.DataFrame) -> tuple[dict, list[dict], dict]:
    geometry_index = pd.MultiIndex.from_frame(data[GEOMETRY_FEATURES])
    groups, _ = pd.factorize(geometry_index)
    predictions = np.empty(len(data), dtype=float)
    fold_metrics: list[dict] = []

    splitter = GroupKFold(n_splits=CV_SPLITS)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(data[FEATURES], data[TARGET], groups),
        start=1,
    ):
        model = build_model()
        model.fit(
            data.iloc[train_index][FEATURES],
            data.iloc[train_index][TARGET],
        )
        fold_predictions = model.predict(data.iloc[test_index][FEATURES])
        predictions[test_index] = fold_predictions
        metrics = regression_metrics(
            data.iloc[test_index][TARGET],
            fold_predictions,
        )
        fold_metrics.append(
            {
                "fold": fold,
                "test_rows": int(len(test_index)),
                "test_geometries": int(np.unique(groups[test_index]).size),
                **metrics,
            }
        )

    metrics = regression_metrics(data[TARGET], predictions)
    absolute_errors = np.abs(data[TARGET].to_numpy() - predictions)
    metrics["absolute_error_p90_db"] = float(np.quantile(absolute_errors, 0.90))
    metrics["absolute_error_p95_db"] = float(np.quantile(absolute_errors, 0.95))
    metrics["absolute_error_p99_db"] = float(np.quantile(absolute_errors, 0.99))

    evaluated = data[GEOMETRY_FEATURES + ["FREQUENCY", TARGET]].copy()
    evaluated["_prediction"] = predictions
    resonance_frequency_errors: list[float] = []
    minimum_s11_errors: list[float] = []
    curve_maes: list[float] = []

    for _, curve in evaluated.groupby(GEOMETRY_FEATURES):
        true_minimum = curve.loc[curve[TARGET].idxmin()]
        predicted_minimum = curve.loc[curve["_prediction"].idxmin()]
        resonance_frequency_errors.append(
            abs(
                float(predicted_minimum["FREQUENCY"])
                - float(true_minimum["FREQUENCY"])
            )
        )
        minimum_s11_errors.append(
            abs(float(predicted_minimum["_prediction"]) - float(true_minimum[TARGET]))
        )
        curve_maes.append(
            float(mean_absolute_error(curve[TARGET], curve["_prediction"]))
        )

    curve_metrics = {
        "resonance_frequency_absolute_error_ghz": summarize(
            resonance_frequency_errors
        ),
        "minimum_s11_absolute_error_db": summarize(minimum_s11_errors),
        "curve_mae_db": summarize(curve_maes),
    }
    return metrics, fold_metrics, curve_metrics


def geometry_domain(data: pd.DataFrame, ranges: dict) -> dict:
    geometries = data[GEOMETRY_FEATURES].drop_duplicates().reset_index(drop=True)
    matrix = geometries.to_numpy(dtype=float)
    minimum = np.array([ranges[name]["min"] for name in GEOMETRY_FEATURES])
    span = np.array(
        [max(ranges[name]["max"] - ranges[name]["min"], 1.0) for name in GEOMETRY_FEATURES]
    )
    normalized = (matrix - minimum) / span
    distances = np.sqrt(
        np.sum((normalized[:, None, :] - normalized[None, :, :]) ** 2, axis=2)
    )
    np.fill_diagonal(distances, np.inf)
    nearest_neighbors = np.min(distances, axis=1)
    return {
        "distance_definition": "Euclidean distance after min-max normalization",
        "warning_threshold": float(np.quantile(nearest_neighbors, 0.95)),
        "training_geometries": [
            {name: float(row[name]) for name in GEOMETRY_FEATURES}
            for row in geometries.to_dict(orient="records")
        ],
    }


def antenna_scope(args: Namespace) -> dict:
    if not args.antenna_family_id:
        return {
            "family_id": None,
            "family": None,
            "variant": None,
            "verified": False,
        }

    selection = validate_antenna_selection(
        args.antenna_family_id,
        args.antenna_variant,
    )
    return {**selection, "verified": bool(args.scope_verified)}


def main() -> None:
    args = parse_args()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_PATH)
    data, audit = prepare_data(raw)
    metrics, fold_metrics, curve_metrics = grouped_validation(data)

    ranges = {
        feature: {
            "min": float(data[feature].min()),
            "max": float(data[feature].max()),
        }
        for feature in FEATURES
    }
    trained_at = datetime.now(timezone.utc).isoformat()
    dataset_hash = file_sha256(DATA_PATH)

    model = build_model()
    model.fit(data[FEATURES], data[TARGET])

    temporary_model = MODEL_PATH.with_suffix(".pkl.tmp")
    joblib.dump(model, temporary_model)
    os.replace(temporary_model, MODEL_PATH)
    model_hash = file_sha256(MODEL_PATH)

    metadata = {
        "schema_version": 2,
        "model_id": f"s11-xgb-{dataset_hash[:12]}",
        "trained_at_utc": trained_at,
        "features": FEATURES,
        "geometry_features": GEOMETRY_FEATURES,
        "target": TARGET,
        "units": {
            "gap": "à confirmer",
            "surface_width": "à confirmer",
            "surface_length": "à confirmer",
            "epsilon_r": "sans unité",
            "FREQUENCY": "GHz",
            "s11": "dB",
        },
        "rows": int(len(raw)),
        "effective_rows": int(len(data)),
        "metrics": metrics,
        "validation": {
            "strategy": "5-fold GroupKFold by complete antenna geometry",
            "external_validation": False,
            "folds": fold_metrics,
            "curve_metrics": curve_metrics,
        },
        "scientific_status": "internal_grouped_validation_only",
        "antenna_scope": antenna_scope(args),
        "training_ranges": ranges,
        "geometry_domain": geometry_domain(data, ranges),
        "data_quality": audit,
        "preprocessing": {
            "input_duplicates": "grouped",
            "conflicting_targets": "mean",
            "scaling": "none (tree-based model)",
        },
        "model_parameters": MODEL_PARAMETERS,
        "library_versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "artifacts": {
            "dataset_sha256": dataset_hash,
            "model_sha256": model_hash,
        },
    }

    temporary_info = INFO_PATH.with_suffix(".json.tmp")
    temporary_info.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_info, INFO_PATH)

    print("===== VALIDATION GROUPEE DU MODELE =====")
    print(f"MAE  : {metrics['mae']:.6f} dB")
    print(f"RMSE : {metrics['rmse']:.6f} dB")
    print(f"R2   : {metrics['r2']:.6f}")
    print(
        "Erreur absolue P95 : "
        f"{metrics['absolute_error_p95_db']:.6f} dB"
    )
    print(f"Géométries distinctes : {audit['geometry_count']}")
    print(f"Modèle sauvegardé : {MODEL_PATH}")
    print(f"Métadonnées : {INFO_PATH}")
    if not metadata["antenna_scope"]["verified"]:
        print(
            "ATTENTION : la famille d'antenne du dataset n'est pas encore "
            "scientifiquement vérifiée."
        )


if __name__ == "__main__":
    main()
