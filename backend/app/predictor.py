from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _artifact_path(environment_name: str, relative_default: str) -> Path:
    configured = os.getenv(environment_name)
    if not configured:
        return ROOT / relative_default
    path = Path(configured)
    return path if path.is_absolute() else ROOT / path


MODEL_PATH = _artifact_path("MODEL_PATH", "models/s11_predictor.pkl")
INFO_PATH = _artifact_path("MODEL_INFO_PATH", "models/model_info.json")

FEATURES = [
    "gap",
    "surface_width",
    "surface_length",
    "epsilon_r",
    "FREQUENCY",
]
GEOMETRY_FEATURES = FEATURES[:-1]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def get_model_info() -> dict:
    if not INFO_PATH.exists():
        return {
            "features": FEATURES,
            "scientific_status": "metadata_missing",
            "antenna_scope": {"verified": False},
        }
    return json.loads(INFO_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Le modèle n'existe pas encore. Exécutez d'abord : "
            "python src/train_model.py"
        )

    expected_hash = (
        get_model_info().get("artifacts", {}).get("model_sha256")
    )
    if expected_hash and _file_sha256(MODEL_PATH) != expected_hash:
        raise RuntimeError(
            "L'empreinte du modèle ne correspond pas aux métadonnées. "
            "L'artefact peut être corrompu ou provenir d'un autre entraînement."
        )

    model = joblib.load(MODEL_PATH)
    configured_features = get_model_info().get("features", FEATURES)
    if configured_features != FEATURES:
        raise RuntimeError(
            "Le schéma de variables du modèle est incompatible avec l'API"
        )
    return model


def validate_model_artifacts() -> dict:
    load_model()
    info = get_model_info()
    return {
        "model_id": info.get("model_id"),
        "scientific_status": info.get("scientific_status", "unknown"),
        "scope_verified": bool(info.get("antenna_scope", {}).get("verified")),
    }


def get_domain_assessment(
    values: dict[str, float | tuple[float, float]],
) -> dict:
    info = get_model_info()
    ranges = info.get("training_ranges", {})
    warnings: list[str] = []

    for feature, value in values.items():
        limits = ranges.get(feature)
        if not limits:
            continue

        candidates = value if isinstance(value, tuple) else (value,)
        outside = any(
            candidate < limits["min"] or candidate > limits["max"]
            for candidate in candidates
        )
        if outside:
            shown_value = (
                f"{value[0]:g} à {value[1]:g}"
                if isinstance(value, tuple)
                else f"{value:g}"
            )
            warnings.append(
                f"{feature} ({shown_value}) est hors de la plage d'entraînement "
                f"[{limits['min']:g}, {limits['max']:g}]."
            )

    width = values.get("surface_width")
    length = values.get("surface_length")
    width_equals_length = info.get("data_quality", {}).get(
        "width_always_equals_length",
        False,
    )
    if (
        width_equals_length
        and isinstance(width, (int, float))
        and isinstance(length, (int, float))
        and not np.isclose(width, length)
    ):
        warnings.append(
            "La largeur et la longueur sont différentes alors qu'elles sont "
            "toujours égales dans les données d'entraînement."
        )

    geometry_domain = info.get("geometry_domain", {})
    geometries = geometry_domain.get("training_geometries", [])
    nearest_distance: float | None = None
    known_geometry: bool | None = None
    if geometries and all(
        isinstance(values.get(feature), (int, float))
        for feature in GEOMETRY_FEATURES
    ):
        distances = []
        for geometry in geometries:
            squared_distance = 0.0
            for feature in GEOMETRY_FEATURES:
                limits = ranges[feature]
                span = max(float(limits["max"]) - float(limits["min"]), 1.0)
                squared_distance += (
                    (float(values[feature]) - float(geometry[feature])) / span
                ) ** 2
            distances.append(float(np.sqrt(squared_distance)))

        nearest_distance = min(distances)
        known_geometry = bool(nearest_distance <= 1e-12)
        warning_threshold = geometry_domain.get("warning_threshold")
        if (
            warning_threshold is not None
            and nearest_distance > float(warning_threshold)
        ):
            warnings.append(
                "Cette combinaison géométrique est éloignée des configurations "
                "utilisées pendant l'entraînement."
            )

    return {
        "warnings": list(dict.fromkeys(warnings)),
        "is_out_of_domain": bool(warnings),
        "known_training_geometry": known_geometry,
        "nearest_geometry_distance": nearest_distance,
    }


def get_extrapolation_warnings(
    values: dict[str, float | tuple[float, float]],
) -> list[str]:
    return get_domain_assessment(values)["warnings"]


def get_quality_summary() -> dict:
    info = get_model_info()
    metrics = info.get("metrics", {})
    return {
        "model_id": info.get("model_id"),
        "scientific_status": info.get("scientific_status", "unknown"),
        "model_scope_verified": bool(
            info.get("antenna_scope", {}).get("verified")
        ),
        "estimated_absolute_error_p95_db": metrics.get(
            "absolute_error_p95_db"
        ),
    }


def predict_s11(
    *,
    gap: float,
    surface_width: float,
    surface_length: float,
    epsilon_r: float,
    frequency: float,
) -> float:
    model = load_model()
    frame = pd.DataFrame(
        [[gap, surface_width, surface_length, epsilon_r, frequency]],
        columns=FEATURES,
    )
    prediction = float(model.predict(frame)[0])
    if not np.isfinite(prediction):
        raise RuntimeError("Le modèle a produit une valeur non finie")
    return prediction


def predict_sweep(
    *,
    gap: float,
    surface_width: float,
    surface_length: float,
    epsilon_r: float,
    start_frequency: float = 0.5,
    end_frequency: float = 4.0,
    points: int = 201,
) -> list[dict]:
    model = load_model()
    frequencies = np.linspace(start_frequency, end_frequency, points)
    frame = pd.DataFrame(
        {
            "gap": gap,
            "surface_width": surface_width,
            "surface_length": surface_length,
            "epsilon_r": epsilon_r,
            "FREQUENCY": frequencies,
        }
    )
    predictions = model.predict(frame)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Le modèle a produit une valeur non finie")
    return [
        {"frequency": float(frequency), "s11": float(s11)}
        for frequency, s11 in zip(frequencies, predictions)
    ]
