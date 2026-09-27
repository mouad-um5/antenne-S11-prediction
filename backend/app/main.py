from contextlib import asynccontextmanager
import json
import logging
import re
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict, Field

from app.catalog import get_antenna_catalog, validate_antenna_selection
from app.curve_analysis import calculate_threshold_bands
from app.predictor import (
    get_domain_assessment,
    get_model_info,
    get_quality_summary,
    predict_s11,
    predict_sweep,
    validate_model_artifacts,
)
from app.settings import get_settings


LOGGER = logging.getLogger("antenna_api")
SETTINGS = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    catalog = get_antenna_catalog()
    artifacts = validate_model_artifacts()
    LOGGER.info(
        "startup_ready catalog_families=%s model_id=%s scope_verified=%s",
        catalog["family_count"],
        artifacts["model_id"],
        artifacts["scope_verified"],
    )
    yield


app = FastAPI(
    title="Antenne S11 Predictor API",
    description=(
        "API de prédiction S11 avec traçabilité du modèle et contrôle du domaine."
    ),
    version="2.0.0",
    docs_url="/docs" if SETTINGS.enable_docs else None,
    redoc_url="/redoc" if SETTINGS.enable_docs else None,
    openapi_url="/openapi.json" if SETTINGS.enable_docs else None,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=list(SETTINGS.allowed_hosts),
)


@app.middleware("http")
async def operational_headers(request: Request, call_next):
    supplied_request_id = request.headers.get("X-Request-ID", "")
    request_id = (
        supplied_request_id
        if re.fullmatch(r"[A-Za-z0-9._-]{1,128}", supplied_request_id)
        else str(uuid.uuid4())
    )
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        LOGGER.exception(
            "request_failed request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    LOGGER.info(
        json.dumps(
            {
                "event": "request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
            }
        )
    )
    return response


class StrictInput(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        str_strip_whitespace=True,
    )


class AntennaInput(StrictInput):
    antenna_family_id: str = Field(..., min_length=1, max_length=100)
    antenna_variant: str = Field(..., min_length=1, max_length=120)


class PredictionInput(AntennaInput):
    gap: float = Field(..., gt=0, description="Gap")
    surface_width: float = Field(..., gt=0, description="Largeur du substrat")
    surface_length: float = Field(..., gt=0, description="Longueur du substrat")
    epsilon_r: float = Field(..., gt=0, description="Permittivité relative")
    frequency: float = Field(..., gt=0, description="Fréquence en GHz")


class SweepInput(AntennaInput):
    gap: float = Field(..., gt=0)
    surface_width: float = Field(..., gt=0)
    surface_length: float = Field(..., gt=0)
    epsilon_r: float = Field(..., gt=0)
    start_frequency: float = Field(0.5, gt=0)
    end_frequency: float = Field(4.0, gt=0)
    points: int = Field(301, ge=10, le=2001)
    threshold_db: float = Field(-10.0, ge=-100, le=0)


class AntennaContext(BaseModel):
    family_id: str
    family: str
    variant: str


class PredictionResponse(BaseModel):
    antenna: AntennaContext
    s11_db: float
    is_extrapolation: bool
    known_training_geometry: bool | None
    nearest_geometry_distance: float | None
    warnings: list[str]
    model_id: str | None
    scientific_status: str
    model_scope_verified: bool
    estimated_absolute_error_p95_db: float | None


class CurvePoint(BaseModel):
    frequency: float
    s11: float


class FrequencyBand(BaseModel):
    start_frequency: float
    end_frequency: float
    bandwidth: float
    center_frequency: float


class SweepResponse(BaseModel):
    antenna: AntennaContext
    curve: list[CurvePoint]
    minimum: CurvePoint
    threshold_db: float
    bands: list[FrequencyBand]
    total_bandwidth: float
    is_extrapolation: bool
    known_training_geometry: bool | None
    nearest_geometry_distance: float | None
    warnings: list[str]
    model_id: str | None
    scientific_status: str
    model_scope_verified: bool
    estimated_absolute_error_p95_db: float | None


def _selection_context(
    family_id: str,
    variant: str,
) -> tuple[dict, list[str]]:
    try:
        selection = validate_antenna_selection(family_id, variant)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    scope = get_model_info().get("antenna_scope", {})
    scope_verified = bool(scope.get("verified"))
    if scope_verified:
        supported = (
            scope.get("family_id") == family_id
            and scope.get("variant") == variant
        )
        if not supported:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Le modèle chargé n'est pas validé pour cette antenne. "
                    f"Antenne supportée : {scope.get('family')} / "
                    f"{scope.get('variant')}."
                ),
            )
        return selection, []

    if SETTINGS.require_verified_model_scope:
        raise HTTPException(
            status_code=503,
            detail=(
                "Le modèle n'est associé à aucune famille d'antenne vérifiée. "
                "Réentraînez-le avec --antenna-family-id, --antenna-variant "
                "et --scope-verified."
            ),
        )

    return selection, [
        "La famille sélectionnée n'est pas encore scientifiquement associée "
        "au modèle chargé. Cette estimation doit être confirmée par CST ou mesure."
    ]


def _domain_values(payload: PredictionInput | SweepInput) -> dict:
    frequency: float | tuple[float, float]
    if isinstance(payload, SweepInput):
        frequency = (payload.start_frequency, payload.end_frequency)
    else:
        frequency = payload.frequency
    return {
        "gap": payload.gap,
        "surface_width": payload.surface_width,
        "surface_length": payload.surface_length,
        "epsilon_r": payload.epsilon_r,
        "FREQUENCY": frequency,
    }


def _response_context(
    payload: PredictionInput | SweepInput,
) -> tuple[dict, dict, list[str], dict]:
    selection, scope_warnings = _selection_context(
        payload.antenna_family_id,
        payload.antenna_variant,
    )
    domain = get_domain_assessment(_domain_values(payload))
    warnings = list(dict.fromkeys(scope_warnings + domain["warnings"]))
    return selection, domain, warnings, get_quality_summary()


@app.get("/")
def home():
    return {
        "service": "Antenne S11 Predictor API",
        "version": app.version,
        "environment": SETTINGS.environment,
        "documentation": app.docs_url,
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "version": app.version}


@app.get("/api/ready")
def readiness():
    try:
        artifacts = validate_model_artifacts()
        get_antenna_catalog()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Le modèle ou le catalogue est indisponible",
        ) from exc

    if (
        SETTINGS.require_verified_model_scope
        and not artifacts["scope_verified"]
    ):
        raise HTTPException(
            status_code=503,
            detail="Le périmètre scientifique du modèle n'est pas vérifié",
        )
    return {"status": "ready", **artifacts}


@app.get("/api/antennas")
def antenna_catalog():
    return get_antenna_catalog()


@app.get("/api/model-info")
def model_info():
    return get_model_info()


@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: PredictionInput):
    selection, domain, warnings, quality = _response_context(payload)
    try:
        value = predict_s11(
            gap=payload.gap,
            surface_width=payload.surface_width,
            surface_length=payload.surface_length,
            epsilon_r=payload.epsilon_r,
            frequency=payload.frequency,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "antenna": selection,
        "s11_db": value,
        "is_extrapolation": domain["is_out_of_domain"],
        "known_training_geometry": domain["known_training_geometry"],
        "nearest_geometry_distance": domain["nearest_geometry_distance"],
        "warnings": warnings,
        **quality,
    }


@app.post("/api/predict-sweep", response_model=SweepResponse)
def sweep(payload: SweepInput):
    if payload.end_frequency <= payload.start_frequency:
        raise HTTPException(
            status_code=422,
            detail="end_frequency doit être supérieure à start_frequency",
        )

    selection, domain, warnings, quality = _response_context(payload)
    try:
        curve = predict_sweep(
            gap=payload.gap,
            surface_width=payload.surface_width,
            surface_length=payload.surface_length,
            epsilon_r=payload.epsilon_r,
            start_frequency=payload.start_frequency,
            end_frequency=payload.end_frequency,
            points=payload.points,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    minimum = min(curve, key=lambda point: point["s11"])
    bands = calculate_threshold_bands(curve, payload.threshold_db)
    return {
        "antenna": selection,
        "curve": curve,
        "minimum": minimum,
        "threshold_db": payload.threshold_db,
        "bands": bands,
        "total_bandwidth": float(sum(band["bandwidth"] for band in bands)),
        "is_extrapolation": domain["is_out_of_domain"],
        "known_training_geometry": domain["known_training_geometry"],
        "nearest_geometry_distance": domain["nearest_geometry_distance"],
        "warnings": warnings,
        **quality,
    }
