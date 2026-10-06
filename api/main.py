from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy import sparse
from sklearn.exceptions import InconsistentVersionWarning


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "variable_mapping.yaml"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "outputs" / "model_reports"
POSITIVE_CLASS = 1
DEFAULT_ALGORITHM = "random_forest"
SUPPORTED_ALGORITHMS = ("random_forest", "xgboost", "lightgbm")


class SurveyInput(BaseModel):
    age: float | None = Field(default=None, examples=[45])
    sex: float | None = Field(default=None, examples=[1])
    bmi: float | None = Field(default=None, examples=[24.3])
    smoking: float | None = Field(default=None, examples=[2])
    drinking: float | None = Field(default=None, examples=[1])
    physical_activity: float | None = Field(default=None, examples=[3])
    walking: float | None = Field(default=None, examples=[4])
    sleep_avg: float | None = Field(default=None, examples=[7])
    stress: float | None = Field(default=None, examples=[2])
    subjective_health: float | None = Field(default=None, examples=[3])


class ModelPrediction(BaseModel):
    model_name: str
    algorithm: str
    prediction: int
    risk_probability: float
    risk_percent: float
    missing_features: list[str]
    used_features: list[str]


class AllPredictionsResponse(BaseModel):
    results: dict[str, ModelPrediction]


class RecommendationItem(BaseModel):
    title: str
    description: str
    changes: dict[str, float]
    expected_risk_probability: float
    expected_risk_percent: float
    improvement_percent_points: float


class RecommendationResponse(BaseModel):
    model_name: str
    baseline: ModelPrediction
    recommendations: list[RecommendationItem]
    combined_recommendation: RecommendationItem | None
    note: str


class AllRecommendationsResponse(BaseModel):
    results: dict[str, RecommendationResponse]


app = FastAPI(
    title="HealthCareNote Risk Prediction API",
    description="Survey-response inference API for obesity, diabetes, and hypertension risk models.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_specs() -> dict[str, dict[str, Any]]:
    return load_config()["model_datasets"]


@lru_cache(maxsize=1)
def load_model_comparison() -> pd.DataFrame | None:
    report_path = REPORTS_DIR / "model_comparison.csv"
    if not report_path.exists():
        return None
    return pd.read_csv(report_path)


def resolve_algorithm(model_name: str, algorithm: str | None = None) -> str:
    if algorithm:
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported algorithm: {algorithm}. Choose one of {', '.join(SUPPORTED_ALGORITHMS)}.",
            )
        return algorithm

    comparison = load_model_comparison()
    if comparison is not None and not comparison.empty:
        candidates = comparison[comparison["model_name"] == model_name].copy()
        if not candidates.empty:
            candidates = candidates[
                candidates["algorithm"].map(lambda value: (MODELS_DIR / f"{model_name}_{value}.joblib").exists())
            ]
        if not candidates.empty:
            candidates["roc_auc"] = pd.to_numeric(candidates["roc_auc"], errors="coerce")
            candidates["f1"] = pd.to_numeric(candidates["f1"], errors="coerce")
            best = candidates.sort_values(["roc_auc", "f1"], ascending=False).iloc[0]
            return str(best["algorithm"])

    return DEFAULT_ALGORITHM


@lru_cache(maxsize=None)
def load_pipeline(model_name: str, algorithm: str) -> Any:
    if model_name not in model_specs():
        raise KeyError(model_name)

    model_path = MODELS_DIR / f"{model_name}_{algorithm}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
        return joblib.load(model_path)


def value_for_feature(payload: dict[str, Any], feature: str) -> float:
    value = payload.get(feature)
    if value is None:
        return np.nan
    return float(value)


def transform_without_pandas(pipeline: Any, payload: dict[str, Any]) -> Any:
    preprocessor = pipeline.named_steps["preprocess"]
    transformed_blocks = []

    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name == "remainder" or transformer == "drop":
            continue

        row = [[value_for_feature(payload, column) for column in columns]]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            if transformer_name == "numeric":
                block = transformer.named_steps["imputer"].transform(row)
            elif transformer_name == "categorical":
                imputed = transformer.named_steps["imputer"].transform(row)
                block = transformer.named_steps["onehot"].transform(imputed)
            else:
                block = transformer.transform(row)

        transformed_blocks.append(block)

    if not transformed_blocks:
        raise ValueError("No model input columns were transformed.")

    if any(sparse.issparse(block) for block in transformed_blocks):
        blocks = [
            block if sparse.issparse(block) else sparse.csr_matrix(block)
            for block in transformed_blocks
        ]
        return sparse.hstack(blocks, format="csr")

    return np.hstack(transformed_blocks)


def predict_model(model_name: str, survey: SurveyInput, algorithm: str | None = None) -> ModelPrediction:
    spec = model_specs().get(model_name)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    selected_algorithm = resolve_algorithm(model_name, algorithm)
    try:
        pipeline = load_pipeline(model_name, selected_algorithm)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model file not found: {exc}") from exc

    payload = survey.model_dump()
    used_features = list(spec["features"])
    missing_features = [feature for feature in used_features if payload.get(feature) is None]
    model_input = transform_without_pandas(pipeline, payload)

    model = pipeline.named_steps["model"]
    probabilities = model.predict_proba(model_input)[0]
    classes = list(model.classes_)
    positive_index = classes.index(POSITIVE_CLASS)
    risk_probability = float(probabilities[positive_index])
    prediction = int(model.predict(model_input)[0])

    return ModelPrediction(
        model_name=model_name,
        algorithm=selected_algorithm,
        prediction=prediction,
        risk_probability=risk_probability,
        risk_percent=round(risk_probability * 100, 2),
        missing_features=missing_features,
        used_features=used_features,
    )


def categorical_values(pipeline: Any) -> dict[str, list[float]]:
    preprocessor = pipeline.named_steps["preprocess"]
    values: dict[str, list[float]] = {}

    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer_name != "categorical":
            continue
        encoder = transformer.named_steps["onehot"]
        for column, categories in zip(columns, encoder.categories_):
            values[column] = [float(category) for category in categories]

    return values


def risk_for_payload(model_name: str, payload: dict[str, Any], algorithm: str | None = None) -> float:
    prediction = predict_model(model_name, SurveyInput(**payload), algorithm=algorithm)
    return prediction.risk_probability


def numeric_candidate_changes(payload: dict[str, Any], used_features: list[str]) -> list[tuple[str, float, str]]:
    changes: list[tuple[str, float, str]] = []

    bmi = payload.get("bmi")
    if "bmi" in used_features and bmi is not None:
        bmi_value = float(bmi)
        targets = [round(bmi_value - 1, 1), round(bmi_value - 3, 1), 25.0, 23.0]
        for target in sorted({target for target in targets if 18.5 <= target < bmi_value}, reverse=True):
            changes.append(("bmi", target, f"BMI를 {target:g}까지 낮추는 시나리오"))

    sleep_avg = payload.get("sleep_avg")
    if "sleep_avg" in used_features and sleep_avg is not None:
        sleep_value = float(sleep_avg)
        for target in [7.0, 8.0]:
            if abs(target - sleep_value) >= 0.5:
                changes.append(("sleep_avg", target, f"평균 수면시간을 {target:g}시간으로 맞추는 시나리오"))

    return changes


def recommendation_title(feature: str, value: float) -> str:
    titles = {
        "bmi": "체중 관리",
        "sleep_avg": "수면 시간 조정",
        "smoking": "흡연 습관 개선",
        "drinking": "음주 습관 개선",
        "physical_activity": "운동 빈도 개선",
        "walking": "걷기 활동 개선",
        "stress": "스트레스 관리",
    }
    return titles.get(feature, f"{feature} 개선")


def recommendation_description(feature: str, current: Any, value: float) -> str:
    if feature in {"bmi", "sleep_avg"}:
        return f"{feature} 값을 {current}에서 {value:g}(으)로 바꾸어 재예측했습니다."
    return (
        f"{feature} 응답 코드를 {current}에서 {value:g}(으)로 바꾸어 재예측했습니다. "
        "코드가 의미하는 실제 문항 선택지는 앱 설문 코드북과 연결해 표시하세요."
    )


def build_recommendation(
    model_name: str,
    baseline_risk: float,
    payload: dict[str, Any],
    feature: str,
    value: float,
    algorithm: str | None = None,
) -> RecommendationItem | None:
    changed = dict(payload)
    changed[feature] = value
    expected_risk = risk_for_payload(model_name, changed, algorithm=algorithm)
    improvement = round((baseline_risk - expected_risk) * 100, 2)
    if improvement <= 0:
        return None

    return RecommendationItem(
        title=recommendation_title(feature, value),
        description=recommendation_description(feature, payload.get(feature), value),
        changes={feature: value},
        expected_risk_probability=expected_risk,
        expected_risk_percent=round(expected_risk * 100, 2),
        improvement_percent_points=improvement,
    )


def recommend_model(
    model_name: str,
    survey: SurveyInput,
    limit: int = 5,
    algorithm: str | None = None,
) -> RecommendationResponse:
    spec = model_specs().get(model_name)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    selected_algorithm = resolve_algorithm(model_name, algorithm)
    pipeline = load_pipeline(model_name, selected_algorithm)
    payload = survey.model_dump()
    used_features = list(spec["features"])
    baseline = predict_model(model_name, survey, algorithm=selected_algorithm)
    recommendations: list[RecommendationItem] = []

    numeric_best: dict[str, RecommendationItem] = {}
    for feature, value, _ in numeric_candidate_changes(payload, used_features):
        item = build_recommendation(
            model_name, baseline.risk_probability, payload, feature, value, algorithm=selected_algorithm
        )
        if item:
            previous = numeric_best.get(feature)
            if previous is None or item.improvement_percent_points > previous.improvement_percent_points:
                numeric_best[feature] = item
    recommendations.extend(numeric_best.values())


    category_options = categorical_values(pipeline)
    actionable_categories = ["smoking", "drinking", "physical_activity", "walking", "stress"]
    for feature in actionable_categories:
        if feature not in used_features or payload.get(feature) is None:
            continue
        current = float(payload[feature])
        best_item: RecommendationItem | None = None
        for value in category_options.get(feature, []):
            if value == current:
                continue
            item = build_recommendation(
                model_name, baseline.risk_probability, payload, feature, value, algorithm=selected_algorithm
            )
            if item and (
                best_item is None
                or item.improvement_percent_points > best_item.improvement_percent_points
            ):
                best_item = item
        if best_item:
            recommendations.append(best_item)

    recommendations.sort(key=lambda item: item.improvement_percent_points, reverse=True)
    recommendations = recommendations[:limit]

    combined = None
    if recommendations:
        combined_payload = dict(payload)
        combined_changes: dict[str, float] = {}
        current_risk = baseline.risk_probability
        for item in recommendations:
            trial_payload = dict(combined_payload)
            trial_payload.update(item.changes)
            trial_risk = risk_for_payload(model_name, trial_payload, algorithm=selected_algorithm)
            if trial_risk < current_risk:
                combined_payload = trial_payload
                combined_changes.update(item.changes)
                current_risk = trial_risk
        if combined_changes:
            combined = RecommendationItem(
                title="복합 개선 시나리오",
                description="상위 개선 제안 중 함께 적용해도 위험도가 낮아지는 항목만 조합한 재예측 결과입니다.",
                changes=combined_changes,
                expected_risk_probability=current_risk,
                expected_risk_percent=round(current_risk * 100, 2),
                improvement_percent_points=round((baseline.risk_probability - current_risk) * 100, 2),
            )

    return RecommendationResponse(
        model_name=model_name,
        baseline=baseline,
        recommendations=recommendations,
        combined_recommendation=combined,
        note=(
            "이 결과는 의학적 처방이 아니라 모델 입력값을 바꿔 재예측한 시나리오입니다. "
            "흡연/음주/운동 같은 범주형 코드의 사용자 문구는 앱 설문 코드북 확정 후 연결해야 합니다."
        ),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "label": spec["label"],
            "features": spec["features"],
            "default_algorithm": resolve_algorithm(name),
            "model_files": {
                algorithm: str(MODELS_DIR / f"{name}_{algorithm}.joblib")
                for algorithm in SUPPORTED_ALGORITHMS
                if (MODELS_DIR / f"{name}_{algorithm}.joblib").exists()
            },
        }
        for name, spec in model_specs().items()
    }


@app.post("/predict/{model_name}", response_model=ModelPrediction)
def predict_one(
    model_name: str,
    survey: SurveyInput,
    algorithm: str | None = Query(default=None, description="random_forest, xgboost, lightgbm"),
) -> ModelPrediction:
    return predict_model(model_name, survey, algorithm=algorithm)


@app.post("/predict", response_model=AllPredictionsResponse)
def predict_all(
    survey: SurveyInput,
    algorithm: str | None = Query(default=None, description="random_forest, xgboost, lightgbm"),
) -> AllPredictionsResponse:
    return AllPredictionsResponse(
        results={name: predict_model(name, survey, algorithm=algorithm) for name in model_specs()}
    )


@app.post("/recommend/{model_name}", response_model=RecommendationResponse)
def recommend_one(
    model_name: str,
    survey: SurveyInput,
    algorithm: str | None = Query(default=None, description="random_forest, xgboost, lightgbm"),
) -> RecommendationResponse:
    return recommend_model(model_name, survey, algorithm=algorithm)


@app.post("/recommend", response_model=AllRecommendationsResponse)
def recommend_all(
    survey: SurveyInput,
    algorithm: str | None = Query(default=None, description="random_forest, xgboost, lightgbm"),
) -> AllRecommendationsResponse:
    return AllRecommendationsResponse(
        results={name: recommend_model(name, survey, algorithm=algorithm) for name in model_specs()}
    )
