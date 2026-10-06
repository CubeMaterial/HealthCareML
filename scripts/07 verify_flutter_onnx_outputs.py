from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from common import MODELS_DIR, OUTPUTS_DIR, setup_logging


ONNX_DIR = MODELS_DIR / "onnx"
REPORT_PATH = OUTPUTS_DIR / "model_reports" / "flutter_onnx_reference.json"
FLUTTER_ASSET_PATH = Path(
    "/Users/g0dzer0/Documents/Workspace/Flutter/health_care_app/assets/models/flutter_onnx_reference.json"
)

REFERENCE_CASES: list[dict[str, Any]] = [
    {
        "case_id": "default_app_input",
        "description": "Flutter survey screen default values",
        "input": {
            "age": 45,
            "sexCode": 1,
            "heightCm": 170.0,
            "weightKg": 70.0,
            "sleepDurationCode": 3,
            "sleepQualityCode": 2,
            "smokingStatusCode": 0,
            "smokingAmountCode": None,
            "drinkingFrequencyCode": 0,
            "drinkingAmountCode": 0,
            "bingeDrinkingFrequencyCode": 0,
            "exerciseFrequencyCode": 2,
            "exerciseDurationCode": 2,
            "strengthTrainingFrequencyCode": 1,
            "walkingDurationCode": 2,
            "walkingDaysCode": 2,
            "stressCode": 2,
            "subjectiveHealthCode": 2,
            "breakfastFrequencyCode": 2,
            "vegetableFrequencyCode": 2,
            "sweetFoodFrequencyCode": 1,
            "saltyFoodPreferenceCode": 1,
            "lateNightSnackFrequencyCode": 1,
        },
    },
    {
        "case_id": "higher_risk_pattern",
        "description": "Higher BMI, smoking, drinking, low activity, short sleep, high stress",
        "input": {
            "age": 62,
            "sexCode": 1,
            "heightCm": 168.0,
            "weightKg": 88.0,
            "sleepDurationCode": 1,
            "sleepQualityCode": 4,
            "smokingStatusCode": 3,
            "smokingAmountCode": 2,
            "drinkingFrequencyCode": 4,
            "drinkingAmountCode": 4,
            "bingeDrinkingFrequencyCode": 4,
            "exerciseFrequencyCode": 0,
            "exerciseDurationCode": 0,
            "strengthTrainingFrequencyCode": 0,
            "walkingDurationCode": 0,
            "walkingDaysCode": 0,
            "stressCode": 4,
            "subjectiveHealthCode": 4,
            "breakfastFrequencyCode": 0,
            "vegetableFrequencyCode": 0,
            "sweetFoodFrequencyCode": 3,
            "saltyFoodPreferenceCode": 2,
            "lateNightSnackFrequencyCode": 3,
        },
    },
    {
        "case_id": "lower_risk_pattern",
        "description": "Younger, normal BMI, non-smoking, lower drinking, active, enough sleep",
        "input": {
            "age": 31,
            "sexCode": 2,
            "heightCm": 164.0,
            "weightKg": 55.0,
            "sleepDurationCode": 3,
            "sleepQualityCode": 0,
            "smokingStatusCode": 0,
            "smokingAmountCode": None,
            "drinkingFrequencyCode": 1,
            "drinkingAmountCode": 0,
            "bingeDrinkingFrequencyCode": 0,
            "exerciseFrequencyCode": 3,
            "exerciseDurationCode": 2,
            "strengthTrainingFrequencyCode": 2,
            "walkingDurationCode": 2,
            "walkingDaysCode": 3,
            "stressCode": 0,
            "subjectiveHealthCode": 0,
            "breakfastFrequencyCode": 3,
            "vegetableFrequencyCode": 3,
            "sweetFoodFrequencyCode": 0,
            "saltyFoodPreferenceCode": 0,
            "lateNightSnackFrequencyCode": 0,
        },
    },
]


def bmi(case_input: dict[str, Any]) -> float:
    meter = float(case_input["heightCm"]) / 100
    return float(case_input["weightKg"]) / (meter * meter)


def model_sleep_avg_hours(case_input: dict[str, Any]) -> float:
    return {
        0: 4.5,
        1: 5.5,
        2: 6.5,
        3: 7.5,
        4: 8.5,
    }.get(int(case_input["sleepDurationCode"]), 8.5)


def model_smoking_code(case_input: dict[str, Any]) -> float:
    return 2.0 if int(case_input["smokingStatusCode"]) <= 1 else 1.0


def model_drinking_code(case_input: dict[str, Any]) -> float:
    drinking_risk_score = (
        int(case_input["drinkingFrequencyCode"])
        + int(case_input["drinkingAmountCode"])
        + int(case_input["bingeDrinkingFrequencyCode"])
    )
    if drinking_risk_score >= 8:
        return 1.0
    if drinking_risk_score >= 4:
        return 2.0
    return 3.0


def model_physical_activity_code(case_input: dict[str, Any]) -> float:
    activity_score = (
        int(case_input["exerciseFrequencyCode"])
        + int(case_input["exerciseDurationCode"])
        + int(case_input["strengthTrainingFrequencyCode"])
    )
    if activity_score <= 2:
        return 1.0
    if activity_score <= 5:
        return 2.0
    return 3.0


def model_walking_code(case_input: dict[str, Any]) -> float:
    walking_score = int(case_input["walkingDurationCode"]) + int(case_input["walkingDaysCode"])
    if walking_score <= 1:
        return 1.0
    if walking_score <= 3:
        return 2.0
    return 3.0


def common_features(case_input: dict[str, Any]) -> dict[str, float]:
    return {
        "age": float(case_input["age"]),
        "sex": float(case_input["sexCode"]),
        "bmi": bmi(case_input),
        "smoking": model_smoking_code(case_input),
        "drinking": model_drinking_code(case_input),
        "physical_activity": model_physical_activity_code(case_input),
        "walking": model_walking_code(case_input),
        "sleep_avg": model_sleep_avg_hours(case_input),
        "stress": float(case_input["stressCode"]) + 1,
        "subjective_health": float(case_input["subjectiveHealthCode"]) + 1,
    }


def obesity_features(case_input: dict[str, Any]) -> dict[str, float]:
    features = common_features(case_input)
    features.pop("bmi")
    return features


def feed_for(session: ort.InferenceSession, features: dict[str, float]) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    for input_meta in session.get_inputs():
        name = input_meta.name
        if name not in features:
            raise KeyError(f"Missing ONNX input {name}")
        feed[name] = np.array([[features[name]]], dtype=np.float32)
    return feed


def predict_percent(model_name: str, case_input: dict[str, Any]) -> float:
    session = ort.InferenceSession(str(ONNX_DIR / f"{model_name}.onnx"), providers=["CPUExecutionProvider"])
    features = obesity_features(case_input) if model_name == "obesity" else common_features(case_input)
    outputs = session.run(None, feed_for(session, features))
    probability = float(outputs[1][0][1])
    return round(probability * 100, 4)


def risk_factors(case_input: dict[str, Any]) -> list[str]:
    factors = []
    if bmi(case_input) >= 25:
        factors.append("BMI 높음")
    if case_input["exerciseFrequencyCode"] == 0 or case_input["exerciseDurationCode"] == 0:
        factors.append("운동 부족")
    if model_sleep_avg_hours(case_input) < 7 or case_input["sleepQualityCode"] >= 3:
        factors.append("수면 부족/질 저하")
    if case_input["smokingStatusCode"] >= 2:
        factors.append("흡연")
    if (
        case_input["drinkingFrequencyCode"] >= 3
        or case_input["drinkingAmountCode"] >= 3
        or case_input["bingeDrinkingFrequencyCode"] >= 3
    ):
        factors.append("음주 빈도/음주량 높음")
    if case_input["stressCode"] >= 3:
        factors.append("스트레스 높음")
    if case_input["vegetableFrequencyCode"] <= 1:
        factors.append("채소 섭취 부족")
    if case_input["sweetFoodFrequencyCode"] >= 2:
        factors.append("단 음식/음료 섭취 잦음")
    if case_input["saltyFoodPreferenceCode"] >= 2:
        factors.append("짠 음식 선호")
    if case_input["lateNightSnackFrequencyCode"] >= 2:
        factors.append("야식 빈도 높음")
    return factors or ["뚜렷한 주요 위험 요인 없음"]


def build_report() -> dict[str, Any]:
    cases = []
    for case in REFERENCE_CASES:
        case_input = case["input"]
        cases.append(
            {
                **case,
                "derived": {"bmi": round(bmi(case_input), 4)},
                "expected": {
                    "diabetesPercent": predict_percent("diabetes", case_input),
                    "hypertensionPercent": predict_percent("hypertension", case_input),
                    "obesityPercent": predict_percent("obesity", case_input),
                    "riskFactors": risk_factors(case_input),
                },
            }
        )

    return {
        "purpose": "Reference outputs for comparing Flutter ONNX inference with Python ONNX Runtime.",
        "tolerance_percent_points": 0.01,
        "models": {
            "diabetes": "assets/models/diabetes.onnx",
            "hypertension": "assets/models/hypertension.onnx",
            "obesity": "assets/models/obesity.onnx",
        },
        "cases": cases,
    }


def main() -> None:
    setup_logging()
    report = build_report()
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Flutter ONNX 기준값 저장: %s", REPORT_PATH)

    if FLUTTER_ASSET_PATH.parent.exists():
        shutil.copy2(REPORT_PATH, FLUTTER_ASSET_PATH)
        logging.info("Flutter assets 기준값 복사: %s", FLUTTER_ASSET_PATH)


if __name__ == "__main__":
    main()
