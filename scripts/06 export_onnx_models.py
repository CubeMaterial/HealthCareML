from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "healthcarenote-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "healthcarenote-cache"))

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from lightgbm import LGBMClassifier
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from xgboost import XGBClassifier

from common import ML_DIR, MODELS_DIR, OUTPUTS_DIR, load_config, read_data, setup_logging


ONNX_DIR = MODELS_DIR / "onnx"
FLUTTER_MODEL_DIR = Path("/Users/g0dzer0/Documents/Workspace/Flutter/health_care_app/assets/models")
MODEL_COMPARISON_PATH = OUTPUTS_DIR / "model_reports" / "model_comparison.csv"
EXPORT_REPORT_PATH = OUTPUTS_DIR / "model_reports" / "onnx_export_report.json"
ONNX_TARGET_OPSET = {"": 15, "ai.onnx.ml": 3}
VALIDATION_ROWS = 100


def calculate_tree_classifier_output_shapes(operator: Any) -> None:
    row_count = operator.inputs[0].type.shape[0]
    class_count = len(operator.raw_operator.classes_)
    operator.outputs[0].type = Int64TensorType([row_count])
    operator.outputs[1].type = FloatTensorType([row_count, class_count])


def register_converters() -> None:
    update_registered_converter(
        XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_tree_classifier_output_shapes,
        convert_xgboost,
        options={"zipmap": [True, False], "nocl": [True, False]},
    )
    update_registered_converter(
        LGBMClassifier,
        "LightGBMLGBMClassifier",
        calculate_tree_classifier_output_shapes,
        convert_lightgbm,
        options={"zipmap": [True, False], "nocl": [True, False]},
    )


def best_models() -> list[dict[str, str]]:
    comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    comparison["roc_auc"] = pd.to_numeric(comparison["roc_auc"], errors="coerce")
    comparison["f1"] = pd.to_numeric(comparison["f1"], errors="coerce")
    comparison = comparison.sort_values(["model_name", "roc_auc", "f1"], ascending=[True, False, False])
    best = comparison.groupby("model_name", as_index=False).head(1)
    return [
        {"model_name": str(row.model_name), "algorithm": str(row.algorithm)}
        for row in best.itertuples(index=False)
    ]


def initial_types(feature_names: list[str]) -> list[tuple[str, FloatTensorType]]:
    return [(feature, FloatTensorType([None, 1])) for feature in feature_names]


def onnx_feed(x: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        column: x[[column]].to_numpy(dtype=np.float32)
        for column in x.columns
    }


def positive_probability_from_onnx(output: list[Any]) -> np.ndarray:
    probabilities = output[1]
    if isinstance(probabilities, list):
        return np.array([row.get(1, row.get("1")) for row in probabilities], dtype=np.float32)
    return np.asarray(probabilities, dtype=np.float32)[:, 1]


def validate_onnx(pipeline: Any, onnx_path: Path, x: pd.DataFrame) -> dict[str, float]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sklearn_prob = pipeline.predict_proba(x)[:, 1]
    onnx_prob = positive_probability_from_onnx(session.run(None, onnx_feed(x)))
    diff = np.abs(sklearn_prob - onnx_prob)
    return {
        "rows": int(len(x)),
        "max_abs_diff": round(float(diff.max()), 8),
        "mean_abs_diff": round(float(diff.mean()), 8),
    }


def schema_for(model_name: str, algorithm: str, feature_names: list[str]) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "algorithm": algorithm,
        "format": "onnx",
        "inputs": [
            {"name": feature, "dtype": "float32", "shape": [1, 1]}
            for feature in feature_names
        ],
        "outputs": [
            {"name": "label", "description": "Predicted class label"},
            {"name": "probabilities", "description": "Class probability tensor; positive class is index 1"},
        ],
        "positive_class_index": 1,
        "risk_percent": "probabilities[1] * 100",
    }


def export_one(model_name: str, algorithm: str, label: str) -> dict[str, Any]:
    model_path = MODELS_DIR / f"{model_name}_{algorithm}.joblib"
    dataset_path = ML_DIR / f"{model_name}_dataset.csv"
    onnx_path = ONNX_DIR / f"{model_name}.onnx"
    schema_path = ONNX_DIR / f"{model_name}_schema.json"

    logging.info("%s %s ONNX 변환 시작", model_name, algorithm)
    pipeline = joblib.load(model_path)
    df = read_data(dataset_path).dropna(subset=[label])
    x = df.drop(columns=[label])

    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_types(list(x.columns)),
        target_opset=ONNX_TARGET_OPSET,
        options={id(pipeline.named_steps["model"]): {"zipmap": False}},
    )
    onnx.checker.check_model(onnx_model)

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(onnx_model.SerializeToString())
    schema_path.write_text(
        json.dumps(schema_for(model_name, algorithm, list(x.columns)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    validation = validate_onnx(pipeline, onnx_path, x.head(VALIDATION_ROWS))
    logging.info(
        "%s ONNX 검증 완료: max_abs_diff=%s mean_abs_diff=%s",
        model_name,
        validation["max_abs_diff"],
        validation["mean_abs_diff"],
    )
    return {
        "model_name": model_name,
        "algorithm": algorithm,
        "joblib_path": str(model_path),
        "onnx_path": str(onnx_path),
        "schema_path": str(schema_path),
        "size_bytes": onnx_path.stat().st_size,
        "validation": validation,
    }


def copy_to_flutter_assets(exported: list[dict[str, Any]]) -> None:
    if not FLUTTER_MODEL_DIR.exists():
        logging.warning("Flutter assets 경로가 없어 복사를 건너뜁니다: %s", FLUTTER_MODEL_DIR)
        return

    for item in exported:
        model_name = item["model_name"]
        shutil.copy2(item["onnx_path"], FLUTTER_MODEL_DIR / f"{model_name}.onnx")
        shutil.copy2(item["schema_path"], FLUTTER_MODEL_DIR / f"{model_name}_schema.json")
        logging.info("%s ONNX Flutter assets 복사 완료", model_name)


def main() -> None:
    setup_logging()
    register_converters()
    config = load_config()
    exported = []
    for item in best_models():
        model_name = item["model_name"]
        label = config["model_datasets"][model_name]["label"]
        exported.append(export_one(model_name, item["algorithm"], label))

    EXPORT_REPORT_PATH.write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding="utf-8")
    copy_to_flutter_assets(exported)
    logging.info("ONNX export 리포트 저장: %s", EXPORT_REPORT_PATH)


if __name__ == "__main__":
    main()
