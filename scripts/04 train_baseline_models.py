from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "healthcarenote-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "healthcarenote-cache"))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import ML_DIR, MODELS_DIR, OUTPUTS_DIR, load_config, read_data, setup_logging


MODEL_COMPRESSION = ("xz", 3)
RF_N_ESTIMATORS = 80
RF_MAX_DEPTH = 14
RF_MIN_SAMPLES_LEAF = 30
ALGORITHM_LABELS = {
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [col for col in x.columns if x[col].nunique(dropna=True) <= 20 and col != "age"]
    numeric_cols = [col for col in x.columns if col not in categorical_cols]

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric_cols),
            ("categorical", categorical_pipe, categorical_cols),
        ]
    )
    return preprocessor


def make_random_forest(_y_train: pd.Series) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


def make_xgboost(y_train: pd.Series) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost 패키지가 설치되어 있지 않습니다. requirements.txt 설치 후 다시 실행하세요.") from exc

    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())
    scale_pos_weight = negative / positive if positive else 1.0
    return XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )


def make_lightgbm(y_train: pd.Series) -> Any:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError("lightgbm 패키지가 설치되어 있지 않습니다. requirements.txt 설치 후 다시 실행하세요.") from exc

    return LGBMClassifier(
        n_estimators=250,
        max_depth=-1,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary",
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        verbose=-1,
    )


MODEL_BUILDERS: dict[str, Callable[[pd.Series], Any]] = {
    "random_forest": make_random_forest,
    "xgboost": make_xgboost,
    "lightgbm": make_lightgbm,
}


def make_pipeline(x: pd.DataFrame, y_train: pd.Series, algorithm: str) -> Pipeline:
    if algorithm not in MODEL_BUILDERS:
        raise ValueError(f"지원하지 않는 알고리즘입니다: {algorithm}")

    return Pipeline(
        [
            ("preprocess", make_preprocessor(x)),
            ("model", MODEL_BUILDERS[algorithm](y_train)),
        ]
    )


def evaluate(y_true: pd.Series, y_pred: pd.Series, y_prob: pd.Series) -> dict[str, Any]:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4) if y_true.nunique() == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_one(model_name: str, label: str, dataset_path: Path) -> list[dict[str, Any]]:
    logging.info("%s 모델 학습 시작: %s", model_name, dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"학습 데이터셋이 없습니다: {dataset_path}. "
            "먼저 python 'scripts/03 make_model_datasets.py'를 실행해 Data/ml/*.csv를 생성하세요."
        )

    df = read_data(dataset_path)
    df = df.dropna(subset=[label])
    y = df[label].astype(int)
    x = df.drop(columns=[label])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    reports: list[dict[str, Any]] = []
    (OUTPUTS_DIR / "model_reports").mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for algorithm in MODEL_BUILDERS:
        logging.info("%s %s 학습 중", model_name, ALGORITHM_LABELS[algorithm])
        pipeline = make_pipeline(x_train, y_train, algorithm)
        pipeline.fit(x_train, y_train)

        y_pred = pipeline.predict(x_test)
        y_prob = pipeline.predict_proba(x_test)[:, 1]
        report = evaluate(y_test, y_pred, y_prob)
        report.update(
            {
                "model_name": model_name,
                "algorithm": algorithm,
                "algorithm_label": ALGORITHM_LABELS[algorithm],
                "train_rows": int(len(x_train)),
                "test_rows": int(len(x_test)),
                "positive_rate": round(float(y.mean()), 4),
            }
        )

        report_path = OUTPUTS_DIR / "model_reports" / f"{model_name}_{algorithm}_report.json"
        model_path = MODELS_DIR / f"{model_name}_{algorithm}.joblib"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        joblib.dump(pipeline, model_path, compress=MODEL_COMPRESSION)
        reports.append(report)
        logging.info("%s %s 모델 저장: %s", model_name, ALGORITHM_LABELS[algorithm], model_path)
        logging.info("%s %s 리포트 저장: %s", model_name, ALGORITHM_LABELS[algorithm], report_path)

    return reports


def render_metric_chart(summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(len(METRIC_COLUMNS), 1, figsize=(11, 15), sharex=True)
    colors = {"RandomForest": "#64748b", "XGBoost": "#f97316", "LightGBM": "#22c55e"}

    for ax, metric in zip(axes, METRIC_COLUMNS):
        pivot = summary.pivot(index="model_name", columns="algorithm_label", values=metric)
        pivot = pivot[[label for label in ALGORITHM_LABELS.values() if label in pivot.columns]]
        pivot.plot(kind="bar", ax=ax, color=[colors.get(col, "#2563eb") for col in pivot.columns])
        ax.set_title(metric.upper().replace("_", "-"), fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("score")
        ax.grid(axis="y", alpha=0.25)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    axes[-1].tick_params(axis="x", rotation=0)
    fig.suptitle("HealthCareNote Model Performance Comparison", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_metric_html(summary: pd.DataFrame, output_path: Path) -> None:
    display_df = summary[
        ["model_name", "algorithm_label", *METRIC_COLUMNS, "train_rows", "test_rows", "positive_rate"]
    ].copy()
    display_df = display_df.rename(
        columns={
            "model_name": "Disease Model",
            "algorithm_label": "Algorithm",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "roc_auc": "ROC-AUC",
            "train_rows": "Train Rows",
            "test_rows": "Test Rows",
            "positive_rate": "Positive Rate",
        }
    )

    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    styled = (
        display_df.style.format({col: "{:.4f}" for col in metric_cols})
        .format({"Positive Rate": "{:.4f}"})
        .background_gradient(subset=metric_cols, cmap="YlGnBu", vmin=0, vmax=1)
        .set_table_styles(
            [
                {"selector": "caption", "props": "caption-side: top; font-size: 22px; font-weight: 700; margin: 16px;"},
                {"selector": "th", "props": "background: #0f172a; color: white; padding: 10px;"},
                {"selector": "td", "props": "padding: 9px; text-align: center;"},
                {"selector": "table", "props": "border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;"},
            ]
        )
        .set_caption("HealthCareNote Model Performance Dashboard")
        .hide(axis="index")
    )
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>HealthCareNote Model Performance</title>
  <style>
    body {{ margin: 0; padding: 32px; background: #f8fafc; color: #0f172a; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; }}
    h1 {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin-bottom: 6px; }}
    p {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #475569; margin-bottom: 24px; }}
    .panel {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06); }}
    img {{ width: 100%; margin-top: 24px; border: 1px solid #e2e8f0; border-radius: 8px; }}
    table {{ width: 100%; }}
  </style>
</head>
<body>
  <main class="wrap">
    <h1>HealthCareNote Model Performance</h1>
    <p>RandomForest, XGBoost, LightGBM 분류 모델의 Accuracy, Precision, Recall, F1, ROC-AUC 비교 결과입니다.</p>
    <section class="panel">
      {styled.to_html()}
      <img src="model_comparison.png" alt="Metric comparison bar chart">
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def write_summary(reports: list[dict[str, Any]]) -> None:
    if not reports:
        return

    summary = pd.DataFrame(reports)
    summary = summary.sort_values(["model_name", "roc_auc", "f1"], ascending=[True, False, False])
    output_dir = OUTPUTS_DIR / "model_reports"
    csv_path = output_dir / "model_comparison.csv"
    html_path = output_dir / "model_comparison.html"
    chart_path = output_dir / "model_comparison.png"

    summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    render_metric_html(summary, html_path)
    render_metric_chart(summary, chart_path)
    logging.info("모델 비교 CSV 저장: %s", csv_path)
    logging.info("모델 비교 HTML 저장: %s", html_path)
    logging.info("모델 비교 차트 저장: %s", chart_path)


def main() -> None:
    setup_logging()
    config = load_config()
    reports: list[dict[str, Any]] = []
    for model_name, spec in config["model_datasets"].items():
        reports.extend(train_one(model_name, spec["label"], ML_DIR / f"{model_name}_dataset.csv"))
    write_summary(reports)


if __name__ == "__main__":
    main()
