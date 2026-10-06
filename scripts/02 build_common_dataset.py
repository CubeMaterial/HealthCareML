from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import ML_DIR, OUTPUTS_DIR, PROCESSED_DIR, apply_variable_rules, load_config, read_data, setup_logging, to_numeric


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows_"
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def source_for_year(config: dict[str, Any], year: int) -> Path:
    extracted = Path(config["sources"][year]["extracted"])
    if extracted.exists():
        return extracted
    return Path(config["sources"][year]["raw"])


def mapped_source_name(config: dict[str, Any], common_name: str, year: int) -> str | None:
    rules = config["variables"].get(common_name, {})
    value = rules.get(year) or rules.get(str(year))
    return str(value).lower() if value else None


def empty_series(index: pd.Index) -> pd.Series:
    return pd.Series(np.nan, index=index)


def extract_common_column(df: pd.DataFrame, config: dict[str, Any], year: int, common_name: str) -> pd.Series:
    source_name = mapped_source_name(config, common_name, year)
    rules = config["variables"].get(common_name)
    if not source_name:
        logging.warning("%s년 %s 매핑 없음. NaN 생성", year, common_name)
        return empty_series(df.index)
    if source_name not in df.columns:
        logging.warning("%s년 %s 원본 컬럼 없음: %s. NaN 생성", year, common_name, source_name)
        return empty_series(df.index)
    return apply_variable_rules(df[source_name], rules)


def build_year_column(df: pd.DataFrame, config: dict[str, Any], year: int) -> pd.Series:
    source_name = mapped_source_name(config, "year", year)
    if source_name and source_name in df.columns:
        raw = df[source_name].astype(str).str.extract(r"(\d{4})", expand=False)
        parsed = pd.to_numeric(raw, errors="coerce")
        return parsed.fillna(year).astype(int)
    return pd.Series(year, index=df.index, dtype="int64")


def build_bmi(clean: pd.DataFrame, raw_df: pd.DataFrame, config: dict[str, Any], year: int) -> pd.Series:
    height = to_numeric(clean["height"])
    weight = to_numeric(clean["weight"])
    calculated = weight / ((height / 100) ** 2)

    source_name = mapped_source_name(config, "bmi_source", year)
    fallback = empty_series(raw_df.index)
    if source_name and source_name in raw_df.columns:
        fallback = apply_variable_rules(raw_df[source_name], config["variables"].get("bmi_source"))

    bmi = calculated.where(calculated.notna(), fallback)
    return to_numeric(bmi).mask((bmi < 10) | (bmi > 60))


def build_sleep_avg(clean: pd.DataFrame) -> pd.Series:
    weekday = to_numeric(clean["sleep_weekday"]).mask(lambda s: (s < 0) | (s > 24))
    weekend = to_numeric(clean["sleep_weekend"]).mask(lambda s: (s < 0) | (s > 24))
    return pd.concat([weekday, weekend], axis=1).mean(axis=1, skipna=True)


def build_binary_label(raw_df: pd.DataFrame, config: dict[str, Any], label_name: str) -> pd.Series:
    rules = config["labels"][label_name]
    source_name = rules["source_variable"].lower()
    if source_name not in raw_df.columns:
        logging.warning("%s 원본 컬럼 없음. NaN 생성", source_name)
        return empty_series(raw_df.index)

    source = to_numeric(raw_df[source_name])
    label = pd.Series(np.nan, index=raw_df.index)
    label = label.mask(source.isin(rules["positive_values"]), 1)
    label = label.mask(source.isin(rules["negative_values"]), 0)
    label = label.mask(source.isin(rules.get("missing_values", [])))
    return label


def build_clean_year(config: dict[str, Any], year: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_path = source_for_year(config, year)
    logging.info("%s년 clean 데이터 생성: %s", year, source_path)
    raw_df = read_data(source_path)

    clean = pd.DataFrame(index=raw_df.index)
    clean["year"] = build_year_column(raw_df, config, year)

    for name in config["common_columns"]:
        if name in {"year", "bmi", "sleep_avg", "diabetes_label", "hypertension_label", "obesity_label"}:
            continue
        clean[name] = extract_common_column(raw_df, config, year, name)

    clean["bmi"] = build_bmi(clean, raw_df, config, year)
    clean["sleep_avg"] = build_sleep_avg(clean)
    clean["diabetes_label"] = build_binary_label(raw_df, config, "diabetes_label")
    clean["hypertension_label"] = build_binary_label(raw_df, config, "hypertension_label")
    cutoff = config["project"]["obesity_bmi_cutoff"]
    clean["obesity_label"] = pd.Series(np.nan, index=clean.index)
    clean.loc[clean["bmi"].notna(), "obesity_label"] = clean.loc[clean["bmi"].notna(), "bmi"].ge(cutoff).astype(int)

    clean = clean[config["common_columns"]]
    output_path = PROCESSED_DIR / f"chs{year}_clean.csv"
    clean.to_csv(output_path, index=False, encoding="utf-8-sig")

    missing_columns = [col for col in clean.columns if clean[col].isna().all()]
    meta = {
        "year": year,
        "source": str(source_path),
        "output": str(output_path),
        "rows": len(clean),
        "columns": len(clean.columns),
        "missing_columns": ", ".join(missing_columns),
    }
    logging.info("%s년 저장 완료: %s rows=%s", year, output_path, len(clean))
    return clean, meta


def write_quality_report(combined: pd.DataFrame, metas: list[dict[str, Any]], config: dict[str, Any]) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Data Quality Report",
        "",
        "## 연도별 행/컬럼 수",
        markdown_table(pd.DataFrame(metas)[["year", "rows", "columns", "missing_columns"]]),
        "",
        "## 사용된 공통 변수",
        ", ".join(config["common_columns"]),
        "",
        "## Label 클래스 분포",
    ]
    for label in ["obesity_label", "diabetes_label", "hypertension_label"]:
        lines.append(f"### {label}")
        counts = combined[label].value_counts(dropna=False).rename_axis(label).reset_index(name="count")
        lines.append(markdown_table(counts))
        lines.append("")

    missing = (
        combined.isna()
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .rename_axis("column")
        .reset_index(name="missing_rate")
    )
    lines.extend(["## 결측률 상위 컬럼", markdown_table(missing), ""])
    lines.extend(
        [
            "## BMI 생성 방식",
            "- 2020은 `oba_bmi`를 사용합니다.",
            "- 2021~2025는 `height`, `weight`가 있으면 직접 계산합니다.",
            "- 2023~2024처럼 `oba_bmi`가 함께 있으면 직접 계산값을 우선하고, 없을 때 `oba_bmi`를 보조로 사용합니다.",
            "- `bmi < 10` 또는 `bmi > 60`은 결측 처리합니다.",
            "",
            "## 수면 변수 누락 연도",
            "- 2023년은 `mtc_17z1`, `mtc_18z1`가 없어 `sleep_avg`가 NaN입니다.",
            "",
            "## Leakage 방지 규칙",
            "- 비만 모델 feature에서 `bmi`, `height`, `weight`를 제외합니다.",
            "- 당뇨 모델 feature에서 `dia_*` 계열을 제외합니다.",
            "- 고혈압 모델 feature에서 `hya_*` 계열을 제외합니다.",
            "- 진단/치료 문항은 label 생성에만 사용합니다.",
        ]
    )
    report_path = OUTPUTS_DIR / "data_quality_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("데이터 품질 리포트 저장: %s", report_path)


def main() -> None:
    setup_logging()
    config = load_config()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ML_DIR.mkdir(parents=True, exist_ok=True)

    clean_frames = []
    metas = []
    for year in config["project"]["years"]:
        clean, meta = build_clean_year(config, year)
        clean_frames.append(clean)
        metas.append(meta)

    combined = pd.concat(clean_frames, ignore_index=True)
    combined_path = ML_DIR / "risk_common_2020_2025.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(metas).to_csv(ML_DIR / "build_common_summary.csv", index=False, encoding="utf-8-sig")
    write_quality_report(combined, metas, config)
    logging.info("통합 데이터 저장: %s rows=%s", combined_path, len(combined))


if __name__ == "__main__":
    main()
