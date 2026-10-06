from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "variable_mapping.yaml"
DATA_DIR = ROOT / "Data"
OUTPUTS_DIR = ROOT / "outputs"
PROCESSED_DIR = DATA_DIR / "processed"
ML_DIR = DATA_DIR / "ml"
MODELS_DIR = ROOT / "models"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def decode_bytes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].map(lambda value: isinstance(value, bytes)).any():
            df[col] = df[col].map(
                lambda value: value.decode("utf-8", errors="ignore")
                if isinstance(value, bytes)
                else value
            )
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def read_csv_with_encoding(path: Path, **kwargs: Any) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "cp949", "euc-kr")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"CSV 인코딩 확인 실패: {path}") from last_error


def read_data(path: Path, **kwargs: Any) -> pd.DataFrame:
    suffix = path.suffix.lower()
    logging.info("데이터 읽기: %s", path)
    if suffix == ".sas7bdat":
        try:
            import pyreadstat  # type: ignore

            df, _ = pyreadstat.read_sas7bdat(str(path), **kwargs)
        except ImportError:
            logging.warning("pyreadstat이 없어 pandas.read_sas로 대체합니다: %s", path)
            df = pd.read_sas(path, format="sas7bdat", **kwargs)
        return normalize_columns(decode_bytes(df))
    if suffix == ".csv":
        return normalize_columns(read_csv_with_encoding(path, **kwargs))
    raise ValueError(f"지원하지 않는 파일 형식입니다: {path}")


def count_rows(path: Path) -> int:
    if path.suffix.lower() == ".sas7bdat":
        total = 0
        for chunk in pd.read_sas(path, format="sas7bdat", chunksize=50_000):
            total += len(chunk)
        return total
    if path.suffix.lower() == ".csv":
        return sum(1 for _ in path.open("rb")) - 1
    raise ValueError(f"지원하지 않는 파일 형식입니다: {path}")


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def apply_variable_rules(series: pd.Series, rules: dict[str, Any] | None) -> pd.Series:
    if rules is None:
        return series

    cleaned = to_numeric(series) if rules.get("type") == "numeric" else series.copy()
    invalid_values = rules.get("invalid_values", [])
    if invalid_values:
        cleaned = cleaned.mask(cleaned.isin(invalid_values))

    valid_range = rules.get("valid_range")
    if valid_range:
        cleaned = to_numeric(cleaned)
        low, high = valid_range
        cleaned = cleaned.mask((cleaned < low) | (cleaned > high))
    return cleaned
