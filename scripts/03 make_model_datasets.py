from __future__ import annotations

import logging

import pandas as pd

from common import ML_DIR, load_config, read_data, setup_logging


def build_model_dataset(common_df: pd.DataFrame, model_name: str, spec: dict) -> pd.DataFrame:
    label = spec["label"]
    features = spec["features"]
    columns = features + [label]

    missing = [col for col in columns if col not in common_df.columns]
    if missing:
        raise KeyError(f"{model_name} 데이터셋 생성 실패. 누락 컬럼: {missing}")

    # label leakage 방지: 명시 feature만 남기고 진단/치료 원천 컬럼은 포함하지 않습니다.
    dataset = common_df[columns].copy()
    dataset = dataset.dropna(subset=[label])
    dataset[label] = dataset[label].astype(int)
    return dataset


def main() -> None:
    setup_logging()
    config = load_config()
    common_path = ML_DIR / "risk_common_2020_2025.csv"
    common_df = read_data(common_path)

    for model_name, spec in config["model_datasets"].items():
        dataset = build_model_dataset(common_df, model_name, spec)
        output_path = ML_DIR / f"{model_name}_dataset.csv"
        dataset.to_csv(output_path, index=False, encoding="utf-8-sig")
        logging.info("%s 데이터셋 저장: %s rows=%s", model_name, output_path, len(dataset))


if __name__ == "__main__":
    main()
