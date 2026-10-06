from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from common import OUTPUTS_DIR, count_rows, load_config, normalize_columns, read_data, setup_logging


def inspect_year(year: int, source_path: Path) -> dict[str, int | str]:
    logging.info("%s년 컬럼 검사 시작: %s", year, source_path)
    if source_path.suffix.lower() == ".sas7bdat":
        sample = next(pd.read_sas(source_path, format="sas7bdat", chunksize=1))
        sample = normalize_columns(sample)
    else:
        sample = read_data(source_path, nrows=1)
    sample = normalize_columns(sample)
    columns = list(sample.columns)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    column_path = OUTPUTS_DIR / f"columns_{year}.csv"
    pd.DataFrame({"year": year, "column": columns}).to_csv(column_path, index=False, encoding="utf-8-sig")

    rows = count_rows(source_path)
    logging.info("%s년: rows=%s, columns=%s, 저장=%s", year, rows, len(columns), column_path)
    return {"year": year, "source": str(source_path), "rows": rows, "columns": len(columns)}


def main() -> None:
    setup_logging()
    config = load_config()
    rows = []
    for year in config["project"]["years"]:
        source_path = Path(config["sources"][year]["raw"])
        if not source_path.exists():
            logging.error("%s년 원자료 파일이 없습니다: %s", year, source_path)
            continue
        rows.append(inspect_year(year, source_path))

    summary_path = OUTPUTS_DIR / "columns_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    logging.info("컬럼 검사 요약 저장: %s", summary_path)


if __name__ == "__main__":
    main()
