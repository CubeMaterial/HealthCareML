from pathlib import Path

import pandas as pd


DATA_DIR = Path("Data")
OUT_DIR = DATA_DIR / "processed"
CHUNK_SIZE = 50_000


BASE_COLUMNS = {
    "examin_code",
    "examin_year",
    "exmprs_no",
    "age",
    "sex",
    "ctprvn_code",
    "pbhlth_code",
    "spot_no",
    "hshld_code",
    "mbhld_code",
    "dong_ty_code",
    "house_ty_code",
    "signgu_code",
    "kstrata",
    "wt_h",
    "wt_p",
    "mbhld_co",
    "rspns_adult_co",
    "reside_adult_co",
}

# Risk-model inputs and labels for diabetes, hypertension, obesity, sleep,
# mental health, cardiovascular symptom awareness, and lifestyle risks.
RISK_PREFIXES = (
    "dia_",
    "hya_",
    "oba",
    "obb_",
    "mt",
    "cva_",
    "mya_",
    "sma_",
    "smb_",
    "smc_",
    "smd_",
    "smf_",
    "dra_",
    "drb_",
    "dre_",
    "drf_",
    "drg_",
    "pha_",
    "phb_",
    "nua_",
    "nuc_",
    "nue_",
    "qoa_",
    "fma_",
    "fmb_",
    "sra_",
    "ira_",
)


def decode_bytes(df: pd.DataFrame) -> pd.DataFrame:
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        if df[col].map(lambda value: isinstance(value, bytes)).any():
            df[col] = df[col].map(
                lambda value: value.decode("utf-8", errors="ignore")
                if isinstance(value, bytes)
                else value
            )
    return df


def selected_columns(columns: list[str]) -> list[str]:
    original_by_lower = {col.lower(): col for col in columns}
    selected = []
    for lower_name, original_name in original_by_lower.items():
        if lower_name in BASE_COLUMNS or lower_name.startswith(RISK_PREFIXES):
            selected.append(original_name)
    return selected


def extract_file(path: Path) -> dict[str, object]:
    first = next(pd.read_sas(path, format="sas7bdat", chunksize=1))
    usecols = selected_columns(list(first.columns))
    out_path = OUT_DIR / f"{path.stem}_risk.csv"

    row_count = 0
    wrote_header = False
    for chunk in pd.read_sas(path, format="sas7bdat", chunksize=CHUNK_SIZE):
        chunk = chunk[usecols]
        chunk = decode_bytes(chunk)
        chunk.columns = [col.lower() for col in chunk.columns]
        chunk.to_csv(
            out_path,
            mode="w" if not wrote_header else "a",
            index=False,
            header=not wrote_header,
            encoding="utf-8-sig",
        )
        wrote_header = True
        row_count += len(chunk)

    return {
        "source_file": path.name,
        "output_file": str(out_path),
        "rows": row_count,
        "columns": len(usecols),
        "column_names": ",".join(col.lower() for col in usecols),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in sorted(DATA_DIR.glob("chs*.sas7bdat")):
        print(f"Extracting {path.name}...")
        summaries.append(extract_file(path))

    summary_path = OUT_DIR / "risk_extract_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
