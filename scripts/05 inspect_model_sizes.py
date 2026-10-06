from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def format_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


def main() -> None:
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        print(f"{path.name}: {format_size(path.stat().st_size)}")


if __name__ == "__main__":
    main()
