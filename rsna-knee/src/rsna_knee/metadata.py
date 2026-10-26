"""Loading helpers for the competition metadata CSVs."""

from pathlib import Path

import pandas as pd

from .constants import LABELS, RAW_DIR


def load_train(csv_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Study-level table: StudyInstanceUID, Report text, 12 label columns.

    Label semantics (verified from raw CSV): only 58 studies carry a complete
    12-label ground truth (explicit 0/1 in every cell). For all other studies
    every label cell is blank = UNLABELED, not negative. We keep NaN for
    unlabeled so downstream code cannot silently treat it as class 0.
    """
    path = Path(csv_dir) / "train.csv"
    df = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    for col in LABELS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # blank -> NaN
    df["labeled"] = df[LABELS].notna().to_numpy().any(axis=1)
    df["n_positives"] = df[LABELS].sum(axis=1, skipna=True).fillna(0).astype("int8")
    return df


def load_train_series(csv_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Series-level table: which series belong to which study + acquisition attrs."""
    path = Path(csv_dir) / "train_series.csv"
    return pd.read_csv(
        path,
        dtype={
            "StudyInstanceUID": str,
            "SeriesInstanceUID": str,
            "Fluid_Sensitive": "int8",
            "Fat_Suppression": "int8",
            "Anatomical_Plane": str,
        },
    )


def load_test(csv_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = Path(csv_dir) / "test.csv"
    return pd.read_csv(path, dtype={"StudyInstanceUID": str})
