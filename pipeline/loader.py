"""
Load and clean josaa_ranks.csv into a pandas DataFrame ready for training.

Key transforms:
  - Infer Exam Type from institute name (IIT → advanced, rest → mains)
  - Cast ranks to int (drop rows where rank is non-numeric / 'P' for PwD rank)
  - Optionally keep only the last round per year (most settled cutoffs)
"""

import pandas as pd
from .config import (
    COL_YEAR, COL_ROUND, COL_INSTITUTE, COL_PROGRAM,
    COL_QUOTA, COL_SEAT_TYPE, COL_GENDER,
    COL_OPEN_RANK, COL_CLOSE_RANK, COL_EXAM_TYPE,
    IIT_KEYWORDS, LAST_ROUND_ONLY,
)


def infer_exam_type(institute_name: str) -> str:
    name = institute_name.lower()
    return "advanced" if any(kw in name for kw in IIT_KEYWORDS) else "mains"


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)

    # Standardise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Drop rows with missing core fields
    df.dropna(subset=[COL_INSTITUTE, COL_PROGRAM, COL_QUOTA,
                       COL_SEAT_TYPE, COL_GENDER,
                       COL_OPEN_RANK, COL_CLOSE_RANK], inplace=True)

    # Cast year and round to int
    df[COL_YEAR]  = pd.to_numeric(df[COL_YEAR],  errors="coerce")
    df[COL_ROUND] = pd.to_numeric(df[COL_ROUND], errors="coerce")
    df.dropna(subset=[COL_YEAR, COL_ROUND], inplace=True)
    df[COL_YEAR]  = df[COL_YEAR].astype(int)
    df[COL_ROUND] = df[COL_ROUND].astype(int)

    # Closing rank: some rows use 'P' prefix for PwD category rank.
    # Strip the P and keep the numeric part; drop anything that won't parse.
    df[COL_CLOSE_RANK] = (
        df[COL_CLOSE_RANK].str.lstrip("P").str.strip()
    )
    df[COL_OPEN_RANK] = (
        df[COL_OPEN_RANK].str.lstrip("P").str.strip()
    )
    df[COL_CLOSE_RANK] = pd.to_numeric(df[COL_CLOSE_RANK], errors="coerce")
    df[COL_OPEN_RANK]  = pd.to_numeric(df[COL_OPEN_RANK],  errors="coerce")
    df.dropna(subset=[COL_CLOSE_RANK], inplace=True)
    df[COL_CLOSE_RANK] = df[COL_CLOSE_RANK].astype(int)
    df[COL_OPEN_RANK]  = df[COL_OPEN_RANK].astype(int)

    # Infer exam type
    df[COL_EXAM_TYPE] = df[COL_INSTITUTE].apply(infer_exam_type)

    # Keep only last round per year (most settled cutoffs)
    if LAST_ROUND_ONLY:
        last_round = df.groupby(COL_YEAR)[COL_ROUND].transform("max")
        df = df[df[COL_ROUND] == last_round].copy()

    df.reset_index(drop=True, inplace=True)
    return df


def summary(df: pd.DataFrame) -> None:
    print(f"Rows          : {len(df):,}")
    print(f"Years         : {sorted(df[COL_YEAR].unique())}")
    print(f"Exam types    : {df[COL_EXAM_TYPE].value_counts().to_dict()}")
    print(f"Quotas        : {sorted(df[COL_QUOTA].unique())}")
    print(f"Seat types    : {sorted(df[COL_SEAT_TYPE].unique())}")
    print(f"Genders       : {sorted(df[COL_GENDER].unique())}")
    print(f"Institutes    : {df[COL_INSTITUTE].nunique()}")
    print(f"Programs      : {df[COL_PROGRAM].nunique()}")
