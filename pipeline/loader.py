"""
Load and clean josaa_ranks.csv into a pandas DataFrame ready for training.

Key transforms:
  - Infer Exam Type from institute name (IIT → advanced, rest → mains)
  - Cast ranks to int (drop rows where rank is non-numeric / 'P' for PwD rank)
  - All rounds are kept; filtering to last-round-only is no longer done here.
"""

import pandas as pd
from .config import (
    COL_YEAR, COL_ROUND, COL_INSTITUTE, COL_PROGRAM,
    COL_QUOTA, COL_SEAT_TYPE, COL_GENDER,
    COL_OPEN_RANK, COL_CLOSE_RANK, COL_EXAM_TYPE,
    IIT_KEYWORDS, CSAB_QUOTA_NORM,
)


def infer_exam_type(institute_name: str) -> str:
    name = institute_name.lower()
    return "advanced" if any(kw in name for kw in IIT_KEYWORDS) else "mains"


def load(csv_path: str, round_col: str | None = None) -> pd.DataFrame:
    """
    round_col: override the round column name (use "Special Round" for CSAB).
               If None, auto-detects: uses "Round" if present, else "Special Round".
    """
    df = pd.read_csv(csv_path, dtype=str)

    # Standardise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # Auto-detect round column if not specified
    if round_col is None:
        round_col = COL_ROUND if COL_ROUND in df.columns else "Special Round"

    # Normalise: rename whatever round column exists to COL_ROUND so the
    # rest of the pipeline always sees the same name.
    if round_col != COL_ROUND and round_col in df.columns:
        df = df.rename(columns={round_col: COL_ROUND})

    # 2016–2017 pre-date the Gender column (female supernumerary seats were
    # introduced in 2018); fill missing Gender with "Gender-Neutral".
    if COL_GENDER in df.columns:
        df[COL_GENDER] = df[COL_GENDER].fillna("Gender-Neutral")
    else:
        df[COL_GENDER] = "Gender-Neutral"

    # CSAB may not have Quota or Seat Type columns — fill with "ALL" if absent
    for col in (COL_QUOTA, COL_SEAT_TYPE):
        if col not in df.columns:
            df[col] = "ALL"

    # Drop rows with missing core fields (Gender excluded — handled above)
    df.dropna(subset=[COL_INSTITUTE, COL_PROGRAM, COL_QUOTA,
                       COL_SEAT_TYPE,
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

    # Normalise CSAB quota names (current-year page uses full strings vs abbreviations)
    df[COL_QUOTA] = df[COL_QUOTA].apply(
        lambda q: CSAB_QUOTA_NORM.get(q.strip().lower(), q)
    )

    # Infer exam type
    df[COL_EXAM_TYPE] = df[COL_INSTITUTE].apply(infer_exam_type)

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
