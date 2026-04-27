"""
JEE pool sizes by year, used for rank-percentile normalisation.

  "Pool size" = approximate total number of rank-holders for that exam and year.
  Dividing a closing rank by the pool size converts it to a fractional percentile
  (0 = top, 1 = bottom) that is comparable across years even as the applicant pool
  grows or shrinks.

Sources
-------
  JEE Advanced: official information bulletins (candidates who appeared and
                obtained a rank).
  JEE Mains:    JoSAA statistics / NTA press releases (unique candidates who
                obtained a final CRL rank, best percentile across sessions).
"""

# JEE Advanced rank-holders (candidates who appeared and received a rank)
_ADVANCED: dict[int, int] = {
    2016: 147678,
    2017: 171012,
    2018: 155158,
    2019: 161319,
    2020: 150838,
    2021: 141699,
    2022: 160038,
    2023: 189744,
    2024: 180372,
    2025: 186584,
}

# JEE Mains Paper-1 (B.E./B.Tech) unique candidates with a final CRL rank
# Post-2019: best-percentile across two sessions; numbers are approximate unique counts.
_MAINS: dict[int, int] = {
    2016: 1007012,
    2017: 1045496,
    2018: 1043739,
    2019:  924893,
    2020:  858273,
    2021:  711840,
    2022:  869010,
    2023: 1034210,
    2024: 1180827,
    2025: 1200000,  # preliminary estimate
}


def get_pool_size(exam_type: str, year: int) -> int:
    """
    Return the approximate total rank-holder count for exam_type and year.
    Falls back to the nearest known year if year is not in the table.
    """
    pool = _ADVANCED if exam_type == "advanced" else _MAINS
    if year in pool:
        return pool[year]
    nearest = min(pool, key=lambda y: abs(y - year))
    return pool[nearest]
