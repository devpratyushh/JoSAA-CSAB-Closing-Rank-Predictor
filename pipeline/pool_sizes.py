"""
JEE pool sizes by year, used for rank-percentile normalisation.

  "Pool size" = approximate total number of rank-holders for that exam and year.
  Dividing a closing rank by the pool size converts it to a fractional percentile
  (0 = top, 1 = bottom) that is comparable across years even as the applicant pool
  grows or shrinks.

National pool sizes
-------------------
  JEE Advanced: official information bulletins (candidates who appeared and
                obtained a rank).
  JEE Mains:    JoSAA statistics / NTA press releases (unique candidates who
                obtained a final CRL rank, best percentile across sessions).

State-specific pool sizes
-------------------------
  For NIT Home-State (HS) and Other-State (OS) quota slots the relevant pool is
  the number of JEE Mains candidates from that state (HS) or from all other
  states (OS = national - state).  State fractions (_STATE_FRACTION) are
  approximate averages derived from NTA state-wise candidate statistics (2019-2022).
"""

# --------------------------------------------------------------------------- #
# National pool sizes                                                          #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# State-specific pool sizes                                                    #
# --------------------------------------------------------------------------- #

# Approximate fraction of the national JEE Mains pool from each state.
# Derived from NTA state-wise candidate statistics (2019-2022 average).
# Fractions are relatively stable year-to-year; only the absolute national
# total changes, which pool_sizes.py already tracks per year.
_STATE_FRACTION: dict[str, float] = {
    "Andhra Pradesh":    0.044,
    "Arunachal Pradesh": 0.002,
    "Assam":             0.008,
    "Bihar":             0.060,
    "Chhattisgarh":      0.013,
    "Delhi":             0.040,
    "Goa":               0.002,
    "Gujarat":           0.046,
    "Haryana":           0.033,
    "Himachal Pradesh":  0.010,
    "Jammu and Kashmir": 0.006,
    "Jharkhand":         0.021,
    "Karnataka":         0.047,
    "Kerala":            0.018,
    "Ladakh":            0.001,
    "Madhya Pradesh":    0.068,
    "Maharashtra":       0.082,
    "Manipur":           0.004,
    "Meghalaya":         0.002,
    "Mizoram":           0.001,
    "Nagaland":          0.002,
    "Odisha":            0.020,
    "Puducherry":        0.002,
    "Punjab":            0.014,
    "Rajasthan":         0.150,
    "Sikkim":            0.001,
    "Tamil Nadu":        0.032,
    "Telangana":         0.065,
    "Tripura":           0.005,
    "Uttar Pradesh":     0.116,
    "Uttarakhand":       0.015,
    "West Bengal":       0.033,
}

# Maps a unique lowercase substring of each NIT's official name to its home state.
# Checked against all 31 NITs appearing in the JoSAA dataset.
_NIT_STATE_MAP: list[tuple[str, str]] = [
    # Named NITs (unique prefix avoids ambiguity)
    ("sardar vallabhbhai",  "Gujarat"),          # SVNIT Surat
    ("motilal nehru",       "Uttar Pradesh"),    # MNNIT Allahabad
    ("maulana azad",        "Madhya Pradesh"),   # MANIT Bhopal
    ("visvesvaraya",        "Maharashtra"),      # VNIT Nagpur
    ("malaviya",            "Rajasthan"),        # MNIT Jaipur
    ("jalandhar",           "Punjab"),           # Dr. B R Ambedkar NIT Jalandhar
    # City / region substrings unique within NIT names
    ("andhra pradesh",      "Andhra Pradesh"),
    ("arunachal",           "Arunachal Pradesh"),
    ("agartala",            "Tripura"),
    ("calicut",             "Kerala"),
    ("durgapur",            "West Bengal"),
    ("hamirpur",            "Himachal Pradesh"),
    ("jamshedpur",          "Jharkhand"),
    ("karnataka",           "Karnataka"),        # NIT Karnataka, Surathkal
    ("kurukshetra",         "Haryana"),
    ("manipur",             "Manipur"),
    ("meghalaya",           "Meghalaya"),
    ("mizoram",             "Mizoram"),
    ("nagaland",            "Nagaland"),
    ("patna",               "Bihar"),
    ("puducherry",          "Puducherry"),
    ("pondicherry",         "Puducherry"),
    ("raipur",              "Chhattisgarh"),
    ("rourkela",            "Odisha"),
    ("sikkim",              "Sikkim"),
    ("silchar",             "Assam"),
    ("srinagar",            "Jammu and Kashmir"),
    ("tiruchirappalli",     "Tamil Nadu"),
    ("uttarakhand",         "Uttarakhand"),
    ("warangal",            "Telangana"),
    ("goa",                 "Goa"),
    ("delhi",               "Delhi"),            # NIT Delhi (only called for NIT slots)
]

# JoSAA quota codes that represent a state-restricted pool
_HS_QUOTA   = "HS"
_OS_QUOTA   = "OS"
# Special state-restricted quotas used in a handful of NITs
_SPECIAL_STATE_QUOTAS: dict[str, str] = {
    "AP": "Andhra Pradesh",
    "GO": "Goa",
    "JK": "Jammu and Kashmir",
    "LA": "Ladakh",
}


def get_nit_state(institute: str) -> str | None:
    """
    Identify the home state of an NIT from its official name.
    Returns None if no match is found (caller should fall back to national pool).
    """
    lower = institute.lower()
    for keyword, state in _NIT_STATE_MAP:
        if keyword in lower:
            return state
    return None


def get_pool_size_for_slot(exam_type: str, quota: str,
                           institute: str, year: int) -> int:
    """
    Return the effective rank-holder pool for normalising a slot's closing rank.

    Logic:
      - JEE Advanced (IITs): national Advanced pool
      - JEE Mains, AI quota: national Mains pool
      - JEE Mains, HS quota: pool of candidates from the NIT's home state
      - JEE Mains, OS quota: national Mains pool minus the home-state pool
      - JEE Mains, AP/GO/JK/LA: pool of candidates from that specific state
      - Fallback (unknown state): national Mains pool
    """
    national = get_pool_size(exam_type, year)

    if exam_type == "advanced":
        return national

    # Special state-restricted quotas (AP, GO, JK, LA)
    if quota in _SPECIAL_STATE_QUOTAS:
        state = _SPECIAL_STATE_QUOTAS[quota]
        frac  = _STATE_FRACTION.get(state, 0.0)
        return max(1, int(round(frac * national)))

    if quota not in (_HS_QUOTA, _OS_QUOTA):
        return national  # AI quota or anything else: national pool

    state = get_nit_state(institute)
    if state is None:
        return national  # unrecognised institute: fall back

    state_pool = max(1, int(round(_STATE_FRACTION.get(state, 0.0) * national)))
    if quota == _HS_QUOTA:
        return state_pool
    else:  # OS
        return max(1, national - state_pool)
