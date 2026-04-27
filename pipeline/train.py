"""
Train the ensemble model that predicts closing rank for every round.

Two signals per slot (institute x program x quota x seat_type x gender):

  1. Year-trend per round
       For each round r, fit LinearRegression(year -> closing_rank).
       Captures how the cutoff for *that specific round* has drifted year-over-year.

  2. Round-progression ratios
       For each year, record ratio[r] = close[r] / close[last_round_in_year].
       Average ratios across years to learn the typical R1->R2->...->Rfinal shape.
       At prediction time: predicted_close[r] = predicted_final_close x ratio[r].

Ensemble prediction for round r in target year Y:
       pred = w * direct_year_trend[r](Y)  +  (1-w) * (final_trend(Y) x ratio[r])
  where w = ENSEMBLE_WEIGHT (default 0.5).

Slots with < MIN_YEARS_FOR_TREND data points for a given round fall back to
the historical median for that round.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import (
    COL_YEAR, COL_ROUND, COL_INSTITUTE, COL_PROGRAM,
    COL_QUOTA, COL_SEAT_TYPE, COL_GENDER, COL_EXAM_TYPE,
    COL_CLOSE_RANK,
    MIN_YEARS_FOR_TREND, ENSEMBLE_WEIGHT,
    MODEL_DIR, MODEL_PATH, DEFAULT_TREND_MODEL,
)
from .loader import load

SLOT_COLS = [COL_INSTITUTE, COL_PROGRAM, COL_QUOTA,
             COL_SEAT_TYPE, COL_GENDER, COL_EXAM_TYPE]

# Supported trend models for the year-signal component
TREND_MODELS = ["ols", "theil_sen", "weighted_ols", "median",
                "ridge", "svr_linear", "svr_rbf", "ar1", "arp"]

_ARP_MAX_P = 3   # maximum AR order tried by AIC selection

# Exponential decay rate for weighted_ols - weight = exp(λ x (year - min_year))
# λ=0.3 gives the most recent year ~10x the weight of the oldest year (9-year span).
_DECAY_LAMBDA = 0.3


class _ARpEstimator:
    """
    AR(p) mean-reversion estimator with AIC-based order selection.

    Fits  (C_t - mu) = phi_1*(C_{t-1}-mu) + ... + phi_p*(C_{t-p}-mu)
    by ridge-regularised OLS in demeaned space.  Order p is chosen from
    {1 ... _ARP_MAX_P} by AIC (penalises added lags).  A small ridge term
    (1e-6 * I) guards against near-singular design matrices when observations
    are nearly collinear.

    Prediction is always one step ahead from the last p observed closing ranks;
    the `year` argument is ignored for the same reason as _AR1Estimator.
    """
    def __init__(self, mu: float, phi: np.ndarray, last_closes: np.ndarray):
        self._mu   = mu
        self._phi  = phi          # shape (p,): coefficients for lags 1 … p
        self._last = last_closes  # shape (p,): [most_recent, …, p-th most recent]

    def predict(self, X) -> np.ndarray:
        pred = self._mu + float(np.dot(self._phi, self._last - self._mu))
        return np.full(len(X), pred)


class _AR1Estimator:
    """
    Explicit AR(1) mean-reversion estimator.

    Fits  C_t = mu + phi * (C_{t-1} - mu) + eps  on consecutive year pairs
    via OLS without intercept in demeaned space.  At prediction time the
    forecast is always one step ahead from the last observed closing rank:

        pred = mu + phi * (last_close - mu)

    The `year` argument passed to predict() is ignored intentionally; AR(1)
    is not a year-indexed function but a state-space update from the most
    recent observation.  phi is clamped to [-1, 1] to guarantee stationarity.
    """
    def __init__(self, mu: float, phi: float, last_close: float):
        self._mu   = mu
        self._phi  = phi
        self._last = last_close

    def predict(self, X) -> np.ndarray:
        pred = self._mu + self._phi * (self._last - self._mu)
        return np.full(len(X), pred)


class _ScaledEstimator:
    """
    Wraps a sklearn estimator with per-feature StandardScaler on X and y.

    SVR and Ridge work in normalized space; this wrapper handles the
    transform/inverse-transform so predict([[year]]) returns raw rank values.
    The RBF kernel has a particularly useful property: when the query year is
    outside the training range, the kernel weights decay to zero and the
    prediction reverts toward the training-target mean - approximating median
    behaviour naturally.
    """
    def __init__(self, estimator, scaler_X, scaler_y):
        self._est = estimator
        self._sx  = scaler_X
        self._sy  = scaler_y

    def predict(self, X):
        X_s = self._sx.transform(np.asarray(X).reshape(-1, 1))
        y_s = self._est.predict(X_s).reshape(-1, 1)
        return self._sy.inverse_transform(y_s).ravel()


class SlotModel:
    """
    Per-slot ensemble of year-trend models and round-progression ratios.
    """

    def __init__(self, trend_model: str = DEFAULT_TREND_MODEL,
                 normalize: bool = False):
        if trend_model not in TREND_MODELS:
            raise ValueError(f"trend_model must be one of {TREND_MODELS}")
        self.trend_model = trend_model
        self.normalize:   bool = normalize
        self._exam_type:  str  = ""   # set during fit() when normalize=True
        self._institute:  str  = ""
        self._quota:      str  = ""
        # {round_no: sklearn estimator}  - year -> close_rank for that round
        self.round_year_models: dict[int, object] = {}
        # {round_no: float}  - close[r] / close[last_round], averaged across years
        self.round_ratios:      dict[int, float] = {}
        # {round_no: float}  - fallback median closing rank per round
        self.round_medians:     dict[int, float] = {}
        self.max_round: int = 1
        self.n_years:   int = 0   # number of distinct years in training data

    def _fit_trend(self, years: np.ndarray, closes: np.ndarray):
        """Return a fitted sklearn estimator, or None (falls back to median)."""
        if len(years) < MIN_YEARS_FOR_TREND or self.trend_model == "median":
            return None
        # Sort by year so consecutive pairs and last_close are correct
        order  = np.argsort(years)
        years  = years[order]
        closes = closes[order]

        if self.trend_model == "ar1":
            mu = float(np.median(closes))
            demeaned = closes - mu
            x_lag = demeaned[:-1]   # C_{t-1} - mu
            y_lag = demeaned[1:]    # C_t     - mu
            # OLS without intercept: phi = (x' y) / (x' x)
            denom = float(np.dot(x_lag, x_lag))
            phi   = float(np.dot(x_lag, y_lag) / denom) if denom > 0 else 0.0
            phi   = float(np.clip(phi, -1.0, 1.0))
            return _AR1Estimator(mu, phi, float(closes[-1]))

        if self.trend_model == "arp":
            mu = float(np.median(closes))
            d  = closes - mu          # demeaned series, sorted by year
            n  = len(d)

            best_p, best_phi, best_aicc = 1, np.array([0.0]), float("inf")
            for p in range(1, min(_ARP_MAX_P, n - 1) + 1):
                n_pairs = n - p
                # AICc requires n_pairs - p - 1 >= 1, i.e. n_pairs >= p + 2
                if n_pairs < p + 2:
                    break
                # Design matrix: column j contains lag-(j+1) of demeaned series
                X = np.column_stack(
                    [d[p - j - 1 : n - j - 1] for j in range(p)]
                )
                y = d[p:]
                # lstsq handles near-singular design matrices via SVD truncation
                phi_p, _, _, _ = np.linalg.lstsq(X, y, rcond=1e-10)
                # AICc = n*log(σ²) + 2k*n/(n-k-1)  (corrected AIC for small n)
                resid  = y - X @ phi_p
                sigma2 = max(float(np.var(resid, ddof=0)), 1e-10)
                aicc   = (n_pairs * np.log(sigma2)
                          + 2 * p * n_pairs / (n_pairs - p - 1))
                if aicc < best_aicc:
                    best_aicc, best_p, best_phi = aicc, p, phi_p

            # Store the last best_p closing ranks: [most_recent … p-th most recent]
            last_closes = closes[-best_p:][::-1].copy()
            return _ARpEstimator(mu, best_phi, last_closes)

        Y = years.reshape(-1, 1)
        if self.trend_model == "ols":
            return LinearRegression().fit(Y, closes)
        if self.trend_model == "theil_sen":
            from sklearn.linear_model import TheilSenRegressor
            import warnings
            from sklearn.exceptions import ConvergenceWarning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                return TheilSenRegressor(random_state=0, max_iter=1000).fit(Y, closes)
        if self.trend_model == "weighted_ols":
            w = np.exp(_DECAY_LAMBDA * (years - years.min()))
            return LinearRegression().fit(Y, closes, sample_weight=w)

        # Models that need normalisation (SVR and Ridge operate in standardised space)
        from sklearn.preprocessing import StandardScaler
        sx = StandardScaler()
        sy = StandardScaler()
        X_s = sx.fit_transform(Y)
        y_s = sy.fit_transform(closes.reshape(-1, 1)).ravel()

        if self.trend_model == "ridge":
            from sklearn.linear_model import Ridge
            # alpha=1.0 in normalised space gives moderate slope shrinkage
            est = Ridge(alpha=1.0).fit(X_s, y_s)
        elif self.trend_model == "svr_linear":
            from sklearn.svm import SVR
            # Linear kernel SVR - regularised linear fit, C=1 is sklearn default
            est = SVR(kernel="linear", C=1.0, epsilon=0.1).fit(X_s, y_s)
        elif self.trend_model == "svr_rbf":
            from sklearn.svm import SVR
            # RBF kernel SVR - key property: predictions outside the training
            # year range revert toward the training-target mean (≈ median).
            est = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale").fit(X_s, y_s)
        else:
            raise ValueError(f"Unknown trend_model: {self.trend_model!r}")

        return _ScaledEstimator(est, sx, sy)

    def _denorm(self, val: float, year: int) -> float:
        """Convert percentile back to absolute rank; no-op when normalize=False."""
        if not self.normalize:
            return val
        from .pool_sizes import get_pool_size_for_slot
        return val * get_pool_size_for_slot(self._exam_type, self._quota, self._institute, year)

    def fit(self, slot_df: pd.DataFrame) -> None:
        """
        slot_df: all rows for one slot, all years, all rounds.
        When normalize=True ranks are divided by pool_size[year] before fitting,
        storing everything in fractional-percentile space.  predict_round() and
        predict_interval() call _denorm() to convert outputs back to absolute ranks.
        """
        if self.normalize:
            from .pool_sizes import get_pool_size_for_slot
            self._exam_type = str(slot_df[COL_EXAM_TYPE].iloc[0])
            self._institute = str(slot_df[COL_INSTITUTE].iloc[0])
            self._quota     = str(slot_df[COL_QUOTA].iloc[0])
            slot_df = slot_df.copy()
            pool_sizes = slot_df[COL_YEAR].map(
                lambda y: get_pool_size_for_slot(
                    self._exam_type, self._quota, self._institute, int(y))
            )
            slot_df[COL_CLOSE_RANK] = slot_df[COL_CLOSE_RANK] / pool_sizes

        self.max_round = int(slot_df[COL_ROUND].max())
        self.n_years   = int(slot_df[COL_YEAR].nunique())

        # Per-round year-trend models
        for r, grp in slot_df.groupby(COL_ROUND):
            r = int(r)
            closes = grp[COL_CLOSE_RANK].values.astype(float)
            years  = grp[COL_YEAR].values.astype(float)
            self.round_medians[r] = float(np.median(closes))
            m = self._fit_trend(years, closes)
            if m is not None:
                self.round_year_models[r] = m

        # Round-progression ratios
        # For each year, compute ratio[r] = close[r] / close[max_round_in_year].
        # Only use years that have *both* the round r and the max round.
        ratio_accum: dict[int, list[float]] = {}
        for year, year_grp in slot_df.groupby(COL_YEAR):
            year_rounds = year_grp.set_index(COL_ROUND)[COL_CLOSE_RANK].to_dict()
            final_r = int(year_grp[COL_ROUND].max())
            final_close = year_rounds.get(final_r)
            if not final_close or final_close == 0:
                continue
            for r, close in year_rounds.items():
                r = int(r)
                ratio_accum.setdefault(r, []).append(close / final_close)

        self.round_ratios = {r: float(np.mean(v)) for r, v in ratio_accum.items()}

        # Historical absolute deviations from the median per round.
        # Used by predict_interval() to build per-slot prediction intervals.
        # Sorted ascending so quantile lookup is O(1).
        self.round_abs_deviations: dict[int, list[float]] = {}
        for r, grp in slot_df.groupby(COL_ROUND):
            r = int(r)
            closes = grp[COL_CLOSE_RANK].values.astype(float)
            med = float(np.median(closes))
            self.round_abs_deviations[r] = sorted(float(abs(c - med)) for c in closes)

    def predict_interval(self, round_no: int, year: int,
                         coverage: float = 0.90) -> tuple[float, float]:
        """
        Prediction interval at the requested coverage level.

        Uses the sorted absolute deviations of historical closing ranks from
        their per-round median as a non-parametric proxy for future prediction
        uncertainty.  The coverage quantile of those deviations is the
        half-width: lower = pred - half_width, upper = pred + half_width.

        Note: this is an in-sample (optimistic) calibration; the true
        leave-one-out coverage may be lower, especially for volatile slots.
        Falls back to ±20 % of the prediction when fewer than 2 observations
        are available.
        """
        pred = self.predict_round(round_no, year)
        devs = self.round_abs_deviations.get(round_no, [])
        if len(devs) >= 2:
            n      = len(devs)
            q_idx  = min(int(np.ceil(n * coverage)) - 1, n - 1)
            # devs are in the same space as the fitted targets;
            # _denorm converts percentile deviations to absolute rank width.
            half_w = self._denorm(devs[q_idx], year)
        else:
            half_w = 0.20 * pred
        return max(1.0, pred - half_w), pred + half_w

    def predict_round(self, round_no: int, year: int,
                      w: float | None = None) -> float:
        """
        Ensemble prediction for a single round.
        w: override for ENSEMBLE_WEIGHT; pass explicitly during weight tuning.
        """
        ew = ENSEMBLE_WEIGHT if w is None else w

        # Signal 1: direct year trend for this round
        if round_no in self.round_year_models:
            direct = float(self.round_year_models[round_no].predict([[year]])[0])
        else:
            direct = self.round_medians.get(round_no,
                     self.round_medians.get(self.max_round, 0))

        # Signal 2: scale from predicted final-round close
        if self.max_round in self.round_year_models:
            pred_final = float(
                self.round_year_models[self.max_round].predict([[year]])[0]
            )
        else:
            pred_final = self.round_medians.get(self.max_round, direct)

        ratio = self.round_ratios.get(round_no,
                self.round_ratios.get(self.max_round, 1.0))
        via_ratio = pred_final * ratio

        pred = ew * direct + (1 - ew) * via_ratio
        return max(1.0, self._denorm(pred, year))

    def predict_all_rounds(self, year: int, rounds: list[int],
                           w: float | None = None) -> dict[int, int]:
        """Return {round_no: predicted_close_rank} for each requested round."""
        return {r: int(round(self.predict_round(r, year, w=w))) for r in rounds}


def train(csv_path: str, model_path: str = MODEL_PATH,
          trend_model: str = DEFAULT_TREND_MODEL,
          normalize: bool = False) -> dict:
    df = load(csv_path)
    print(f"Training on {len(df):,} rows  |  "
          f"{df[COL_YEAR].nunique()} years  |  "
          f"{df[COL_ROUND].nunique()} rounds  |  "
          f"trend={trend_model}  |  normalize={normalize}")

    slots: dict[tuple, SlotModel] = {}
    for key, grp in df.groupby(SLOT_COLS):
        m = SlotModel(trend_model=trend_model, normalize=normalize)
        m.fit(grp)
        slots[tuple(key)] = m

    model = {
        "slots":      slots,
        "slot_cols":  SLOT_COLS,
        "trend_model": trend_model,
        "normalize":  normalize,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained {len(slots):,} slot models -> {model_path}")
    return model


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "josaa_ranks.csv"
    train(csv_path)
