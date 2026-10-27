"""#01 Mean-variance optimisation -- YOU implement these.

Start from your existing MV paper code, but follow the v2 rules:
1. Inputs `mu` (expected *simple* returns) and `cov` are per-period or annual,
   just be consistent; weights are long-only and sum to 1 unless stated.
2. Every function returns a pandas Series indexed like `mu` / `cov`.
3. Out-of-sample use: estimate mu/cov on a past window only, then hold the
   weights over the next period (see notebooks/01_mean_variance_v2/README.md).

Tests: tests/test_mv.py (skipped until implemented).
"""
from __future__ import annotations

import pandas as pd


def min_variance_unconstrained(cov: pd.DataFrame) -> pd.Series:
    """Global minimum-variance weights with only sum(w)=1 (shorting allowed).

    Closed form: w = inv(S) 1 / (1' inv(S) 1). Derive it yourself with a
    Lagrangian before coding -- this is a classic interview question.
    """
    raise NotImplementedError


def mean_variance(mu: pd.Series, cov: pd.DataFrame, lam: float,
                  w_max: float = 1.0) -> pd.Series:
    """Solve  min  lam * w'S w - (1 - lam) * mu'w
             s.t. sum(w) = 1, 0 <= w_i <= w_max.

    lam = 1 -> minimum variance, lam = 0 -> maximum expected return.
    Hint: scipy.optimize.minimize(method="SLSQP") with bounds and an equality
    constraint; check the solver's `success` flag and raise if it failed.
    """
    raise NotImplementedError


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0,
               w_max: float = 1.0) -> pd.Series:
    """Long-only tangency portfolio: maximise (mu'w - rf) / sqrt(w'S w).

    The objective is a ratio, not a quadratic programme -- explain in your
    README how you solve it (e.g. SLSQP on the negative Sharpe ratio).
    """
    raise NotImplementedError
