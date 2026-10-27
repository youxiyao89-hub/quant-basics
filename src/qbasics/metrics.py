"""#01 Performance metrics -- YOU implement these (week of 10/26).

Write each function yourself first. The tests in tests/test_metrics.py use
tiny hand-computable examples; they are skipped until you remove the
NotImplementedError.

Conventions
-----------
* `r` is a pandas Series of *simple* periodic returns (e.g. daily).
* `periods` is the number of periods per year (252 for A-share trading days).
* Annual return is geometric (compounded), not mean x 252.
"""
from __future__ import annotations

import pandas as pd


def nav(r: pd.Series) -> pd.Series:
    """Net asset value starting at 1.0: cumulative product of (1 + r)."""
    raise NotImplementedError


def annualized_return(r: pd.Series, periods: int = 252) -> float:
    """Geometric annual return: NAV_end ** (periods / n) - 1."""
    raise NotImplementedError


def annualized_vol(r: pd.Series, periods: int = 252) -> float:
    """Sample standard deviation (ddof=1) x sqrt(periods)."""
    raise NotImplementedError


def sharpe_ratio(r: pd.Series, rf_annual: float = 0.0, periods: int = 252) -> float:
    """(annualized mean excess return) / (annualized vol).

    Convert rf to a per-period rate first: (1 + rf_annual) ** (1 / periods) - 1.
    State in your README which risk-free rate you use and its date/source.
    """
    raise NotImplementedError


def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough fall of NAV, returned as a negative number (e.g. -0.25)."""
    raise NotImplementedError
