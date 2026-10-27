import numpy as np
import pandas as pd
import pytest

from conftest import call_or_skip
from qbasics import mv

ASSETS = ["X", "Y"]
MU = pd.Series([0.10, 0.20], index=ASSETS)
# uncorrelated: vol 20% and 30%
COV = pd.DataFrame([[0.04, 0.0], [0.0, 0.09]], index=ASSETS, columns=ASSETS)


def test_min_variance_two_uncorrelated_assets():
    # w_i proportional to 1/sigma_i^2 -> 25 : 11.11
    w = call_or_skip(mv.min_variance_unconstrained, COV)
    assert w["X"] == pytest.approx(25 / (25 + 1 / 0.09))
    assert w.sum() == pytest.approx(1.0)


def test_min_variance_first_order_condition_with_correlation():
    cov = pd.DataFrame([[0.04, 0.012], [0.012, 0.09]], index=ASSETS, columns=ASSETS)
    w = call_or_skip(mv.min_variance_unconstrained, cov)
    g = cov.values @ w.values           # must be the same for every asset
    assert g[0] == pytest.approx(g[1])


def test_lambda_one_is_min_variance():
    w = call_or_skip(mv.mean_variance, MU, COV, lam=1.0)
    assert w["X"] == pytest.approx(25 / (25 + 1 / 0.09), abs=1e-4)


def test_lambda_zero_is_max_return_and_respects_cap():
    w = call_or_skip(mv.mean_variance, MU, COV, lam=0.0)
    assert w["Y"] == pytest.approx(1.0, abs=1e-4)
    w_cap = call_or_skip(mv.mean_variance, MU, COV, lam=0.0, w_max=0.6)
    assert w_cap["Y"] == pytest.approx(0.6, abs=1e-4)
    assert (w_cap >= -1e-9).all() and w_cap.sum() == pytest.approx(1.0)


def test_max_sharpe_tangency_uncorrelated():
    # tangency is proportional to inv(S)(mu - rf) = [2.5, 2.222...]
    w = call_or_skip(mv.max_sharpe, MU, COV, rf=0.0)
    expected = np.array([0.10 / 0.04, 0.20 / 0.09])
    expected = expected / expected.sum()
    assert w.values == pytest.approx(expected, abs=1e-3)
