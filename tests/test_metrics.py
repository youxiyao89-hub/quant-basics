import numpy as np
import pandas as pd
import pytest

from conftest import call_or_skip
from qbasics import metrics as m

R = pd.Series([0.10, -0.50, 0.20])   # NAV: 1.10, 0.55, 0.66


def test_nav():
    assert call_or_skip(m.nav, R).tolist() == pytest.approx([1.10, 0.55, 0.66])


def test_max_drawdown_is_negative_peak_to_trough():
    assert call_or_skip(m.max_drawdown, R) == pytest.approx(-0.50)


def test_annualized_return_is_geometric():
    # 3 periods with periods=3 means exactly one "year": 0.66 - 1
    assert call_or_skip(m.annualized_return, R, periods=3) == pytest.approx(-0.34)


def test_annualized_vol():
    r = pd.Series([0.01, -0.01])
    expected = np.std([0.01, -0.01], ddof=1) * np.sqrt(252)
    assert call_or_skip(m.annualized_vol, r) == pytest.approx(expected)


def test_sharpe_zero_rf():
    r = pd.Series([0.01, 0.03, -0.01, 0.02])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert call_or_skip(m.sharpe_ratio, r) == pytest.approx(expected)
