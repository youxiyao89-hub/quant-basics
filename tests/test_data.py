import numpy as np
import pandas as pd
import pytest

from qbasics.data.tushare_io import (
    TushareCache,
    hfq_prices,
    log_returns,
    members_asof,
    month_ranges,
    simple_returns,
    to_wide,
    union_members,
)


def _daily():
    return pd.DataFrame({
        "ts_code": ["A.SH", "A.SH", "B.SZ"],
        "trade_date": ["20240102", "20240103", "20240102"],
        "open": [10.0, 9.0, 5.0], "high": [10.5, 9.5, 5.2],
        "low": [9.8, 8.9, 4.9], "close": [10.0, 9.0, 5.0],
    })


def _adj():
    return pd.DataFrame({
        "ts_code": ["A.SH", "A.SH", "B.SZ"],
        "trade_date": ["20240102", "20240103", "20240102"],
        # A pays a dividend before 01-03: raw price drops, factor rises
        "adj_factor": [1.0, 1.1, 2.0],
    })


def test_hfq_is_raw_times_factor():
    p = hfq_prices(_daily(), _adj())
    a = p[p.ts_code == "A.SH"].set_index("trade_date")
    assert a["hfq_close"].tolist() == pytest.approx([10.0, 9.9])
    # identity check used in the data layer: hfq / raw == adj_factor
    assert np.allclose(p["hfq_close"] / p["close"], p["adj_factor"])


def test_hfq_refuses_missing_factor():
    with pytest.raises(ValueError, match="without adj_factor"):
        hfq_prices(_daily(), _adj().iloc[:2])


def test_members_asof_uses_only_published_snapshots():
    w = pd.DataFrame({
        "index_code": ["000300.SH"] * 4,
        "con_code": ["A", "B", "A", "C"],
        "trade_date": ["20240131", "20240131", "20240229", "20240229"],
        "weight": [1, 1, 1, 1],
    })
    assert members_asof(w, "2024-01-15") == set()          # nothing published yet
    assert members_asof(w, "2024-02-10") == {"A", "B"}     # January snapshot
    assert members_asof(w, "2024-02-29") == {"A", "C"}     # February snapshot
    assert union_members(w) == ["A", "B", "C"]


def test_month_ranges_cover_leap_february():
    assert month_ranges("20240115", "20240310") == [
        ("20240101", "20240131"), ("20240201", "20240229"), ("20240301", "20240331")]


def test_simple_vs_log_returns():
    px = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
    assert simple_returns(px)["A"].tolist() == pytest.approx([0.10, -0.10])
    assert log_returns(px)["A"].tolist() == pytest.approx([np.log(1.1), np.log(0.9)])


def test_to_wide():
    p = hfq_prices(_daily(), _adj())
    w = to_wide(p)
    assert list(w.columns) == ["A.SH", "B.SZ"] and w.shape == (2, 2)


class _FakePro:
    def __init__(self):
        self.calls = 0

    def daily(self, **kw):
        self.calls += 1
        return _daily()[_daily().ts_code == kw["ts_code"]]


def test_cache_hits_network_once(tmp_path):
    fake = _FakePro()
    c = TushareCache(pro=fake, cache_dir=tmp_path, pause=0)
    a1 = c.daily("A.SH", "20240101", "20240131")
    a2 = c.daily("A.SH", "20240101", "20240131")
    assert fake.calls == 1
    pd.testing.assert_frame_equal(a1.reset_index(drop=True), a2.reset_index(drop=True))
