"""Cached Tushare access plus small, testable helpers.

Design
------
* Every API call is cached to ``<DATA_DIR>/<api_name>/<cache_key>.parquet``.
  Re-running a script never re-downloads, so experiments are reproducible and
  you stay well inside your point/frequency limits.
* All transformation logic lives in *pure functions* (no network), so it can
  be unit-tested with tiny hand-made tables.

Point requirements (Tushare Pro, checked 2026-10-17 at
https://tushare.pro/document/1?doc_id=108): ``daily`` needs 120 points;
``index_weight`` (monthly index constituents) needs 2000. ``fund_daily`` (ETF
bars) is listed as 2000 in the points table but 5000 on its own page -- test
it with your token; fall back to AKShare if it is refused.
"""
from __future__ import annotations

import calendar
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from .. import config

PRICE_COLS = ["open", "high", "low", "close"]


# --------------------------------------------------------------------------
# Cached client
# --------------------------------------------------------------------------
class TushareCache:
    """Thin wrapper around ``tushare.pro_api`` that caches every response.

    Parameters
    ----------
    pro : optional
        An object exposing Tushare API methods (``pro.daily(...)`` etc.).
        Pass a fake in tests; leave ``None`` to build the real client.
    """

    def __init__(
        self,
        pro=None,
        cache_dir: Optional[Path] = None,
        pause: float = 0.35,
        retries: int = 3,
    ) -> None:
        if pro is None:
            import tushare as ts

            pro = ts.pro_api(config.tushare_token())
        self.pro = pro
        self.cache_dir = Path(cache_dir or config.DATA_DIR)
        self.pause = pause
        self.retries = retries

    # -- generic -----------------------------------------------------------
    def query(self, api_name: str, cache_key: str, **params) -> pd.DataFrame:
        path = self.cache_dir / api_name / f"{cache_key}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        fn: Callable = getattr(self.pro, api_name)
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            try:
                df = fn(**params)
                break
            except Exception as err:  # network / rate-limit errors
                last_err = err
                time.sleep(self.pause * 4 * attempt)
        else:
            raise RuntimeError(f"{api_name}({params}) failed after {self.retries} tries") from last_err
        if df is None:
            df = pd.DataFrame()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        time.sleep(self.pause)
        return df

    # -- specific endpoints ------------------------------------------------
    def daily(self, ts_code: str, start: str, end: str) -> pd.DataFrame:
        """Unadjusted daily bars (open/high/low/close in CNY)."""
        return self.query("daily", f"{ts_code}_{start}_{end}",
                          ts_code=ts_code, start_date=start, end_date=end)

    def adj_factor(self, ts_code: str, start: str, end: str) -> pd.DataFrame:
        return self.query("adj_factor", f"{ts_code}_{start}_{end}",
                          ts_code=ts_code, start_date=start, end_date=end)

    def index_weight(self, index_code: str, start: str, end: str) -> pd.DataFrame:
        """Monthly constituents and weights, queried one calendar month at a time
        (Tushare recommends month-start/month-end dates for this endpoint)."""
        frames = [
            self.query("index_weight", f"{index_code}_{m0}",
                       index_code=index_code, start_date=m0, end_date=m1)
            for m0, m1 in month_ranges(start, end)
        ]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=["index_code", "con_code", "trade_date", "weight"])
        return pd.concat(frames, ignore_index=True).drop_duplicates()


# --------------------------------------------------------------------------
# Pure helpers (no network) -- these are what the tests exercise
# --------------------------------------------------------------------------
def month_ranges(start: str, end: str) -> list[tuple[str, str]]:
    """Calendar months covering [start, end] as ('YYYYMM01', 'YYYYMMDD-last') pairs."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out = []
    cur = pd.Timestamp(year=s.year, month=s.month, day=1)
    while cur <= e:
        last = calendar.monthrange(cur.year, cur.month)[1]
        out.append((cur.strftime("%Y%m%d"), cur.replace(day=last).strftime("%Y%m%d")))
        cur = (cur + pd.offsets.MonthBegin(1)).normalize()
    return out


def _as_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str), format="%Y%m%d")


def hfq_prices(daily: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """Back-adjusted (hfq) prices = raw price x cumulative adj_factor.

    Back-adjustment keeps historical values fixed when new dividends arrive,
    so results are reproducible (forward-adjusted prices are rewritten every
    time a new corporate action happens).

    Raises if any bar has no matching adjustment factor -- silent NaNs are how
    data bugs hide.
    """
    keys = ["ts_code", "trade_date"]
    if daily.duplicated(keys).any():
        raise ValueError("daily has duplicated (ts_code, trade_date) rows")
    m = daily.merge(adj[keys + ["adj_factor"]], on=keys, how="left", validate="one_to_one")
    missing = m["adj_factor"].isna()
    if missing.any():
        bad = m.loc[missing, keys].head(5).to_dict("records")
        raise ValueError(f"{int(missing.sum())} bars without adj_factor, e.g. {bad}")
    out = m[keys].copy()
    out["trade_date"] = _as_date(out["trade_date"])
    for c in PRICE_COLS:
        out[c] = m[c].astype(float)                    # raw, tradable price
        out[f"hfq_{c}"] = m[c].astype(float) * m["adj_factor"].astype(float)
    out["adj_factor"] = m["adj_factor"].astype(float)
    if "vol" in m:
        out["vol"] = m["vol"]
    if "amount" in m:
        out["amount"] = m["amount"]
    return out.sort_values(keys).reset_index(drop=True)


def members_asof(weights: pd.DataFrame, date) -> set[str]:
    """Index constituents according to the latest snapshot dated on/before `date`.

    This is the anti-look-ahead rule for building a universe: on any date you
    may only use the membership list that was already published. Snapshots are
    monthly, so membership changes inside a month show up at the next snapshot.
    """
    d = pd.Timestamp(date)
    snap = _as_date(weights["trade_date"])
    eligible = snap[snap <= d]
    if eligible.empty:
        return set()
    latest = eligible.max()
    return set(weights.loc[snap == latest, "con_code"])


def union_members(weights: pd.DataFrame) -> list[str]:
    """Every stock that was ever a constituent in the snapshots (download list)."""
    return sorted(set(weights["con_code"]))


def to_wide(prices: pd.DataFrame, col: str = "hfq_close") -> pd.DataFrame:
    """Long (ts_code, trade_date, ...) -> wide date x ts_code table."""
    return prices.pivot(index="trade_date", columns="ts_code", values=col).sort_index()


def simple_returns(price_wide: pd.DataFrame) -> pd.DataFrame:
    """r_t = P_t / P_{t-1} - 1. Portfolio returns are linear in weights only for
    simple returns, so use these inside mean-variance optimisation."""
    return price_wide.pct_change(fill_method=None).iloc[1:]


def log_returns(price_wide: pd.DataFrame) -> pd.DataFrame:
    """r_t = ln P_t - ln P_{t-1}. Additive over time; good for compounding and
    statistics, but a weighted sum of log returns is NOT the portfolio log return."""
    return np.log(price_wide).diff().iloc[1:]


def check_panel(prices: pd.DataFrame, codes: Iterable[str]) -> pd.DataFrame:
    """One-line-per-stock health report: rows, first/last date, missing closes."""
    rep = (prices.groupby("ts_code")
           .agg(rows=("trade_date", "size"), first=("trade_date", "min"),
                last=("trade_date", "max"), nan_close=("close", lambda s: int(s.isna().sum())))
           .reindex(sorted(codes)))
    rep["downloaded"] = rep["rows"].notna()
    return rep
