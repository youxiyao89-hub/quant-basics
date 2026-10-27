"""Download CSI 300 historical constituents + daily bars + adjustment factors.

Usage (from the repo root, after filling .env):
    python scripts/download_hs300.py --start 20221101 --end 20251231

What it does
1. index_weight for 000300.SH, one call per month (needs 2000 points).
2. The union of every stock that was a constituent in any month -> download
   list. Using the *historical* union (not today's list) avoids survivorship
   and look-ahead bias.
3. daily + adj_factor for each stock, cached under data/.
4. Builds back-adjusted prices, saves data/hs300_prices.parquet, and runs
   self-checks that must all pass.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qbasics import config  # noqa: E402
from qbasics.data.tushare_io import (  # noqa: E402
    TushareCache, check_panel, hfq_prices, members_asof, union_members)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="000300.SH")
    ap.add_argument("--start", default="20221101")
    ap.add_argument("--end", default="20251231")
    args = ap.parse_args()

    tc = TushareCache()
    w = tc.index_weight(args.index, args.start, args.end)
    if w.empty:
        sys.exit("index_weight returned nothing -- check your points (needs 2000) and dates.")
    codes = union_members(w)
    snaps = sorted(w["trade_date"].unique())
    print(f"{len(snaps)} monthly snapshots, {len(codes)} distinct constituents")

    frames = []
    for i, code in enumerate(codes, 1):
        d = tc.daily(code, args.start, args.end)
        a = tc.adj_factor(code, args.start, args.end)
        if d.empty:
            print(f"  [{i}/{len(codes)}] {code}: no bars in range")
            continue
        frames.append(hfq_prices(d, a))
        if i % 50 == 0:
            print(f"  [{i}/{len(codes)}] downloaded")
    prices = pd.concat(frames, ignore_index=True)

    # ---- self-checks (each must hold; a failure means stop and investigate)
    per_snap = w.groupby("trade_date")["con_code"].nunique()
    assert np.allclose(prices["hfq_close"] / prices["close"], prices["adj_factor"]), \
        "hfq/raw != adj_factor"
    assert not prices.duplicated(["ts_code", "trade_date"]).any(), "duplicate bars"
    print(f"constituents per snapshot: min {per_snap.min()}, max {per_snap.max()} (expect 300)")
    mid = pd.Timestamp(snaps[len(snaps) // 2])
    print(f"members as of {mid.date()}: {len(members_asof(w, mid))}")

    out = config.DATA_DIR / "hs300_prices.parquet"
    prices.to_parquet(out, index=False)
    w.to_parquet(config.DATA_DIR / "hs300_weights.parquet", index=False)
    rep = check_panel(prices, codes)
    print(rep.describe(include="all").T[["count"]].T)
    print(f"saved {len(prices):,} rows -> {out}")


if __name__ == "__main__":
    main()
