# 01 · Mean-variance v2 (out-of-sample)

Upgrade of my in-sample MV paper (20 holdings of the E Fund CSI 300 ETF, 2022-11 to 2025-12).

## Why v2

| v1 problem | v2 fix |
|---|---|
| Universe = today's top holdings, so the sample is chosen with hindsight (e.g. stocks that became top weights *because* they rallied) | Universe on each rebalance date = CSI 300 constituents from the latest published monthly snapshot (`members_asof`) |
| mu and Sigma estimated and evaluated on the same 3 years (in-sample Sharpe 2.40) | Rolling estimation window (e.g. 252 days), hold weights for the next month, evaluate only on held-out months |
| Log returns inside a portfolio model | Simple returns (portfolio return is linear in weights only for simple returns) |
| Concentrated weights (58% in one stock) | Weight cap (e.g. 10%) and Ledoit-Wolf covariance shrinkage as variants |
| No costs | Turnover x one-way cost; report gross and net |

## Experiment (write it down before running)

- Rebalance: first trading day of each month; trade at next day's open (or close, but state it).
- Strategies: equal weight (1/N), min-variance, max-Sharpe, each with/without 10% cap.
- Benchmarks: 1/N on the same universe; CSI 300 index.
- Metrics: annual return, vol, Sharpe, max drawdown, turnover, net-of-cost return.
- Reference: DeMiguel, Garlappi & Uppal (2009), *Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy?*, Review of Financial Studies 22(5).

## Deliverables

1. `mv_v2.ipynb` with the backtest and one figure of cumulative net returns.
2. A results table, and three sentences on what changed versus v1 and why.
