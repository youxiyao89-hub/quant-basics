# quant-basics

Basic quantitative-finance models rebuilt from scratch on China A-share data — one model per week, each with a short write-up and tests. This repository is my learning log on the way to factor research and live trading.

**Author:** Yao Youxi (finance undergraduate, Guangzhou Nanfang College)

## Roadmap

| # | Model | Status |
|---|---|---|
| 00 | Data layer: cached Tushare download, back-adjusted prices, point-in-time index membership | done |
| 01 | Mean-variance v2: out-of-sample rolling backtest vs. equal weight and CSI 300 (+ performance metrics) | in progress |
| 02 | CAPM and rolling beta | planned |
| 03 | Fama-French three-factor regression | planned |
| 04 | Single-factor portfolio sorts | planned |
| 05 | Information coefficient (IC / RankIC / ICIR) | planned |
| 06 | Fama-MacBeth regression with Newey-West errors | planned |
| 07 | Momentum and short-term reversal | planned |
| 08 | Option pricing: Black-Scholes, binomial tree, Monte Carlo | planned |
| 09 | Machine-learning stock selection with walk-forward training | planned |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then put your Tushare token in .env
pytest                        # data-layer tests pass; model tests are skipped until implemented
python scripts/download_hs300.py --start 20221101 --end 20251231
```

## Principles

- **No look-ahead.** Universes come from index snapshots that were already published on each date; prices are back-adjusted (fixed history), never forward-adjusted.
- **Out-of-sample first.** Parameters are estimated on past data only; in-sample results are labelled as such.
- **Known-answer tests.** Every model has at least one test whose answer can be checked by hand.

## Data

Market data comes from [Tushare Pro](https://tushare.pro) and is **not** redistributed here (see its terms). Scripts download and cache it locally under `data/` (git-ignored).

## Use of AI tools

I write the core model code myself. AI assistants are used for code review, debugging explanations and scaffolding (repository layout, data-layer boilerplate); this is noted in each module's docstring where relevant.

## License

MIT
