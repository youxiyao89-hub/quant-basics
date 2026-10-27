"""Paths and secrets.

The Tushare token is read from the environment variable TUSHARE_TOKEN, which
can be put in a local `.env` file (git-ignored). Never hard-code the token.
"""
from __future__ import annotations

import os
from pathlib import Path

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("QB_DATA_DIR", PROJECT_ROOT / "data")).resolve()


def tushare_token() -> str:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token or token == "your_token_here":
        raise RuntimeError(
            "TUSHARE_TOKEN is not set. Copy .env.example to .env and put your token there."
        )
    return token
