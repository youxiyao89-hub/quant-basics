import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def call_or_skip(fn, *args, **kwargs):
    """Run fn; if it is still a TODO stub, skip the test instead of failing."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"TODO: implement {fn.__module__}.{fn.__name__}")
