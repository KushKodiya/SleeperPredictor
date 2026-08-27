"""M1 cache-mechanism tests. No network: the loader is injected (PRD §10)."""

import polars as pl
import pytest

from ffdraft.data.nflverse import _cached


def _make_loader(frame):
    state = {"calls": 0}

    def loader():
        state["calls"] += 1
        return frame

    return loader, state


def test_cache_hit_returns_identical_and_skips_loader(tmp_path):
    frame = pl.DataFrame({"gsis_id": ["00-1"], "x": [2]})
    loader, state = _make_loader(frame)

    df1 = _cached("t", loader, {"gsis_id"}, "src", cache_dir=tmp_path)
    df2 = _cached("t", loader, {"gsis_id"}, "src", cache_dir=tmp_path)

    assert state["calls"] == 1  # second call is a cache hit
    assert df1.equals(df2)


def test_refresh_bypasses_cache(tmp_path):
    frame = pl.DataFrame({"gsis_id": ["00-1"]})
    loader, state = _make_loader(frame)

    _cached("t", loader, {"gsis_id"}, "src", cache_dir=tmp_path)
    _cached("t", loader, {"gsis_id"}, "src", cache_dir=tmp_path, refresh=True)

    assert state["calls"] == 2  # refresh re-fetched


def test_missing_column_raises_with_available_listed(tmp_path):
    loader, _ = _make_loader(pl.DataFrame({"a": [1]}))
    with pytest.raises(ValueError) as exc:
        _cached("t2", loader, {"gsis_id"}, "nflverse.x", cache_dir=tmp_path)
    assert "gsis_id" in str(exc.value) and "'a'" in str(exc.value)
