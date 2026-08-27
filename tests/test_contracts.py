import polars as pl
import pytest

from ffdraft.contracts import assert_columns


def test_passes_when_all_present():
    df = pl.DataFrame({"a": [1], "b": [2]})
    assert_columns(df, {"a", "b"}, "src")  # must not raise


def test_raises_listing_missing_and_available():
    df = pl.DataFrame({"a": [1], "z": [2]})
    with pytest.raises(ValueError) as exc:
        assert_columns(df, {"a", "b", "c"}, "mysrc")
    msg = str(exc.value)
    assert "mysrc" in msg
    # missing columns listed
    assert "'b'" in msg and "'c'" in msg
    # available columns listed
    assert "'a'" in msg and "'z'" in msg
