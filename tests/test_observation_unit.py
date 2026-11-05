"""Tests for the observation unit helper."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from spatialdata_io.readers._utils._utils import _set_observation_unit


@pytest.mark.parametrize(
    ("unit", "size_um", "expected"),
    [
        (None, 2, ("subcellular_bin", 2.0)),
        (None, 9.99, ("subcellular_bin", 9.99)),
        (None, 10, ("spot_or_bin", 10.0)),
        (None, 55, ("spot_or_bin", 55.0)),
        (None, None, ("spot_or_bin", None)),
        ("cell", None, ("cell", None)),
        ("cell", 5, ("cell", 5.0)),
    ],
)
def test_set_observation_unit(unit: str | None, size_um: float | None, expected: tuple[str, float | None]) -> None:
    table = ad.AnnData(np.zeros((1, 1)))
    _set_observation_unit(table, unit, size_um)
    uns = table.uns["spatialdata_io"]
    assert (uns["observation_unit"], uns["unit_size_um"]) == expected


def test_set_observation_unit_merges_into_existing_uns() -> None:
    table = ad.AnnData(np.zeros((1, 1)))
    table.uns["spatialdata_io"] = {"pitch_unknown": True}
    _set_observation_unit(table, None, 8)
    assert table.uns["spatialdata_io"] == {
        "pitch_unknown": True,
        "observation_unit": "subcellular_bin",
        "unit_size_um": 8.0,
    }
