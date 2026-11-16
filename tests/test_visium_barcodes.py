"""visium() joins the counts barcodes to tissue_positions only exactly or by a declared ``barcode_rule``."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from spatialdata import read_zarr

from spatialdata_io import visium
from tests.test_visium_protein import _BARCODES, _space_ranger

# a re-export spelling (GSE256397 stored ``Injury72h_AAACAAGTATCTCCCA.1`` for ``AAACAAGTATCTCCCA-1``)
_RULE = {"strip_prefix": "Injury72h_", "suffix_from": ".1", "suffix_to": "-1"}


def _respell(root: Path, barcodes: list[str]) -> Path:
    with h5py.File(root / "S1_filtered_feature_bc_matrix.h5", "r+") as f:
        del f["matrix/barcodes"]
        f["matrix/barcodes"] = np.array(barcodes, dtype="S")
    return root


def test_exact_barcodes_are_unchanged(tmp_path: Path) -> None:
    sdata = visium(_space_ranger(tmp_path / "S1"))
    table = sdata["table"]
    assert list(table.obs_names) == _BARCODES and "barcode_deposited" not in table.obs
    assert not np.isnan(table.obsm["spatial"]).any()
    assert table.uns["visium_barcodes"] == {"rule": {}, "match_rate": 1.0, "n_counts_barcodes": 3, "n_matched": 3}


def test_mismatched_barcodes_raise(tmp_path: Path) -> None:
    root = _respell(_space_ranger(tmp_path / "S1"), [f"Injury72h_{b[:-2]}.1" for b in _BARCODES])
    with pytest.raises(ValueError, match=r"3 of 3 counts barcodes are not in tissue_positions.*barcode_rule"):
        visium(root)


def test_one_unmatched_barcode_raises(tmp_path: Path) -> None:
    root = _respell(_space_ranger(tmp_path / "S1"), [*_BARCODES[:2], "TTTT-1"])
    with pytest.raises(ValueError, match="1 of 3 counts barcodes"):
        visium(root)


def test_declared_rule_joins_the_positions(tmp_path: Path) -> None:
    deposited = [f"Injury72h_{b[:-2]}.1" for b in _BARCODES]
    sdata = visium(_respell(_space_ranger(tmp_path / "S1"), deposited), barcode_rule=_RULE)
    for t in sdata.tables.values():
        assert list(t.obs_names) == _BARCODES
        assert list(t.obs["barcode_deposited"]) == deposited
        assert t.uns["visium_barcodes"] == {"rule": _RULE, "match_rate": 1.0, "n_counts_barcodes": 3, "n_matched": 3}
    np.testing.assert_array_equal(sdata["table"].obsm["spatial"], [[100, 100], [150, 100], [200, 100]])
    assert list(sdata["table"].obs["in_tissue"]) == [1, 1, 1]
    sdata.write(tmp_path / "x.zarr")
    assert read_zarr(tmp_path / "x.zarr")["table"].uns["visium_barcodes"]["rule"] == _RULE


@pytest.mark.parametrize(
    ("rule", "match"),
    [
        ({"strip_prefix": "Sham_"}, "changes no barcode"),
        ({"suffix_from": ".1"}, "both `suffix_from` and `suffix_to`"),
        ({"prefix": "Injury72h_"}, "Unknown barcode_rule keys"),
    ],
)
def test_bad_rules_raise(tmp_path: Path, rule: dict[str, str], match: str) -> None:
    root = _respell(_space_ranger(tmp_path / "S1"), [f"Injury72h_{b[:-2]}.1" for b in _BARCODES])
    with pytest.raises(ValueError, match=match):
        visium(root, barcode_rule=rule)


def test_rule_that_merges_barcodes_raises(tmp_path: Path) -> None:
    root = _respell(_space_ranger(tmp_path / "S1"), ["A_AAAC-1", "B_AAAC-1", "A_AAAT-1"])
    with pytest.raises(ValueError, match="same name"):
        visium(root, barcode_rule={"strip_prefix": "[AB]_"})
