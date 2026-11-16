"""visium() joins the counts barcodes to tissue_positions only exactly or by a declared ``barcode_rule``."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.sparse as sp
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
    assert table.uns["visium_barcodes"] == {
        "rule": {},
        "min_barcode_match": 1.0,
        "match_rate": 1.0,
        "n_counts_barcodes": 3,
        "n_matched": 3,
        "unmatched": {"n": 0, "barcodes": []},
    }


def test_mismatched_barcodes_raise(tmp_path: Path) -> None:
    root = _respell(_space_ranger(tmp_path / "S1"), [f"Injury72h_{b[:-2]}.1" for b in _BARCODES])
    with pytest.raises(ValueError, match=r"3 of 3 counts barcodes are not in tissue_positions.*barcode_rule"):
        visium(root)


def test_one_unmatched_barcode_raises_by_default(tmp_path: Path) -> None:
    root = _respell(_space_ranger(tmp_path / "S1"), [*_BARCODES[:2], "TTTT-1"])
    with pytest.raises(ValueError, match="1 of 3 counts barcodes.*min_barcode_match 1.0"):
        visium(root)


def _many_spots(root: Path, n: int, unmatched: int) -> Path:
    """``n`` spots on one array row, 100 µm apart; the last ``unmatched`` counts barcodes are not in the positions."""
    (root / "spatial").mkdir(parents=True)
    bcs = [f"S{i:04d}-1" for i in range(n)]
    counts = sp.csc_matrix(np.arange(1, n + 1, dtype=np.int32)[None, :])  # 1 feature x n barcodes
    with h5py.File(root / "S1_filtered_feature_bc_matrix.h5", "w") as f:
        f.attrs["library_ids"] = np.array([b"S1"])
        m = f.create_group("matrix")
        m["barcodes"] = np.array(bcs[: n - unmatched] + [f"X{i:04d}-1" for i in range(unmatched)], dtype="S")
        m["data"], m["indices"], m["indptr"] = counts.data, counts.indices, counts.indptr
        m["shape"] = np.array(counts.shape)
        g = m.create_group("features")
        g["id"] = g["name"] = np.array([b"G1"])
        g["feature_type"] = np.array([b"Gene Expression"])
        g["genome"] = np.array([b"GRCh38"])
        g["_all_tag_keys"] = np.array([b"genome"])
    rows = [f"{b},1,0,{2 * i},100,{100 + 50 * i}" for i, b in enumerate(bcs)]
    header = "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres"
    (root / "spatial" / "tissue_positions.csv").write_text("\n".join([header, *rows]) + "\n")
    return root


def test_join_above_min_barcode_match_keeps_and_records_unmatched(tmp_path: Path) -> None:
    sdata = visium(_many_spots(tmp_path / "S1", 200, 1), min_barcode_match=0.99)  # 199 of 200 = 99.5 %
    table = sdata["table"]
    assert table.n_obs == 200 and table.obs_names[-1] == "X0000-1"  # kept, not dropped
    assert np.isnan(table.obsm["spatial"][-1]).all() and not np.isnan(table.obsm["spatial"][:-1]).any()
    assert table.X.sum() == sum(range(1, 201))
    info = table.uns["visium_barcodes"]
    assert (info["match_rate"], info["min_barcode_match"], info["n_matched"]) == (0.995, 0.99, 199)
    assert info["unmatched"] == {"n": 1, "barcodes": ["X0000-1"]}
    assert len(sdata.shapes["S1"]) == 200
    sdata.write(tmp_path / "x.zarr")
    back = read_zarr(tmp_path / "x.zarr")["table"].uns["visium_barcodes"]["unmatched"]
    assert int(back["n"]) == 1 and list(back["barcodes"]) == ["X0000-1"]


def test_join_below_min_barcode_match_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"1 of 50 counts barcodes.*match rate 0.9800 < min_barcode_match 0.99"):
        visium(_many_spots(tmp_path / "S1", 50, 1), min_barcode_match=0.99)  # 49 of 50 = 98 %


def test_declared_rule_joins_the_positions(tmp_path: Path) -> None:
    deposited = [f"Injury72h_{b[:-2]}.1" for b in _BARCODES]
    sdata = visium(_respell(_space_ranger(tmp_path / "S1"), deposited), barcode_rule=_RULE)
    for t in sdata.tables.values():
        assert list(t.obs_names) == _BARCODES
        assert list(t.obs["barcode_deposited"]) == deposited
        assert t.uns["visium_barcodes"]["rule"] == _RULE
        assert t.uns["visium_barcodes"]["match_rate"] == 1.0
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
