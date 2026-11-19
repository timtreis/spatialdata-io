"""Tests for the Stereo-seq GEM reader: same bins as stereoseq_gef on the same rows, cells as DNB-square unions."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest
from spatialdata import read_zarr

from spatialdata_io import stereoseq_gef, stereoseq_gem
from tests.test_stereoseq_gef import BIN1_ROWS, GENES, write_tissue_gef

_HEADER = "#FileFormat=GEMv0.1\n#SortedBy=None\n#BinSize=1\n#STOmicsChip=CHIP\n#OffsetX=3\n#OffsetY=4\n"


def _write_gem(path: Path, rows: list[tuple], cells: bool = False) -> Path:
    cols = "geneID\tx\ty\tMIDCount\tExonCount" + ("\tCellID" if cells else "")
    lines = [_HEADER + cols] + ["\t".join(map(str, r)) for r in rows]
    text = "\n".join(lines) + "\n"
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)
    return path


@pytest.mark.parametrize("name", ["GSM1_CHIP.tissue.gem", "GSM1_CHIP.tissue.gem.gz"])
def test_bins_match_the_gef_reader_on_the_same_rows(tmp_path: Path, name: str) -> None:
    gef = write_tissue_gef(tmp_path / "GSM1_CHIP.tissue.gef", BIN1_ROWS, GENES, attrs={"resolution": [500]})
    # ExonCount: the count minus 1, so a bin can have MIDCount but no exon count
    gem = _write_gem(tmp_path / name, [(g, x, y, c, c - 1) for g, x, y, c in BIN1_ROWS])
    a, b = stereoseq_gef(gef, bin_size=2)["table"], stereoseq_gem(gem, bin_size=2)["table"]
    b = b[:, list(a.var_names)]
    np.testing.assert_array_equal(b.X.toarray(), a.X.toarray())
    np.testing.assert_array_equal(b.obsm["spatial"], a.obsm["spatial"])
    assert (b.obs[["array_row", "array_col"]].to_numpy() == a.obs[["array_row", "array_col"]].to_numpy()).all()
    assert b.X.sum() == sum(c for *_, c in BIN1_ROWS)  # every MIDCount kept
    assert b.layers["exon"].sum() == sum(c - 1 for *_, c in BIN1_ROWS)  # and every ExonCount
    uns = b.uns["spatialdata_io"]
    assert (uns["reader"], uns["resolution_nm"], uns["resolution_source"]) == ("stereoseq_gem", 500, "platform_default")
    assert uns["gem_header"]["OffsetX"] == "3" and uns["gem_header"]["STOmicsChip"] == "CHIP"


def test_cells_are_the_convex_hull_of_their_dnb_squares(tmp_path: Path) -> None:
    rows = [  # cell 7: an L of 3 DNBs (gene A twice on one DNB); cell 9: one DNB
        ("A", 10, 10, 2, 1, 7),
        ("B", 10, 10, 1, 1, 7),
        ("A", 11, 10, 3, 3, 7),
        ("A", 10, 11, 1, 0, 7),
        ("B", 20, 20, 5, 5, 9),
    ]
    sdata = stereoseq_gem(cellbin_gem=_write_gem(tmp_path / "GSM1_CHIP.cellbin.gem", rows, cells=True))
    table, cells = sdata["table"], sdata["CHIP_cell_boundaries"]
    assert list(table.obs["cell_id"]) == [7, 9] and list(table.obs["dnbCount"]) == [3, 1]
    np.testing.assert_array_equal(table[:, ["A", "B"]].X.toarray(), [[6, 1], [0, 5]])
    np.testing.assert_array_equal(table[:, ["A", "B"]].layers["exon"].toarray(), [[4, 1], [0, 5]])
    # the L's convex hull: its 2 x 2 bounding square minus the half square at the free corner
    assert [g.area for g in cells.geometry] == [3.5, 1.0] and set(cells.geom_type) == {"Polygon"}
    assert table.uns["spatialdata_io"]["cell_geometry"] == "convex hull of DNB squares"
    np.testing.assert_allclose(table.obsm["spatial"][0], [10 + 1 / 3 + 0.5, 10 + 1 / 3 + 0.5])
    sdata.write(tmp_path / "c.zarr")
    assert read_zarr(tmp_path / "c.zarr")["table"].n_obs == 2


def test_rejections(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        stereoseq_gem()
    gem = _write_gem(tmp_path / "GSM1_CHIP.tissue.gem", [("A", 1, 1, 1, 1)])
    with pytest.raises(ValueError, match="no CellID"):
        stereoseq_gem(cellbin_gem=gem)
