"""Tests for dbit_matrix() on synthetic DBiT-seq / spatial-CITE-seq / xDBiT matrices."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from spatialdata import read_zarr

from spatialdata_io import dbit_matrix


def test_tsv_rna_and_protein_on_one_grid(tmp_path: Path) -> None:
    # RNA as spots x genes, protein as ADTs x spots with oligo suffixes; protein lacks one pixel
    pd.DataFrame({"G1": [1, 0, 2], "G2": [0, 3, 0]}, index=["1x1", "2x1", "1x2"]).to_csv(
        tmp_path / "rna.tsv.gz", sep="\t"
    )
    pd.DataFrame({"1x2": [5], "1x1": [7]}, index=["CD4.AACAAGACCCTTGAG"]).to_csv(tmp_path / "adt.tsv.gz", sep="\t")

    sdata = dbit_matrix(tmp_path / "rna.tsv.gz", protein=tmp_path / "adt.tsv.gz", pixel_width_um=10, pitch_um=20)

    rna, adt = sdata["table"], sdata["table_protein"]
    np.testing.assert_array_equal(rna.X.toarray(), [[1, 0], [0, 3], [2, 0]])
    assert list(adt.var_names) == ["CD4"] and list(adt.obs_names) == ["1x2", "1x1"]
    np.testing.assert_array_equal(adt.X.toarray(), [[5], [7]])
    np.testing.assert_array_equal(adt.obs["instance_id"], [2, 0])  # the RNA table's pixels
    assert sdata.shapes["spots"].geometry.iloc[2].bounds == (15, 35, 25, 45)  # pixel 1x2: centre (20, 40)
    sdata.write(tmp_path / "x.zarr")
    assert set(read_zarr(tmp_path / "x.zarr").tables) == {"table", "table_protein"}


def test_h5ad_with_um_positions(tmp_path: Path) -> None:
    obs = pd.DataFrame({"array_row": [1, 2], "array_col": [3, 3], "um_row": [100, 200], "um_col": [300, 300]})
    obs.index = ["3x1-A1", "3x2-A1"]
    a = ad.AnnData(np.array([[4.0, 0.0], [1.0, 2.0]], dtype=np.float32), obs=obs)
    a.uns["spatial"] = {"A1": {"images": {"hires": np.zeros((4, 4))}}}
    a.write_h5ad(tmp_path / "x.h5ad")

    sdata = dbit_matrix(tmp_path / "x.h5ad", pixel_width_um=50)

    table = sdata["table"]
    assert "spatial" not in table.uns and list(table.obs["array_x"]) == [3, 3]
    assert sdata.shapes["spots"].geometry.iloc[1].bounds == (275, 175, 325, 225)


def test_index_grid_needs_pitch(tmp_path: Path) -> None:
    pd.DataFrame({"G": [1]}, index=["1x1"]).to_csv(tmp_path / "r.csv")
    with pytest.raises(ValueError, match="pitch_um"):
        dbit_matrix(tmp_path / "r.csv", pixel_width_um=10)
