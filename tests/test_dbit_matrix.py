"""Tests for dbit_matrix() on synthetic DBiT-seq / spatial-CITE-seq / xDBiT matrices."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
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
    for t in (rna, adt):
        uns = t.uns["spatialdata_io"]
        assert (uns["observation_unit"], uns["unit_size_um"]) == ("spot_or_bin", 10.0)
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
    assert table.uns["spatialdata_io"]["pitch_unknown"] is False
    assert sdata.shapes["spots"].geometry.iloc[1].bounds == (275, 175, 325, 225)


def test_index_grid_without_pitch_is_in_grid_units(tmp_path: Path) -> None:
    pd.DataFrame({"G": [1, 2]}, index=["1x1", "3x2"]).to_csv(tmp_path / "r.csv")
    sdata = dbit_matrix(tmp_path / "r.csv", pixel_width_um=25)
    uns = sdata["table"].uns["spatialdata_io"]
    assert uns["pitch_unknown"] is True and uns["pixel_width_um"] == 25  # metadata only, no scale derived
    assert sdata.shapes["spots"].geometry.iloc[1].bounds == (2.5, 1.5, 3.5, 2.5)


def test_mtx_folder_from_seurat_rds(tmp_path: Path) -> None:
    import scipy.io
    import scipy.sparse as sp

    d = tmp_path / "rds"
    d.mkdir()
    scipy.io.mmwrite(d / "matrix.mtx", sp.coo_matrix(np.array([[1, 0], [2, 3], [0, 4]])))  # 3 features x 2 spots
    (d / "features.tsv").write_text("name\tassay\nG1\tSpatial\nmir1\tmicroRNA\nG1\tlncRNA\n")
    (d / "barcodes.tsv").write_text("31x35\n18x41\n")
    # the explicit positions win over the spot names: "31x35" is row 31, column 35
    (d / "positions.csv").write_text("barcode,array_row,array_col\n18x41,18,41\n31x35,31,35\n")

    sdata = dbit_matrix(d, pixel_width_um=50)

    table = sdata["table"]
    assert list(table.var_names) == ["G1", "mir1", "G1-1"] and list(table.var["assay"]) == [
        "Spatial",
        "microRNA",
        "lncRNA",
    ]
    np.testing.assert_array_equal(table.X.toarray(), [[1, 2, 0], [0, 3, 4]])
    assert list(table.obs["array_x"]) == [35, 41] and list(table.obs["array_y"]) == [31, 18]
