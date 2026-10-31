"""The Antibody Capture table of visium() on a synthetic Space Ranger Gene and Protein output."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from PIL import Image
from spatialdata import read_zarr

from spatialdata_io import visium

_BARCODES = ["AAAC-1", "AAAG-1", "AAAT-1"]
_FEATURES = [("CD3", "Gene Expression"), ("G2", "Gene Expression"), ("CD3", "Antibody Capture")]


def _space_ranger(root: Path) -> Path:
    (root / "spatial").mkdir(parents=True)
    # features x barcodes, as in a 10x HDF5 matrix
    counts = sp.csc_matrix(np.array([[1, 0, 2], [0, 3, 0], [7, 8, 9]]))
    with h5py.File(root / "S1_filtered_feature_bc_matrix.h5", "w") as f:
        f.attrs["library_ids"] = np.array([b"S1"])
        m = f.create_group("matrix")
        m["barcodes"] = np.array(_BARCODES, dtype="S")
        m["data"], m["indices"], m["indptr"] = counts.data, counts.indices, counts.indptr
        m["shape"] = np.array(counts.shape)
        g = m.create_group("features")
        g["id"] = g["name"] = np.array([n for n, _ in _FEATURES], dtype="S")
        g["feature_type"] = np.array([t for _, t in _FEATURES], dtype="S")
        g["genome"] = np.array([b"GRCh38"] * len(_FEATURES))
        g["_all_tag_keys"] = np.array([b"genome"])
    positions = ["barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres"]
    positions += [f"{b},1,0,{i},100,{100 + 50 * i}" for i, b in enumerate(_BARCODES)]
    (root / "spatial" / "tissue_positions.csv").write_text("\n".join(positions) + "\n")
    scalefactors = {"spot_diameter_fullres": 40.0, "tissue_hires_scalef": 0.1, "tissue_lowres_scalef": 0.03}
    (root / "spatial" / "scalefactors_json.json").write_text(json.dumps(scalefactors))
    Image.fromarray(np.zeros((30, 40), dtype=np.uint8)).save(root / "spatial" / "tissue_hires_image.png")  # grayscale
    return root


def test_antibody_capture_is_a_second_table_on_the_same_spots(tmp_path: Path) -> None:
    sdata = visium(_space_ranger(tmp_path / "S1"))

    rna, protein = sdata["table"], sdata["table_protein"]
    # the antibody keeps its name although the gene CD3 has it too
    assert list(rna.var_names) == ["CD3", "G2"] and list(protein.var_names) == ["CD3"]
    np.testing.assert_array_equal(rna.X.toarray(), [[1, 0], [0, 3], [2, 0]])
    np.testing.assert_array_equal(protein.X.toarray(), [[7], [8], [9]])
    for t in (rna, protein):
        assert t.uns["spatialdata_attrs"]["region"] == "S1"
        assert list(t.obs["spot_id"]) == [0, 1, 2]
    assert sdata.images["S1_hires_image"].shape == (1, 30, 40)
    sdata.write(tmp_path / "x.zarr")
    assert set(read_zarr(tmp_path / "x.zarr").tables) == {"table", "table_protein"}


def test_gene_expression_only_has_no_protein_table(tmp_path: Path) -> None:
    sdata = visium(_space_ranger(tmp_path / "S1"), gex_only=True)
    assert set(sdata.tables) == {"table"}


def test_spot_diameter_from_spacing_without_scale_factors(tmp_path: Path) -> None:
    root = _space_ranger(tmp_path / "S1")
    (root / "spatial" / "scalefactors_json.json").unlink()  # GSE263617 deposits none
    sdata = visium(root)
    assert sdata.shapes["S1"]["radius"].iloc[0] == 0.55 * 50 / 2  # spots 50 px apart
    assert not sdata.images and set(sdata.coordinate_systems) == {"S1"}


def test_protein_file_in_either_orientation(tmp_path: Path) -> None:
    root = _space_ranger(tmp_path / "S1")
    # a SPOTS-style ADT table, antibodies x barcodes, missing one spot and with one barcode off the array
    pd.DataFrame({"AAAT-1": [3], "AAAC-1": [5], "TTTT-1": [9]}, index=["CD4"]).to_csv(root / "adt.csv.gz")
    sdata = visium(root, gex_only=True, protein_file="adt.csv.gz")
    protein = sdata["table_protein"]
    assert list(protein.obs_names) == ["AAAC-1", "AAAT-1"] and list(protein.obs["spot_id"]) == [0, 2]
    np.testing.assert_array_equal(protein.X.toarray(), [[5], [3]])
    with pytest.raises(ValueError, match="already holds Antibody Capture"):
        visium(root, protein_file="adt.csv.gz")
