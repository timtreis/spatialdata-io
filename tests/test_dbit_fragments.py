"""Tests for dbit_fragments() on a synthetic DBiT deposit."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.sparse as sp
from PIL import Image
from spatialdata import read_zarr

from spatialdata_io import dbit_fragments

_BARCODES = ["AAAAAAAACCCCCCCC", "AAAAAAAAGGGGGGGG", "TTTTTTTTCCCCCCCC", "TTTTTTTTGGGGGGGG"]


def _spatial(root: Path, image: bool = True) -> Path:
    folder = root / "GSM1_spatial"
    folder.mkdir(parents=True)
    rows = [
        f"{b},{int(i != 3)},{i % 2},{i // 2},{100 + 30 * (i % 2)}, {200 + 30 * (i // 2)}"
        for i, b in enumerate(_BARCODES)
    ]
    (folder / "tissue_positions_list.csv").write_text("\n".join(rows) + "\n")
    (folder / "scalefactors_json.json").write_text(
        json.dumps({"spot_diameter_fullres": 15.0, "tissue_hires_scalef": 0.5, "tissue_lowres_scalef": 0.5})
    )
    if image:
        Image.fromarray(np.zeros((300, 400, 4), dtype=np.uint8)).save(folder / "tissue_hires_image.png")
    return root


def _fragments(path: Path) -> Path:
    lines = [
        "# id=x",
        "# reference_path=hg38",
        f"chr1\t100\t300\t{_BARCODES[0]}-1\t3",  # insertions 100, 299 -> tile 0 twice
        f"chr1\t4990\t5100\t{_BARCODES[0]}-1\t1",  # 4990 -> tile 0, 5099 -> tile 1
        f"chr2\t10\t50\t{_BARCODES[1]}-1\t1",  # chr2 tile 0 twice
        f"chrM\t10\t50\t{_BARCODES[1]}-1\t1",  # no feature: 2 insertions outside
        "chr1\t10\t50\tNOTINPOSITIONS-1\t1",  # unassigned
    ]
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    return path


def test_tiles_counts_shapes_image_and_round_trip(tmp_path: Path) -> None:
    root = _spatial(tmp_path)
    sdata = dbit_fragments(root, fragments=_fragments(tmp_path / "f.tsv.gz"), genome_build="hg38", pixel_width_um=50.0)

    table = sdata["table"]
    assert list(table.obs_names) == _BARCODES[:2]  # pixels with >= 1 fragment
    assert table.X.dtype == np.int32 and sp.issparse(table.X)
    assert table.var_names[0] == "chr1:0-5000"
    assert table.var.loc["chr1:5000-10000", "start"] == 5000
    chr2 = table.var.index.get_loc("chr2:0-5000")
    np.testing.assert_array_equal(table.X[0, :2].toarray(), [[3, 1]])
    assert table.X[1, chr2] == 2 and table.X.sum() == 6
    np.testing.assert_array_equal(table.obs["n_fragments"], [2, 2])
    uns = table.uns["spatialdata_io"]
    assert uns["feature_space"] == "tiles_5kb" and uns["counting"] == "tn5_insertions"
    assert (uns["n_unassigned"], uns["n_insertions_outside_features"], uns["n_fragments"]) == (1, 2, 4)
    assert uns["um_per_image_px"] == pytest.approx(50 / 15)
    assert uns["grid_step_px"] == pytest.approx(30)
    assert uns["centre_spacing_um"] == pytest.approx(100)
    assert uns["pitch_unknown"] is False

    square = sdata.shapes["epi_spots"].geometry.iloc[0]
    assert square.bounds == (192.5, 92.5, 207.5, 107.5)  # centre (pxl_x, pxl_y) = (200, 100), side 15
    assert sdata.images["tissue_hires"].shape == (3, 300, 400)
    assert sdata.coordinate_systems == ["global"]

    sdata.write(tmp_path / "x.zarr")
    read = read_zarr(tmp_path / "x.zarr")
    assert read["table"].uns["spatialdata_io"]["feature_space"] == "tiles_5kb"


def test_peaks_and_peak_matrix(tmp_path: Path) -> None:
    root = _spatial(tmp_path, image=False)
    peaks = tmp_path / "peaks.bed"
    peaks.write_text("track name=x\nchr1\t90\t110\tp1\nchr2\t0\t20\tp2\n")
    sdata = dbit_fragments(root, fragments=_fragments(tmp_path / "f.tsv.gz"), peaks=peaks, genome_build="hg38")
    table = sdata["table"]
    assert list(table.var_names) == ["chr1:90-110", "chr2:0-20"]
    np.testing.assert_array_equal(table.X.toarray(), [[1, 0], [0, 1]])
    assert table.uns["spatialdata_io"]["pitch_unknown"] is True
    assert not sdata.images

    h5 = tmp_path / "raw_peak_bc_matrix.h5"
    counts = sp.csc_matrix(np.array([[0, 2, 0], [1, 0, 0]]))  # features x barcodes
    with h5py.File(h5, "w") as f:
        m = f.create_group("matrix")
        m["shape"], m["data"], m["indices"], m["indptr"] = counts.shape, counts.data, counts.indices, counts.indptr
        m["barcodes"] = np.array([f"{_BARCODES[2]}-1", f"{_BARCODES[3]}-1", "NOTINPOSITIONS-1"], dtype="S")
        m.create_group("features")["id"] = np.array([b"chr1:1-2", b"chr2:3-4"])
        m["features"]["genome"] = np.array([b"hg38", b"hg38"])
    table = dbit_fragments(root, peak_matrix=h5, genome_build="hg38")["table"]
    assert list(table.obs_names) == _BARCODES[2:]
    np.testing.assert_array_equal(table.X.toarray(), [[0, 1], [2, 0]])
    assert table.uns["spatialdata_io"]["feature_space"] == "peaks_deposited"
    assert table.uns["spatialdata_io"]["n_unassigned"] == 1


def test_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        dbit_fragments(tmp_path, genome_build="hg38")
    with pytest.raises(ValueError, match="Unknown genome_build"):
        dbit_fragments(tmp_path, fragments=tmp_path / "f", genome_build="GRCh38")
    with pytest.raises(FileNotFoundError, match="tissue_positions_list"):
        dbit_fragments(tmp_path, fragments=tmp_path / "f", genome_build="hg38")
