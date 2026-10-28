from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
import scipy.sparse as sp
from spatialdata import SpatialData, read_zarr
from spatialdata.transformations import Scale, get_transformation

from spatialdata_io import stereoseq, stereoseq_gef
from spatialdata_io.readers.stereoseq import _bin_name, _default_dataset_id, _resolution_nm

PAD = 32767

CELL_DTYPE = np.dtype(
    [
        ("id", "<u4"),
        ("x", "<i4"),
        ("y", "<i4"),
        ("offset", "<u4"),
        ("geneCount", "<u2"),
        ("expCount", "<u2"),
        ("dnbCount", "<u2"),
        ("area", "<u2"),
        ("cellTypeID", "<u2"),
        ("clusterID", "<u2"),
    ]
)


def write_cellbin(
    path: Path,
    cells: list[dict[str, Any]],
    borders: list[np.ndarray | None],
    cell_exp: list[list[tuple[int, int]]],
    genes: list[str],
    attrs: dict[str, Any] | None = None,
    gene_dtype: str = "S64",
) -> Path:
    """Write a synthetic cellbin GEF. ``borders`` hold the valid (k, 2) offsets per cell; they are padded to 32."""
    n = len(cells)
    cell = np.zeros(n, dtype=CELL_DTYPE)
    offset = 0
    for i, c in enumerate(cells):
        cell[i] = (
            i,
            c["x"],
            c["y"],
            offset,
            len(cell_exp[i]),
            sum(v for _, v in cell_exp[i]),
            c.get("dnbCount", 10),
            c.get("area", 4),
            0,
            c.get("clusterID", 1),
        )
        offset += len(cell_exp[i])
    border = np.full((n, 32, 2), PAD, dtype=np.int16)
    for i, b in enumerate(borders):
        if b is not None:
            border[i, : len(b)] = b
    flat = [e for row in cell_exp for e in row]
    exp = np.array(flat, dtype=[("geneID", "<u4"), ("count", "<u2")])
    gene = np.zeros(len(genes), dtype=[("geneName", gene_dtype), ("offset", "<u4"), ("cellCount", "<u4")])
    gene["geneName"] = [g.encode() for g in genes]
    with h5py.File(path, "w") as f:
        g = f.create_group("cellBin")
        g.create_dataset("cell", data=cell)
        g.create_dataset("cellBorder", data=border)
        g.create_dataset("cellExp", data=exp)
        g.create_dataset("gene", data=gene)
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
    return path


def _three_cells(tmp_path: Path, name: str = "GSM1_CHIP.cellbin.gef", attrs: Any = "default") -> Path:
    square = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]])
    if attrs == "default":
        attrs = {"resolution": np.array([500], dtype=np.uint32)}
    return write_cellbin(
        tmp_path / name,
        cells=[{"x": 100, "y": 200}, {"x": 300, "y": 400}, {"x": 500, "y": 600}],
        borders=[square, square, square],
        cell_exp=[[(0, 3), (1, 5)], [(1, 2)], [(0, 7)]],
        genes=["G1", "G2"],
        attrs=attrs,
    )


def test_cellbin_only_basic(tmp_path: Path) -> None:
    sdata = stereoseq_gef(cellbin_gef=_three_cells(tmp_path))
    table = sdata.tables["table"]
    assert table.shape == (3, 2)
    assert list(table.var_names) == ["G1", "G2"]
    assert sp.isspmatrix_csr(table.X)
    assert table.X.dtype == np.int32
    np.testing.assert_array_equal(table.X.toarray(), [[3, 5], [0, 2], [7, 0]])
    shapes = sdata.shapes["CHIP_cell_boundaries"]
    assert list(shapes.index) == list(table.obs["cell_id"])
    scale = get_transformation(shapes, "global")
    assert isinstance(scale, Scale)
    np.testing.assert_allclose(scale.scale, [0.5, 0.5])
    meta = table.uns["spatialdata_io"]
    assert meta["resolution_source"] == "header"
    assert meta["resolution_nm"] == 500
    assert meta["reader"] == "stereoseq_gef"
    assert meta["n_cells_fallback_geometry"] == 0
    assert meta["gef_kind"] is None and meta["source_bins"] is None
    assert table.uns["spatialdata_attrs"]["instance_key"] == "cell_id"
    assert table.uns["spatialdata_attrs"]["region_key"] == "region"
    for col in ["area", "dnbCount", "geneCount", "expCount", "clusterID", "cellTypeID"]:
        assert col in table.obs


def test_border_padding_and_fallback(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from spatialdata._logging import logger

    quad = np.array([[0, 0], [4, 0], [4, 4], [0, 4]])
    path = write_cellbin(
        tmp_path / "GSM1_CHIP.cellbin.gef",
        cells=[{"x": 10, "y": 20, "area": 16}, {"x": 50, "y": 60, "area": 9}, {"x": 80, "y": 90, "area": 0}],
        borders=[quad, np.array([[1, 1], [2, 2]]), np.array([[1, 1], [1, 1], [1, 1]])],
        cell_exp=[[(0, 1)], [(0, 1)], [(0, 1)]],
        genes=["G1"],
        attrs={"resolution": [500]},
    )
    old = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING):
            sdata = stereoseq_gef(cellbin_gef=path)
    finally:
        logger.propagate = old
    shapes = sdata.shapes["CHIP_cell_boundaries"]
    poly = shapes.geometry.iloc[0]
    coords = np.array(poly.exterior.coords)
    assert coords.max() < PAD
    assert set(map(tuple, coords[:-1])) == {(10, 20), (14, 20), (14, 24), (10, 24)}
    sq = shapes.geometry.iloc[1]
    assert sq.area == pytest.approx(9)
    assert sq.centroid.x == pytest.approx(50)
    assert sq.centroid.y == pytest.approx(60)
    assert shapes.geometry.iloc[2].area == pytest.approx(1)
    assert sdata.tables["table"].uns["spatialdata_io"]["n_cells_fallback_geometry"] == 2
    assert any("fallback" in r.message for r in caplog.records)


def test_resolution_default(tmp_path: Path) -> None:
    sdata = stereoseq_gef(cellbin_gef=_three_cells(tmp_path, attrs={}))
    meta = sdata.tables["table"].uns["spatialdata_io"]
    assert meta["resolution_nm"] == 500
    assert meta["resolution_source"] == "platform_default"


def test_resolution_scalar_and_other_value(tmp_path: Path) -> None:
    path = _three_cells(tmp_path, attrs={"resolution": 715})
    assert _resolution_nm(h5py.File(path, "r")) == (715, "header")


def test_resolution_conflict_across_files(tmp_path: Path) -> None:
    a = _three_cells(tmp_path, "a.cellbin.gef", {"resolution": [500]})
    b = _three_cells(tmp_path, "b.cellbin.gef", {"resolution": [715]})
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb, pytest.raises(ValueError, match="resolution"):
        _resolution_nm(fa, fb)


@pytest.mark.parametrize(
    ("bin_size", "resolution", "expected"),
    [(50, 500, "square_025um"), (20, 500, "square_010um"), (5, 500, "square_002p5um"), (100, 500, "square_050um")],
)
def test_bin_name(bin_size: int, resolution: int, expected: str) -> None:
    assert _bin_name(bin_size, resolution) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [("GSM8330060_B02009F6.cellbin.gef", "B02009F6"), ("B02009F6.adjusted.cellbin.gef", "B02009F6")],
)
def test_default_dataset_id(name: str, expected: str) -> None:
    assert _default_dataset_id(Path("/some/dir") / name) == expected


def test_rejections(tmp_path: Path) -> None:
    good = _three_cells(tmp_path)
    with pytest.raises(ValueError, match="gef"):
        stereoseq_gef()
    for bad in (0, -5, 2.5, True):
        with pytest.raises(ValueError, match="bin_size"):
            stereoseq_gef(cellbin_gef=good, bin_size=bad)  # type: ignore[arg-type]
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f.create_group("something")
    with pytest.raises(ValueError, match="GEF"):
        stereoseq_gef(cellbin_gef=other)
    nocell = tmp_path / "nocell.gef"
    with h5py.File(nocell, "w") as f:
        f.create_group("geneExp")
    with pytest.raises(ValueError, match="cellBin"):
        stereoseq_gef(cellbin_gef=nocell)


def test_gene_s32_and_u4_coordinates(tmp_path: Path) -> None:
    path = write_cellbin(
        tmp_path / "x.cellbin.gef",
        cells=[{"x": 1, "y": 2}],
        borders=[np.array([[0, 0], [2, 0], [2, 2]])],
        cell_exp=[[(0, 1)]],
        genes=["G1"],
        gene_dtype="S32",
    )
    sdata = stereoseq_gef(cellbin_gef=path)
    assert list(sdata.tables["table"].var_names) == ["G1"]


def test_cellbin_roundtrip(tmp_path: Path) -> None:
    sdata = stereoseq_gef(cellbin_gef=_three_cells(tmp_path))
    out = tmp_path / "out.zarr"
    sdata.write(out)
    back: SpatialData = read_zarr(out)
    assert list(back.shapes["CHIP_cell_boundaries"].index) == list(back.tables["table"].obs["cell_id"])


def test_saw_reader_untouched() -> None:
    assert str(inspect.signature(stereoseq)) == (
        "(path: 'str | Path', dataset_id: 'str | None' = None, read_square_bin: 'bool' = True, "
        "optional_tif: 'bool' = False, imread_kwargs: 'Mapping[str, Any]' = mappingproxy({}), "
        "image_models_kwargs: 'Mapping[str, Any]' = mappingproxy({})) -> 'SpatialData'"
    )
