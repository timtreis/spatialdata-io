from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from spatialdata import SpatialData, read_zarr
from spatialdata.transformations import Scale, get_transformation

from spatialdata_io import stereoseq, stereoseq_gef
from spatialdata_io.readers.stereoseq import _bin_name, _default_dataset_id, _read_bins, _resolution_nm

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
    cell_offsets: list[int] | None = None,
) -> Path:
    """Write a synthetic cellbin GEF. ``borders`` hold the valid (k, 2) offsets per cell; they are padded to 32.

    ``cell_offsets`` places each cell's ``cellExp`` rows at explicit offsets instead of contiguously in cell order.
    """
    n = len(cells)
    cell = np.zeros(n, dtype=CELL_DTYPE)
    offset = 0
    for i, c in enumerate(cells):
        if cell_offsets is not None:
            offset = cell_offsets[i]
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
    exp = np.zeros(sum(len(r) for r in cell_exp), dtype=[("geneID", "<u4"), ("count", "<u2")])
    for i, row in enumerate(cell_exp):
        start = int(cell["offset"][i])
        for j, e in enumerate(row):
            exp[start + j] = e
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


def test_cell_exp_noncontiguous_offsets(tmp_path: Path) -> None:
    square = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2]])
    path = write_cellbin(
        tmp_path / "GSM1_CHIP.cellbin.gef",
        cells=[{"x": 10, "y": 10}, {"x": 20, "y": 20}, {"x": 30, "y": 30}, {"x": 40, "y": 40}],
        borders=[square] * 4,
        cell_exp=[[(0, 3), (2, 4)], [], [(1, 5)], [(2, 6), (0, 7)]],
        genes=["G1", "G2", "G3"],
        attrs={"resolution": [500]},
        cell_offsets=[3, 5, 2, 0],
    )
    table = stereoseq_gef(cellbin_gef=path).tables["table"]
    np.testing.assert_array_equal(table.X.toarray(), [[3, 0, 4], [0, 0, 0], [0, 5, 0], [7, 0, 6]])


def test_polygonal_part_of_geometry_collection_kept(tmp_path: Path) -> None:
    # a spike makes make_valid return GeometryCollection[Polygon, LineString]
    spiked = np.array([[0, 0], [4, 0], [4, 4], [2, 4], [2, 6], [2, 4], [0, 4]])
    path = write_cellbin(
        tmp_path / "GSM1_CHIP.cellbin.gef",
        cells=[{"x": 10, "y": 20, "area": 4}],
        borders=[spiked],
        cell_exp=[[(0, 1)]],
        genes=["G1"],
        attrs={"resolution": [500]},
    )
    sdata = stereoseq_gef(cellbin_gef=path)
    assert sdata.tables["table"].uns["spatialdata_io"]["n_cells_fallback_geometry"] == 0
    geom = sdata.shapes["CHIP_cell_boundaries"].geometry.iloc[0]
    assert geom.geom_type == "Polygon"
    assert geom.area == pytest.approx(16)


GENES = ["G1", "G2", "G3"]
BIN1_ROWS = [
    ("G1", 0, 0, 1),
    ("G1", 1, 1, 2),
    ("G1", 2, 0, 4),
    ("G1", 5, 5, 1),
    ("G2", 1, 0, 3),
    ("G2", 3, 3, 5),
    ("G3", 0, 1, 7),
]
BIN2_ROWS = [("G1", 0, 0, 99), ("G1", 2, 2, 1), ("G2", 4, 4, 2)]


def _compound(
    rows: list[tuple[str, int, int, int]], genes: list[str], xy: str, count: str, gene_dtype: str
) -> tuple[np.ndarray, np.ndarray]:
    ordered = [r for g in genes for r in rows if r[0] == g]
    exp = np.zeros(len(ordered), dtype=[("x", xy), ("y", xy), ("count", count)])
    for i, (_, x, y, c) in enumerate(ordered):
        exp[i] = (x, y, c)
    gene = np.zeros(len(genes), dtype=[("gene", gene_dtype), ("offset", "<u4"), ("count", "<u4")])
    offset = 0
    for i, g in enumerate(genes):
        n = sum(r[0] == g for r in rows)
        gene[i] = (g.encode(), offset if n else 0, n)
        offset += n
    return exp, gene


def write_tissue_gef(
    path: Path,
    rows: list[tuple[str, int, int, int]],
    genes: list[str],
    attrs: dict[str, Any] | None = None,
    precomputed: dict[int, list[tuple[str, int, int, int]]] | None = None,
    *,
    exp_attrs: dict[str, Any] | None = None,
    xy_dtype: str = "<i4",
    gene_dtype: str = "S64",
    pre_count_dtype: str = "u1",
    extras: bool = False,
) -> Path:
    """Write a synthetic tissue/full GEF; ``rows`` are ``(gene, x, y, count)``."""
    with h5py.File(path, "w") as f:
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
        groups = {1: (rows, "u1")} | {b: (r, pre_count_dtype) for b, r in (precomputed or {}).items()}
        for b, (r, count_dtype) in groups.items():
            exp, gene = _compound(r, genes, xy_dtype, count_dtype, gene_dtype)
            g = f.create_group(f"geneExp/bin{b}")
            e = g.create_dataset("expression", data=exp)
            e.attrs["minX"], e.attrs["maxX"] = int(exp["x"].min()), int(exp["x"].max())
            e.attrs["minY"], e.attrs["maxY"] = int(exp["y"].min()), int(exp["y"].max())
            if b == 1:
                for k, v in (exp_attrs or {}).items():
                    e.attrs[k] = v
            g.create_dataset("gene", data=gene)
        if extras:
            junk = np.zeros(3, dtype=[("gene", "S64"), ("MIDcount", "<u4")])
            f.create_group("stat").create_dataset("gene", data=junk)
            f.create_group("wholeExp").create_dataset("bin1", data=np.zeros((2, 2), dtype="u1"))
            f.create_group("wholeExpExon").create_dataset("bin1", data=np.zeros((2, 2), dtype="u1"))
    return path


def _bins(path: Path, bin_size: int, **kwargs: Any) -> tuple[Any, Any, str, str]:
    with h5py.File(path, "r") as f:
        return _read_bins(f, bin_size=bin_size, dataset_id="CHIP", transform={}, **kwargs)


def _tissue(tmp_path: Path, name: str = "GSM1_CHIP.tissue.gef", **kwargs: Any) -> Path:
    kwargs.setdefault("attrs", {"resolution": [500]})
    return write_tissue_gef(tmp_path / name, BIN1_ROWS, GENES, **kwargs)


def test_bins_aggregated_from_bin1(tmp_path: Path) -> None:
    table, squares, kind, source = _bins(_tissue(tmp_path), 2)
    assert (kind, source) == ("bin1_only", "aggregated_from_bin1")
    assert list(table.var_names) == GENES
    assert sp.isspmatrix_csr(table.X) and table.X.dtype == np.int32
    np.testing.assert_array_equal(table.X.toarray(), [[3, 3, 7], [4, 0, 0], [0, 5, 0], [1, 0, 0]])
    assert list(table.obs["bin_id"]) == [0, 1, 2, 3]
    assert list(squares.index) == [0, 1, 2, 3]
    corners = [tuple(g.bounds[:2]) for g in squares.geometry]
    assert corners == [(0, 0), (2, 0), (2, 2), (4, 4)]
    assert all(g.geom_type == "Polygon" and g.area == 4 for g in squares.geometry)


def test_bins_precomputed_used(tmp_path: Path) -> None:
    path = _tissue(tmp_path, precomputed={2: BIN2_ROWS})
    table, squares, kind, source = _bins(path, 2)
    assert (kind, source) == ("full", "precomputed")
    np.testing.assert_array_equal(table.X.toarray(), [[99, 0, 0], [1, 0, 0], [0, 2, 0]])
    assert [tuple(g.bounds[:2]) for g in squares.geometry] == [(0, 0), (2, 2), (4, 4)]
    # a full file without the requested bin falls back to aggregating bin1
    table, _, kind, source = _bins(path, 4)
    assert (kind, source) == ("full", "aggregated_from_bin1")
    np.testing.assert_array_equal(table.X.toarray(), [[7, 8, 7], [1, 0, 0]])


def test_streaming_matches_naive(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(6)]
    cells = sorted({(int(x), int(y)) for x, y in rng.integers(0, 40, size=(300, 2))})
    rows = [(g, x, y, int(rng.integers(1, 9))) for g in genes for x, y in cells if rng.random() < 0.5]
    path = write_tissue_gef(tmp_path / "s.tissue.gef", rows, genes, {"resolution": [500]})
    df = pd.DataFrame(rows, columns=["gene", "x", "y", "count"])
    df["kx"], df["ky"] = df["x"] // 7, df["y"] // 7
    naive = df.pivot_table(index=["ky", "kx"], columns="gene", values="count", aggfunc="sum", fill_value=0)
    for chunk_rows in (2**24, 5, 1):
        table, squares, _, _ = _bins(path, 7, chunk_rows=chunk_rows)
        np.testing.assert_array_equal(table.X.toarray(), naive[genes].to_numpy())
        got = [(int(g.bounds[0]) // 7, int(g.bounds[1]) // 7) for g in squares.geometry]
        assert got == [(kx, ky) for ky, kx in naive.index]


def test_partial_edge_bin(tmp_path: Path) -> None:
    path = write_tissue_gef(tmp_path / "e.tissue.gef", [("G1", 5, 1, 1)], ["G1"], {"resolution": [500]})
    _, squares, _, _ = _bins(path, 2)
    assert squares.geometry.iloc[0].bounds == (4, 0, 6, 2)


def test_legacy_attr_and_dtype_variants(tmp_path: Path) -> None:
    pre = {2: BIN2_ROWS}
    ref = _tissue(tmp_path, "ref.tissue.gef", precomputed=pre)
    legacy = _tissue(
        tmp_path,
        "legacy.tissue.gef",
        attrs={"version": 1, "resolution": 500},
        exp_attrs={"resolution": np.array([500], dtype=np.uint32)},
        precomputed=pre,
        xy_dtype="<u4",
        gene_dtype="S32",
        pre_count_dtype="u2",
    )
    for b in (2, 4):
        t1, s1, k1, src1 = _bins(ref, b)
        t2, s2, k2, src2 = _bins(legacy, b)
        assert (k1, src1) == (k2, src2)
        assert list(t1.var_names) == list(t2.var_names)
        np.testing.assert_array_equal(t1.X.toarray(), t2.X.toarray())
        assert [g.bounds for g in s1.geometry] == [g.bounds for g in s2.geometry]


def test_tissue_only_names(tmp_path: Path) -> None:
    sdata = stereoseq_gef(gef=_tissue(tmp_path), bin_size=2)
    assert list(sdata.tables) == ["table"]
    assert list(sdata.shapes) == ["CHIP_square_001um"]
    table = sdata.tables["table"]
    meta = table.uns["spatialdata_io"]
    assert meta["n_cells_fallback_geometry"] is None
    assert (meta["gef_kind"], meta["source_bins"]) == ("bin1_only", "aggregated_from_bin1")
    assert (meta["bin_size_dnb"], meta["bin_size_um"], meta["resolution_nm"]) == (2, 1.0, 500)
    assert table.uns["spatialdata_attrs"]["instance_key"] == "bin_id"
    assert list(table.obs["region"].unique()) == ["CHIP_square_001um"]
    shapes = sdata.shapes["CHIP_square_001um"]
    assert list(shapes.index) == list(table.obs["bin_id"])
    scale = get_transformation(shapes, "global")
    assert isinstance(scale, Scale)
    np.testing.assert_allclose(scale.scale, [0.5, 0.5])


def test_bins_roundtrip(tmp_path: Path) -> None:
    sdata = stereoseq_gef(gef=_tissue(tmp_path), bin_size=2)
    sdata.write(tmp_path / "out.zarr")
    back = read_zarr(tmp_path / "out.zarr")
    assert back.tables["table"].shape == (4, 3)


def test_ignores_other_groups(tmp_path: Path) -> None:
    plain = _bins(_tissue(tmp_path, "a.tissue.gef"), 2)
    extra = _bins(_tissue(tmp_path, "b.tissue.gef", extras=True), 2)
    np.testing.assert_array_equal(plain[0].X.toarray(), extra[0].X.toarray())
    assert plain[2:] == extra[2:]


def test_tissue_resolution_conflict(tmp_path: Path) -> None:
    path = _tissue(tmp_path, attrs={"resolution": [500]}, exp_attrs={"resolution": [715]})
    with pytest.raises(ValueError, match="resolution"):
        stereoseq_gef(gef=path)


def test_tissue_resolution_from_bin1_header(tmp_path: Path) -> None:
    path = _tissue(tmp_path, attrs={}, exp_attrs={"resolution": [715]})
    meta = stereoseq_gef(gef=path, bin_size=2).tables["table"].uns["spatialdata_io"]
    assert (meta["resolution_nm"], meta["resolution_source"]) == (715, "header")


def test_gef_rejections(tmp_path: Path) -> None:
    other = tmp_path / "other.h5"
    with h5py.File(other, "w") as f:
        f.create_group("something")
    with pytest.raises(ValueError, match="GEF"):
        stereoseq_gef(gef=other)
