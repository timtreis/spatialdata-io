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
    with h5py.File(path, "r") as f:
        assert _resolution_nm(f) == (715, "header")


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


def test_gene_s32_names(tmp_path: Path) -> None:
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
# bin2 x/y are DNB-unit corners, multiples of 2 (as in real full .gef files, e.g. GSE315416 bin50/bin100)
BIN2_ROWS = [("G1", 0, 0, 99), ("G1", 2, 2, 1), ("G2", 4, 4, 2)]


def _compound(
    rows: list[tuple[str, int, int, int]],
    genes: list[str],
    xy: str,
    count: str,
    gene_dtype: str,
    row_order: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    layout = row_order or genes
    ordered = [r for g in layout for r in rows if r[0] == g]
    exp = np.zeros(len(ordered), dtype=[("x", xy), ("y", xy), ("count", count)])
    for i, (_, x, y, c) in enumerate(ordered):
        exp[i] = (x, y, c)
    gene = np.zeros(len(genes), dtype=[("gene", gene_dtype), ("offset", "<u4"), ("count", "<u4")])
    offset = {}
    pos = 0
    for g in layout:
        offset[g] = pos
        pos += sum(r[0] == g for r in rows)
    for i, g in enumerate(genes):
        n = sum(r[0] == g for r in rows)
        gene[i] = (g.encode(), offset.get(g, 0) if n else 0, n)
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
    row_order: list[str] | None = None,
) -> Path:
    """Write a synthetic tissue/full GEF; ``rows`` are ``(gene, x, y, count)``."""
    with h5py.File(path, "w") as f:
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
        groups = {1: (rows, "u1")} | {b: (r, pre_count_dtype) for b, r in (precomputed or {}).items()}
        for b, (r, count_dtype) in groups.items():
            exp, gene = _compound(r, genes, xy_dtype, count_dtype, gene_dtype, row_order if b == 1 else None)
            g = f.create_group(f"geneExp/bin{b}")
            e = g.create_dataset("expression", data=exp)
            if len(exp):
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


def test_zero_count_gene_and_nonmonotone_offsets(tmp_path: Path) -> None:
    genes = ["G1", "G0", "G2", "G3"]  # G0 has no rows; rows are laid out G3, G2, G1 (offsets decrease)
    path = write_tissue_gef(
        tmp_path / "o.tissue.gef", BIN1_ROWS, genes, {"resolution": [500]}, row_order=["G3", "G2", "G1"]
    )
    table, _, _, _ = _bins(path, 2, chunk_rows=3)
    assert list(table.var_names) == genes
    np.testing.assert_array_equal(table.X.toarray(), [[3, 0, 3, 7], [4, 0, 0, 0], [0, 0, 5, 0], [1, 0, 0, 0]])


def test_empty_expression_raises(tmp_path: Path) -> None:
    path = write_tissue_gef(tmp_path / "z.tissue.gef", [], ["G1"], {"resolution": [500]})
    with pytest.raises(ValueError, match="no rows"):
        _bins(path, 2)


def test_negative_coordinates_raise(tmp_path: Path) -> None:
    path = write_tissue_gef(tmp_path / "n.tissue.gef", [("G1", -1, 0, 1)], ["G1"], {"resolution": [500]})
    with pytest.raises(ValueError, match="Negative"):
        _bins(path, 2)


def _pair(tmp_path: Path, cell_attrs: Any = "default", tissue_attrs: Any = None) -> tuple[Path, Path]:
    cellbin = _three_cells(tmp_path, "GSM1_CELLCHIP.cellbin.gef", cell_attrs)
    tissue = _tissue(tmp_path, "GSM2_TISSCHIP.tissue.gef", **({} if tissue_attrs is None else {"attrs": tissue_attrs}))
    return tissue, cellbin


def test_pair_names_and_tables(tmp_path: Path) -> None:
    tissue, cellbin = _pair(tmp_path)
    sdata = stereoseq_gef(gef=tissue, cellbin_gef=cellbin, dataset_id="CHIP")
    assert set(sdata.tables) == {"table", "square_025um"}
    assert set(sdata.shapes) == {"CHIP_cell_boundaries", "CHIP_square_025um"}
    for table_name, region, key in [
        ("table", "CHIP_cell_boundaries", "cell_id"),
        ("square_025um", "CHIP_square_025um", "bin_id"),
    ]:
        table = sdata.tables[table_name]
        attrs = table.uns["spatialdata_attrs"]
        assert attrs["region"] == region
        assert attrs["instance_key"] == key
        assert list(table.obs["region"].unique()) == [region]
        assert list(table.obs[key]) == list(sdata.shapes[region].index)
    cells_meta = sdata.tables["table"].uns["spatialdata_io"]
    bins_meta = sdata.tables["square_025um"].uns["spatialdata_io"]
    assert (cells_meta["gef_kind"], cells_meta["source_bins"]) == (None, None)
    assert cells_meta["n_cells_fallback_geometry"] == 0
    assert bins_meta["n_cells_fallback_geometry"] is None
    assert (bins_meta["gef_kind"], bins_meta["source_bins"]) == ("bin1_only", "aggregated_from_bin1")
    for meta in (cells_meta, bins_meta):
        assert (meta["resolution_nm"], meta["resolution_source"], meta["bin_size_dnb"]) == (500, "header", 50)
        assert meta["bin_size_um"] == 25.0


def test_pair_gene_lists_independent(tmp_path: Path) -> None:
    tissue, cellbin = _pair(tmp_path)
    sdata = stereoseq_gef(gef=tissue, cellbin_gef=cellbin)
    assert list(sdata.tables["table"].var_names) == ["G1", "G2"]
    assert list(sdata.tables["square_025um"].var_names) == GENES
    assert list(sdata.tables["table"].var_names) != list(sdata.tables["square_025um"].var_names)


def test_pair_resolution_conflict(tmp_path: Path) -> None:
    tissue, cellbin = _pair(tmp_path, cell_attrs={"resolution": [715]}, tissue_attrs={"resolution": [500]})
    with pytest.raises(ValueError, match="resolution"):
        stereoseq_gef(gef=tissue, cellbin_gef=cellbin)


def test_pair_resolution_shared_when_one_states_it(tmp_path: Path) -> None:
    tissue, cellbin = _pair(tmp_path, cell_attrs={}, tissue_attrs={"resolution": [715]})
    sdata = stereoseq_gef(gef=tissue, cellbin_gef=cellbin, bin_size=20)
    assert set(sdata.tables) == {"table", "square_014p3um"}
    for t in sdata.tables.values():
        assert t.uns["spatialdata_io"]["resolution_nm"] == 715
        assert t.uns["spatialdata_io"]["resolution_source"] == "header"


def test_pair_roundtrip(tmp_path: Path) -> None:
    tissue, cellbin = _pair(tmp_path)
    sdata = stereoseq_gef(gef=tissue, cellbin_gef=cellbin)
    sdata.write(tmp_path / "pair.zarr")
    back = read_zarr(tmp_path / "pair.zarr")
    assert back.tables["table"].shape == (3, 2)
    assert back.tables["square_025um"].n_obs == sdata.tables["square_025um"].n_obs
    assert set(back.shapes) == {"TISSCHIP_cell_boundaries", "TISSCHIP_square_025um"}
    for name, key in [("table", "cell_id"), ("square_025um", "bin_id")]:
        region = back.tables[name].uns["spatialdata_attrs"]["region"]
        assert list(back.tables[name].obs[key]) == list(back.shapes[region].index)


def test_dataset_id_mismatch_ok(tmp_path: Path) -> None:
    tissue, cellbin = _pair(tmp_path)
    sdata = stereoseq_gef(gef=tissue, cellbin_gef=cellbin)
    assert set(sdata.shapes) == {"TISSCHIP_cell_boundaries", "TISSCHIP_square_025um"}
    sdata = stereoseq_gef(gef=tissue, cellbin_gef=cellbin, dataset_id="X")
    assert set(sdata.shapes) == {"X_cell_boundaries", "X_square_025um"}


def _centres(shapes: Any) -> np.ndarray:
    return np.stack([shapes.geometry.centroid.x, shapes.geometry.centroid.y], axis=1)


def test_obsm_spatial_cells(tmp_path: Path) -> None:
    sdata = stereoseq_gef(cellbin_gef=_three_cells(tmp_path))
    table = sdata.tables["table"]
    np.testing.assert_allclose(table.obsm["spatial"], [[100, 200], [300, 400], [500, 600]])
    np.testing.assert_allclose(table.obsm["spatial"], _centres(sdata.shapes["CHIP_cell_boundaries"]))


def test_obsm_spatial_and_array_index_bins(tmp_path: Path) -> None:
    sdata = stereoseq_gef(gef=_tissue(tmp_path), bin_size=2)
    table = sdata.tables["table"]
    np.testing.assert_allclose(table.obsm["spatial"], _centres(sdata.shapes["CHIP_square_001um"]))
    np.testing.assert_allclose(table.obsm["spatial"], [[1, 1], [3, 1], [3, 3], [5, 5]])
    assert list(table.obs["array_row"]) == [0, 0, 1, 2]
    assert list(table.obs["array_col"]) == [0, 1, 1, 2]


def test_pair_obsm_and_shared_scale(tmp_path: Path) -> None:
    tissue, cellbin = _pair(tmp_path)
    sdata = stereoseq_gef(gef=tissue, cellbin_gef=cellbin)
    np.testing.assert_allclose(
        sdata.tables["table"].obsm["spatial"], _centres(sdata.shapes["TISSCHIP_cell_boundaries"])
    )
    np.testing.assert_allclose(
        sdata.tables["square_025um"].obsm["spatial"], _centres(sdata.shapes["TISSCHIP_square_025um"])
    )
    scales = [get_transformation(s, "global") for s in sdata.shapes.values()]
    assert all(isinstance(s, Scale) for s in scales)
    np.testing.assert_allclose(scales[0].scale, [0.5, 0.5])  # type: ignore[union-attr]
    np.testing.assert_allclose(scales[0].scale, scales[1].scale)  # type: ignore[union-attr]


def test_precomputed_misaligned_falls_back(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from spatialdata._logging import logger

    bad = [("G1", 1, 1, 99), ("G2", 3, 5, 2)]  # not multiples of 2
    path = _tissue(tmp_path, precomputed={2: bad})
    old = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING):
            table, squares, kind, source = _bins(path, 2)
    finally:
        logger.propagate = old
    assert (kind, source) == ("full", "aggregated_from_bin1")
    assert any("multiple" in r.message for r in caplog.records)
    ref, ref_squares, _, _ = _bins(_tissue(tmp_path, "ref.tissue.gef"), 2)
    np.testing.assert_array_equal(table.X.toarray(), ref.X.toarray())
    assert [g.bounds for g in squares.geometry] == [g.bounds for g in ref_squares.geometry]


def test_precomputed_gene_list_differs_from_bin1(tmp_path: Path) -> None:
    genes2 = ["G2", "G1"]
    exp, gene = _compound(BIN2_ROWS[:2] + [("G2", 4, 4, 2)], genes2, "<i4", "u1", "S64")
    path = _tissue(tmp_path)
    with h5py.File(path, "a") as f:
        g = f.create_group("geneExp/bin2")
        g.create_dataset("expression", data=exp)
        g.create_dataset("gene", data=gene)
    table, _, _, source = _bins(path, 2)
    assert source == "precomputed"
    assert list(table.var_names) == genes2
    np.testing.assert_array_equal(table.X.toarray(), [[0, 99], [0, 1], [2, 0]])


def test_dataset_id_validated_up_front(tmp_path: Path) -> None:
    path = _three_cells(tmp_path, "a b (1).gef")
    with pytest.raises(ValueError, match=r"dataset_id.*a b \(1\)|a b \(1\).*dataset_id"):
        stereoseq_gef(cellbin_gef=path)
    assert "table" in stereoseq_gef(cellbin_gef=path, dataset_id="ok").tables


def test_gef_given_cellbin_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cellbin_gef"):
        stereoseq_gef(gef=_three_cells(tmp_path))


def test_border_pad_mid_row_uses_leading_points(tmp_path: Path) -> None:
    border = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2], [PAD, PAD], [9, 9], [8, 8]])
    path = write_cellbin(
        tmp_path / "GSM1_CHIP.cellbin.gef",
        cells=[{"x": 10, "y": 20, "area": 16}],
        borders=[border],
        cell_exp=[[(0, 1)]],
        genes=["G1"],
        attrs={"resolution": [500]},
    )
    geom = stereoseq_gef(cellbin_gef=path).shapes["CHIP_cell_boundaries"].geometry.iloc[0]
    assert geom.area == pytest.approx(16)
    assert geom.bounds == (8, 18, 12, 22)


def test_border_all_valid_row(tmp_path: Path) -> None:
    ang = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    border = np.round(np.stack([10 * np.cos(ang), 10 * np.sin(ang)], axis=1)).astype(int)
    path = write_cellbin(
        tmp_path / "GSM1_CHIP.cellbin.gef",
        cells=[{"x": 10, "y": 20}],
        borders=[border],
        cell_exp=[[(0, 1)]],
        genes=["G1"],
        attrs={"resolution": [500]},
    )
    sdata = stereoseq_gef(cellbin_gef=path)
    assert sdata.tables["table"].uns["spatialdata_io"]["n_cells_fallback_geometry"] == 0


def test_resolution_invalid(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resolution"):
        stereoseq_gef(cellbin_gef=_three_cells(tmp_path, attrs={"resolution": 0}))
    multi = _three_cells(tmp_path, "m.cellbin.gef", {"resolution": np.array([500, 500])})
    with pytest.raises(ValueError, match="resolution"):
        stereoseq_gef(cellbin_gef=multi)


def test_zero_counts_eliminated(tmp_path: Path) -> None:
    rows = [("G1", 0, 0, 1), ("G1", 4, 4, 0), ("G2", 4, 4, 0), ("G2", 2, 0, 3)]
    path = write_tissue_gef(tmp_path / "z.tissue.gef", rows, ["G1", "G2"], {"resolution": [500]})
    table, squares, _, _ = _bins(path, 2)
    np.testing.assert_array_equal(table.X.toarray(), [[1, 0], [0, 3]])
    assert table.X.nnz == 2
    assert len(squares) == 2 and list(table.obs["bin_id"]) == [0, 1]
    cells = write_cellbin(
        tmp_path / "GSM1_CHIP.cellbin.gef",
        cells=[{"x": 1, "y": 1}],
        borders=[np.array([[0, 0], [2, 0], [2, 2]])],
        cell_exp=[[(0, 0), (1, 4)]],
        genes=["G1", "G2"],
        attrs={"resolution": [500]},
    )
    assert stereoseq_gef(cellbin_gef=cells).tables["table"].X.nnz == 1


def test_uns_roundtrip(tmp_path: Path) -> None:
    sdata = stereoseq_gef(cellbin_gef=_three_cells(tmp_path))
    before = dict(sdata.tables["table"].uns["spatialdata_io"])
    sdata.write(tmp_path / "u.zarr")
    after = read_zarr(tmp_path / "u.zarr").tables["table"].uns["spatialdata_io"]
    assert after.keys() == before.keys()
    for k, v in before.items():
        if v is not None:  # None values may not survive zarr; documented by the key check above
            assert after[k] == v


def test_omics_header_recorded(tmp_path: Path) -> None:
    adt = _tissue(tmp_path, "GSM1_CHIP_Stereo-CITE_ADT.tissue.gef", attrs={"omics": [b"Proteomics"], "resolution": 500})
    assert stereoseq_gef(gef=adt, bin_size=2)["table"].uns["spatialdata_io"]["omics"] == "Proteomics"
    assert stereoseq_gef(gef=_tissue(tmp_path), bin_size=2)["table"].uns["spatialdata_io"]["omics"] is None


def test_gene_id_and_name_fields(tmp_path: Path) -> None:
    ref = _tissue(tmp_path)
    new = _tissue(tmp_path, "new.tissue.gef")
    with h5py.File(new, "r+") as f:
        for name in list(f["geneExp"]):
            old = f[f"geneExp/{name}/gene"][:]
            gene = np.zeros(
                len(old), dtype=[("geneID", "S64"), ("geneName", "S64"), ("offset", "<u4"), ("count", "<u4")]
            )
            gene["geneID"], gene["geneName"] = [b"ID" + g for g in old["gene"]], old["gene"]
            gene["offset"], gene["count"] = old["offset"], old["count"]
            del f[f"geneExp/{name}/gene"]
            f[f"geneExp/{name}"].create_dataset("gene", data=gene)
    t1, *_ = _bins(ref, 2)
    t2, *_ = _bins(new, 2)
    assert list(t1.var_names) == list(t2.var_names)
    np.testing.assert_array_equal(t1.X.toarray(), t2.X.toarray())
