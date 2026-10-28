from __future__ import annotations

import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse as sp
from spatialdata import SpatialData
from spatialdata.transformations import Identity, Scale, get_transformation

from spatialdata_io import cell_table

HEADER = ",fov,volume,center_x,center_y,min_x,GeneA,GeneB,Blank-1"
ROWS = [
    "c1,0,523.6,10.5,20.5,9,1,2,0",
    "c2,0,904.8,30.0,40.0,29,3,0,1",
    "c3,1,113.1,50.25,60.75,49,0,5,2",
]


def _write(path: Path, rows: list[str] = ROWS, header: str = HEADER, junk: bool = False) -> Path:
    lines = ([] if not junk else ["junk title"]) + [header, *rows]
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def csv(tmp_path: Path) -> Path:
    return _write(tmp_path / "counts.csv")


def test_wide_csv_basic(csv: Path) -> None:
    sdata = cell_table(csv, unit="um")
    table = sdata.tables["table"]
    assert table.shape == (3, 2)
    assert list(table.var_names) == ["GeneA", "GeneB"]
    assert table.obsm["blank"].shape == (3, 1)
    assert "cells" in sdata.shapes
    assert isinstance(get_transformation(sdata["cells"], "global"), Identity)
    assert list(table.obs.index) == ["c1", "c2", "c3"]
    assert list(table.obs["cell_id"]) == [0, 1, 2]
    assert list(sdata["cells"].index) == [0, 1, 2]
    assert set(table.obs["region"]) == {"cells"}


def test_unit_mm_scale(csv: Path) -> None:
    sdata = cell_table(csv, unit="mm")
    tf = get_transformation(sdata["cells"], "global")
    assert isinstance(tf, Scale)
    assert list(tf.scale) == [1000, 1000]
    assert list(sdata["cells"].geometry.x) == [10.5, 30.0, 50.25]

    sdata = cell_table(csv, unit="px", pixel_size=0.108)
    tf = get_transformation(sdata["cells"], "global")
    assert isinstance(tf, Scale)
    assert list(tf.scale) == [0.108, 0.108]
    assert list(sdata["cells"].geometry.x) == [10.5, 30.0, 50.25]


def test_unit_required(csv: Path) -> None:
    with pytest.raises(TypeError):
        cell_table(csv)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="pixel_size"):
        cell_table(csv, unit="px")
    with pytest.raises(ValueError, match="inch"):
        cell_table(csv, unit="inch")  # type: ignore[arg-type]


def test_normalized_counts_rejected(tmp_path: Path) -> None:
    rows = [ROWS[0].replace(",1,2,0", ",0.5,2,0"), *ROWS[1:]]
    with pytest.raises(ValueError, match="non-integer"):
        cell_table(_write(tmp_path / "n.csv", rows), unit="um")


def test_integral_floats_accepted(tmp_path: Path) -> None:
    rows = [r.replace(",1,2,0", ",3.0,2.0,0") for r in ROWS[:1]] + ROWS[1:]
    sdata = cell_table(_write(tmp_path / "f.csv", rows), unit="um")
    assert sdata.tables["table"].X[0, 0] == 3


def test_missing_xy_rejected(csv: Path) -> None:
    with pytest.raises(ValueError, match="nope"):
        cell_table(csv, unit="um", x="nope")


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    rows = [ROWS[0], ROWS[0], ROWS[2]]
    with pytest.raises(ValueError, match="duplicate"):
        cell_table(_write(tmp_path / "d.csv", rows), unit="um")


def test_radius_sources(csv: Path) -> None:
    sdata = cell_table(csv, unit="um")
    radius = sdata.tables["table"].uns["spatialdata_io"]["radius"]
    assert radius["source"] == "derived:volume"
    assert len(radius["value_um"]) == 3
    assert np.all(np.asarray(radius["value_um"]) > 0)
    assert np.allclose(radius["value_um"][0], (3 * 523.6 / (4 * np.pi)) ** (1 / 3))
    assert np.allclose(sdata["cells"]["radius"].to_numpy(), radius["value_um"])

    sdata = cell_table(csv, unit="um", cell_radius=7.0)
    assert sdata.tables["table"].uns["spatialdata_io"]["radius"] == {"source": "stated", "value_um": 7.0}

    sdata = cell_table(csv, unit="mm")
    assert sdata.tables["table"].uns["spatialdata_io"]["radius"] == {"source": "default", "value_um": 5.0}
    assert np.allclose(sdata["cells"]["radius"].to_numpy(), 5.0 / 1000)


def test_radius_from_area(tmp_path: Path) -> None:
    rows = [r.replace(",523.6,", ",78.5398,", 1) for r in ROWS[:1]] + ROWS[1:]
    header = HEADER.replace("volume", "area")
    sdata = cell_table(
        _write(tmp_path / "a.csv", rows, header), unit="um", metadata_columns=["area", "center_x", "center_y"]
    )
    radius = sdata.tables["table"].uns["spatialdata_io"]["radius"]
    assert radius["source"] == "derived:area"
    assert np.isclose(radius["value_um"][0], 5.0, atol=1e-3)


def test_uns_metadata(csv: Path) -> None:
    uns = cell_table(csv, unit="px", pixel_size=0.108).tables["table"].uns["spatialdata_io"]
    assert uns["reader"] == "cell_table"
    assert uns["coordinates"] == {"x": "center_x", "y": "center_y", "unit": "px", "pixel_size": 0.108}
    assert uns["radius"]["source"] == "default"
    assert uns["n_coordinates_unmatched"] == 0


def test_skiprows(tmp_path: Path, csv: Path) -> None:
    junk = _write(tmp_path / "junk.csv", junk=True)
    a = cell_table(csv, unit="um").tables["table"]
    b = cell_table(junk, unit="um", skiprows=1).tables["table"]
    assert a.shape == b.shape
    assert list(a.obs.index) == list(b.obs.index)
    assert np.array_equal(a.X, b.X)


def test_metadata_columns_override(csv: Path) -> None:
    default = cell_table(csv, unit="um").tables["table"]
    assert "min_x" in default.obs.columns
    sdata = cell_table(csv, unit="um", metadata_columns=["fov", "volume", "center_x", "center_y"])
    table = sdata.tables["table"]
    assert "min_x" not in table.obs.columns
    assert "min_x" in table.var_names


def test_other_layouts_not_implemented(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        cell_table(tmp_path / "x.h5ad", unit="um")


def test_roundtrip(csv: Path, tmp_path: Path) -> None:
    sdata = cell_table(csv, unit="um")
    p = tmp_path / "out.zarr"
    sdata.write(p)
    back = SpatialData.read(p)
    assert back.tables["table"].shape == (3, 2)
    assert back.tables["table"].uns["spatialdata_io"]["coordinates"]["unit"] == "um"


def test_build_sparse() -> None:
    import pandas as pd
    import scipy.sparse as sp

    from spatialdata_io.readers.cell_table import _build

    obs = pd.DataFrame({"center_x": [1.0, 2.0], "center_y": [3.0, 4.0]}, index=["a", "b"])
    X = sp.csr_matrix(np.array([[1, 0, 2], [0, 3, 0]]))
    sdata = _build(
        X,
        pd.Index(["G1", "G2", "BLANK_1"]),
        obs,
        unit="um",
        pixel_size=None,
        x="center_x",
        y="center_y",
        split_by=None,
        cell_radius=None,
        n_coordinates_unmatched=0,
    )
    table = sdata.tables["table"]
    assert table.shape == (2, 2)
    assert table.obsm["blank"].shape == (2, 1)
    assert sp.issparse(table.X)


GENES = ["GeneA", "GeneB", "Blank-1"]


def _split_files(
    tmp_path: Path,
    n: int = 20,
    extra: int = 0,
    *,
    genes_as_rows: bool = True,
    sep: str = "\t",
    ids: list[str] | None = None,
) -> tuple[Path, Path, np.ndarray]:
    ids = ids or [f"c{i}" for i in range(n)]
    rng = np.random.default_rng(0)
    mat = rng.integers(0, 5, size=(n, len(GENES)))
    df = pd.DataFrame(mat, index=ids, columns=GENES)
    if genes_as_rows:
        df = df.T
    counts = tmp_path / "count.tsv"
    df.to_csv(counts, sep=sep, index_label="gene" if genes_as_rows else "cells")
    coord_ids = ids + [f"extra{i}" for i in range(extra)]
    coords = pd.DataFrame({"cells": coord_ids, "X": np.arange(len(coord_ids), dtype=float), "Y": 2.0, "area": 10.0})
    cpath = tmp_path / "coordinates.tsv"
    coords.sample(frac=1, random_state=1).to_csv(cpath, sep="\t", index=False)
    return counts, cpath, mat


def test_split_genes_by_cells(tmp_path: Path) -> None:
    counts, coords, mat = _split_files(tmp_path)
    table = cell_table(counts, coords, unit="um", x="X", y="Y", cell_id="cells").tables["table"]
    assert table.shape == (20, 2)
    assert table.obsm["blank"].shape == (20, 1)
    np.testing.assert_array_equal(np.asarray(table.X), mat[:, :2])
    assert list(table.obs["X"]) == list(np.arange(20.0))


def test_split_cells_by_genes(tmp_path: Path) -> None:
    counts, coords, mat = _split_files(tmp_path, genes_as_rows=False)
    table = cell_table(counts, coords, unit="um", x="X", y="Y", cell_id="cells").tables["table"]
    np.testing.assert_array_equal(np.asarray(table.X), mat[:, :2])


def test_explicit_orientation_wins(tmp_path: Path) -> None:
    counts, coords, _ = _split_files(tmp_path)
    with pytest.raises(ValueError, match="have no coordinates"):
        cell_table(counts, coords, unit="um", x="X", y="Y", cell_id="cells", genes_as_rows=False)


def test_ambiguous_orientation_rejected(tmp_path: Path) -> None:
    counts, coords, _ = _split_files(tmp_path)
    coords.write_text("cells\tX\tY\nzzz\t1\t2\n")
    with pytest.raises(ValueError, match="genes_as_rows"):
        cell_table(counts, coords, unit="um", x="X", y="Y", cell_id="cells")


def test_mtx_triplet(tmp_path: Path) -> None:
    counts, coords, mat = _split_files(tmp_path)
    mtx = tmp_path / "exp.mtx.gz"
    with gzip.open(mtx, "wb") as f:
        scipy.io.mmwrite(f, sp.coo_matrix(mat.T))  # type: ignore[arg-type]  # genes x cells
    for name, labels in (("feats.txt.gz", GENES), ("obs.txt.gz", [f"c{i}" for i in range(20)])):
        with gzip.open(tmp_path / name, "wt") as f:
            f.write("\n".join(labels) + "\n")
    table = cell_table(mtx, coords, unit="um", x="X", y="Y", cell_id="cells").tables["table"]
    assert sp.issparse(table.X)
    assert table.shape == (20, 2)
    np.testing.assert_array_equal(table.X.toarray(), mat[:, :2])  # type: ignore[attr-defined]
    assert table.obsm["blank"].shape == (20, 1)


def test_mtx_missing_siblings(tmp_path: Path) -> None:
    _, coords, mat = _split_files(tmp_path)
    mtx = tmp_path / "lonely.mtx"
    scipy.io.mmwrite(str(mtx), sp.coo_matrix(mat))
    with pytest.raises(ValueError, match="features"):
        cell_table(mtx, coords, unit="um", x="X", y="Y", cell_id="cells")


def test_gzip_tsv_counts(tmp_path: Path) -> None:
    counts, coords, mat = _split_files(tmp_path)
    gz = tmp_path / "count.tsv.gz"
    with gzip.open(gz, "wb") as f:
        f.write(counts.read_bytes())
    table = cell_table(gz, coords, unit="um", x="X", y="Y", cell_id="cells").tables["table"]
    np.testing.assert_array_equal(np.asarray(table.X), mat[:, :2])


def test_join_numeric_vs_string_ids(tmp_path: Path) -> None:
    counts, coords, mat = _split_files(tmp_path, ids=[str(i) for i in range(20)])
    table = cell_table(counts, coords, unit="um", x="X", y="Y", cell_id="cells").tables["table"]
    assert list(table.obs.index) == [str(i) for i in range(20)]
    np.testing.assert_array_equal(np.asarray(table.X), mat[:, :2])


def test_unnamed_index_column(tmp_path: Path) -> None:
    counts, _, _ = _split_files(tmp_path)
    cpath = tmp_path / "centroids.csv"
    lines = [",cell,x,y"] + [f"{i},c{i},{i}.5,1.5" for i in range(20)]
    cpath.write_text("\n".join(lines) + "\n")
    table = cell_table(counts, cpath, unit="um", x="x", y="y", cell_id="cell").tables["table"]
    assert table.shape == (20, 2)
    assert not any(c.startswith("Unnamed") for c in table.obs.columns)


def test_counts_cell_without_coords_rejected(tmp_path: Path) -> None:
    counts, coords, _ = _split_files(tmp_path)
    lines = coords.read_text().splitlines()
    coords.write_text("\n".join(lines[:-8]) + "\n")  # drop 8 rows
    with pytest.raises(ValueError, match=r"8 cells.*\['c") as exc:
        cell_table(counts, coords, unit="um", x="X", y="Y", cell_id="cells")
    assert exc.value.args[0].count("'c") <= 5


def test_extra_coord_rows_dropped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from spatialdata._logging import logger

    logger.propagate = True
    counts, coords, _ = _split_files(tmp_path, extra=1)
    with caplog.at_level(logging.WARNING):
        sdata = cell_table(counts, coords, unit="um", x="X", y="Y", cell_id="cells")
    assert sdata.tables["table"].uns["spatialdata_io"]["n_coordinates_unmatched"] == 1
    assert sdata.tables["table"].shape[0] == 20
    assert "Dropped" not in caplog.text

    counts, coords, _ = _split_files(tmp_path, extra=5)
    with caplog.at_level(logging.WARNING):
        sdata = cell_table(counts, coords, unit="um", x="X", y="Y", cell_id="cells")
    assert sdata.tables["table"].uns["spatialdata_io"]["n_coordinates_unmatched"] == 5
    assert "Dropped 5 of 25" in caplog.text


def test_coord_metadata_into_obs(tmp_path: Path) -> None:
    counts, coords, _ = _split_files(tmp_path)
    sdata = cell_table(counts, coords, unit="um", x="X", y="Y", cell_id="cells")
    assert "area" in sdata.tables["table"].obs.columns
    assert sdata.tables["table"].uns["spatialdata_io"]["radius"]["source"] == "derived:area"
