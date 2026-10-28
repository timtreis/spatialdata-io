from __future__ import annotations

import gzip
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.io
import scipy.sparse as sp
from spatialdata import SpatialData, join_spatialelement_table
from spatialdata._logging import logger
from spatialdata.transformations import Identity, Scale, get_transformation

from spatialdata_io import cell_table
from spatialdata_io.readers.cell_table import _build

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


def _dense(X: object) -> np.ndarray:
    return sp.csr_matrix(X).toarray()


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
    assert np.array_equal(_dense(a.X), _dense(b.X))


def test_metadata_columns_override(csv: Path) -> None:
    default = cell_table(csv, unit="um").tables["table"]
    assert "min_x" in default.obs.columns
    sdata = cell_table(csv, unit="um", metadata_columns=["fov", "volume", "center_x", "center_y"])
    table = sdata.tables["table"]
    assert "min_x" not in table.obs.columns
    assert "min_x" in table.var_names


SPLIT_HEADER = "id,sample,volume,center_x,center_y,GeneA,GeneB"


def _split_csv(tmp_path: Path, samples: list[str], ids: list[str] | None = None) -> Path:
    ids = ids or [f"c{i}" for i in range(len(samples))]
    rows = [f"{i},{s},{100.0 + k},{k}.5,{k}.25,{k},1" for k, (i, s) in enumerate(zip(ids, samples, strict=True))]
    return _write(tmp_path / "split.csv", rows, SPLIT_HEADER)


def test_split_by(tmp_path: Path) -> None:
    sdata = cell_table(_split_csv(tmp_path, ["a", "b", "a", "b", "b"]), unit="um", split_by="sample")
    assert set(sdata.shapes) == {"cells_a", "cells_b"}
    assert list(sdata.tables) == ["table"]
    table = sdata["table"]
    assert set(table.obs["region"].cat.categories) == {"cells_a", "cells_b"}
    attrs = table.uns["spatialdata_attrs"]
    assert set(attrs["region"]) == {"cells_a", "cells_b"}
    assert (len(sdata["cells_a"]), len(sdata["cells_b"])) == (2, 3)
    for name in sdata.shapes:
        ids = table.obs.loc[table.obs["region"] == name, "cell_id"]
        assert sorted(ids) == sorted(sdata[name].index)
    # per-cell radius stays aligned with the cell it came from
    r = np.asarray(table.uns["spatialdata_io"]["radius"]["value_um"])
    assert r.shape == (5,)
    assert np.allclose(
        sdata["cells_b"]["radius"].to_numpy(), np.cbrt(3 * np.array([101.0, 103.0, 104.0]) / (4 * np.pi))
    )


def test_split_sanitizes_names(tmp_path: Path) -> None:
    sdata = cell_table(_split_csv(tmp_path, ["slice 1", "x/y", "slice 1"]), unit="um", split_by="sample")
    assert set(sdata.shapes) == {"cells_slice_1", "cells_x_y"}
    sdata.write(tmp_path / "out.zarr")
    back = SpatialData.read(tmp_path / "out.zarr")
    assert set(back["table"].obs["region"].cat.categories) == {"cells_slice_1", "cells_x_y"}


def test_split_name_collision_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same element name"):
        cell_table(_split_csv(tmp_path, ["a b", "a_b"]), unit="um", split_by="sample")


@pytest.mark.parametrize("values", [["A", "a"], ["S1", "s1"]])
def test_split_case_collision_rejected(tmp_path: Path, values: list[str]) -> None:
    with pytest.raises(ValueError, match="same element name") as e:
        cell_table(_split_csv(tmp_path, values), unit="um", split_by="sample")
    assert values[0] in str(e.value)
    assert values[1] in str(e.value)


def test_split_prefix_collision_rejected(tmp_path: Path) -> None:
    # "b_x" in split "a" and "x" in split "a_b" both become "a_b_x"; "x" repeats, so prefixing kicks in
    rows = ["b_x,a,1,1,1,1,1", "x,a_b,1,2,2,1,1", "x,a,1,3,3,1,1"]
    with pytest.raises(ValueError, match="not unique after prefixing"):
        cell_table(_write(tmp_path / "q.csv", rows, SPLIT_HEADER), unit="um", split_by="sample")


def test_split_by_roundtrip_join(tmp_path: Path) -> None:
    sdata = cell_table(_split_csv(tmp_path, ["a", "b", "a", "b", "b"]), unit="um", split_by="sample")
    sdata.write(tmp_path / "out.zarr")
    back = SpatialData.read(tmp_path / "out.zarr")
    table = back["table"]
    for name in ("cells_a", "cells_b"):
        expected = table.obs.loc[table.obs["region"] == name, "cell_id"]
        assert sorted(back[name].index) == sorted(expected)
        _, joined = join_spatialelement_table(back, name, table_name="table", how="inner")
        assert joined.n_obs == len(expected)
        assert sorted(joined.obs["cell_id"]) == sorted(expected)


def test_split_by_in_coordinates_file(tmp_path: Path) -> None:
    counts, coords, _ = _split_files(tmp_path, n=6)
    df = pd.read_csv(coords, sep="\t")
    df["sample"] = ["s 1" if int(c[1:]) % 2 else "s2" for c in df["cells"]]
    df.to_csv(coords, sep="\t", index=False)
    sdata = cell_table(counts, coords, unit="um", x="X", y="Y", split_by="sample")
    assert set(sdata.shapes) == {"cells_s_1", "cells_s2"}
    assert sdata["table"].n_obs == 6
    assert len(sdata["cells_s_1"]) == 3


def test_split_by_missing_column(csv: Path) -> None:
    with pytest.raises(ValueError, match="nope"):
        cell_table(csv, unit="um", split_by="nope")


def test_duplicate_ids_across_splits_allowed(tmp_path: Path) -> None:
    sdata = cell_table(_split_csv(tmp_path, ["a", "b"], ids=["c1", "c1"]), unit="um", split_by="sample")
    table = sdata["table"]
    assert table.obs.index.is_unique
    assert table.n_obs == 2
    assert set(table.obs.index) == {"a_c1", "b_c1"}


def test_duplicate_ids_within_split_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        cell_table(_split_csv(tmp_path, ["a", "a"], ids=["c1", "c1"]), unit="um", split_by="sample")


def test_split_by_missing_values_rejected(tmp_path: Path) -> None:
    f = _write(tmp_path / "s.csv", ["c1,a,1,1,1,1,1", "c2,,1,2,2,1,1"], SPLIT_HEADER)
    with pytest.raises(ValueError, match="missing"):
        cell_table(f, unit="um", split_by="sample")


def test_split_by_h5ad(tmp_path: Path) -> None:
    f = _h5ad(tmp_path / "a.h5ad", obsm={"spatial": np.array([[1.0, 3.0], [2.0, 4.0]])})
    adata = ad.read_h5ad(f)
    adata.obs["batch"] = ["p", "q"]
    adata.write_h5ad(f)
    sdata = cell_table(f, unit="um", split_by="batch")
    assert set(sdata.shapes) == {"cells_p", "cells_q"}


def test_roundtrip(csv: Path, tmp_path: Path) -> None:
    sdata = cell_table(csv, unit="um")
    p = tmp_path / "out.zarr"
    sdata.write(p)
    back = SpatialData.read(p)
    assert back.tables["table"].shape == (3, 2)
    assert back.tables["table"].uns["spatialdata_io"]["coordinates"]["unit"] == "um"


def test_build_sparse() -> None:
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
    np.testing.assert_array_equal(_dense(table.X), mat[:, :2])
    assert list(table.obs["X"]) == list(np.arange(20.0))


def test_split_cells_by_genes(tmp_path: Path) -> None:
    counts, coords, mat = _split_files(tmp_path, genes_as_rows=False)
    table = cell_table(counts, coords, unit="um", x="X", y="Y", cell_id="cells").tables["table"]
    np.testing.assert_array_equal(_dense(table.X), mat[:, :2])


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
    np.testing.assert_array_equal(_dense(table.X), mat[:, :2])


def test_join_numeric_vs_string_ids(tmp_path: Path) -> None:
    counts, coords, mat = _split_files(tmp_path, ids=[str(i) for i in range(20)])
    table = cell_table(counts, coords, unit="um", x="X", y="Y", cell_id="cells").tables["table"]
    assert list(table.obs.index) == [str(i) for i in range(20)]
    np.testing.assert_array_equal(_dense(table.X), mat[:, :2])


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


def test_extra_coord_rows_dropped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(logger, "propagate", True)
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


def test_leading_zero_ids(tmp_path: Path) -> None:
    ids = ["007", "008", "010"]
    n = len(ids)
    counts = tmp_path / "count.tsv"
    counts.write_text(
        "gene\t" + "\t".join(ids) + "\nGeneA\t" + "\t".join(["1"] * n) + "\nGeneB\t" + "\t".join(["2"] * n) + "\n"
    )
    coords = tmp_path / "coordinates.tsv"
    coords.write_text("cells\tX\tY\n" + "\n".join(f"{i}\t{k}\t1" for k, i in enumerate(ids)) + "\n")
    table = cell_table(counts, coords, unit="um", x="X", y="Y", cell_id="cells").tables["table"]
    assert list(table.obs.index) == ids
    assert list(table.obs["X"]) == list(range(n))


def test_large_integer_ids(tmp_path: Path) -> None:
    ids = ["123456789012345678", "123456789012345679"]
    counts = tmp_path / "count.csv"
    counts.write_text("," + ",".join(["GeneA", "GeneB"]) + "\n" + "\n".join(f"{i},1,2" for i in ids) + "\n")
    coords = tmp_path / "coordinates.csv"
    coords.write_text("cell,x,y\n" + "\n".join(f"{i},{k},1" for k, i in enumerate(ids)) + "\n,5,5\n")
    table = cell_table(counts, coords, unit="um", x="x", y="y", cell_id="cell").tables["table"]
    assert list(table.obs.index) == ids


def test_unnamed_index_column_default(tmp_path: Path) -> None:
    counts, _, _ = _split_files(tmp_path)
    cpath = tmp_path / "centroids.csv"
    cpath.write_text("\n".join([",cell,x,y"] + [f"{i},c{i},{i}.5,1.5" for i in range(20)]) + "\n")
    table = cell_table(counts, cpath, unit="um", x="x", y="y").tables["table"]
    assert table.shape == (20, 2)
    assert list(table.obs.index)[:2] == ["c0", "c1"]
    assert not any(c.startswith("Unnamed") for c in table.obs.columns)


def test_counts_index_leading_zero(tmp_path: Path) -> None:
    counts = tmp_path / "count.csv"
    counts.write_text(",GeneA,GeneB\n007,1,2\n010,3,4\n")
    coords = tmp_path / "coordinates.csv"
    coords.write_text("cell,x,y\n007,1,1\n010,2,2\n")
    table = cell_table(counts, coords, unit="um", x="x", y="y", cell_id="cell").tables["table"]
    assert list(table.obs.index) == ["007", "010"]


def test_wide_csv_leading_zero_ids(tmp_path: Path) -> None:
    path = _write(tmp_path / "wide.csv", ["007,0,523.6,10.5,20.5,9,1,2,0", "010,0,904.8,30.0,40.0,29,3,0,1"])
    table = cell_table(path, unit="um").tables["table"]
    assert list(table.obs.index) == ["007", "010"]


def test_mtx_barcodes_leading_zero(tmp_path: Path) -> None:
    ids = ["007", "010"]
    mtx = tmp_path / "exp.mtx"
    scipy.io.mmwrite(str(mtx), sp.coo_matrix(np.array([[1, 3], [2, 4]])))  # genes x cells
    (tmp_path / "features.tsv").write_text("GeneA\nNA\n")
    (tmp_path / "barcodes.tsv").write_text("\n".join(ids) + "\n")
    coords = tmp_path / "coordinates.csv"
    coords.write_text("cell,x,y\n007,1,1\n010,2,2\n")
    table = cell_table(mtx, coords, unit="um", x="x", y="y", cell_id="cell").tables["table"]
    assert list(table.obs.index) == ids
    assert list(table.var_names) == ["GeneA", "NA"]


def _h5ad(
    path: Path,
    X: np.ndarray | None = None,
    obs_cols: dict[str, list[float]] | None = None,
    obsm: dict[str, np.ndarray] | None = None,
    layers: dict[str, np.ndarray] | None = None,
    raw: tuple[np.ndarray, list[str]] | None = None,
) -> Path:
    X = np.array([[1, 2, 0], [3, 0, 4]]) if X is None else X
    obs = pd.DataFrame(obs_cols or {}, index=pd.Index(["c1", "c2"]))
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=["g1", "g2", "blank-1"]))
    for k, v in (obsm or {}).items():
        adata.obsm[k] = v
    for k, v in (layers or {}).items():
        adata.layers[k] = v
    if raw is not None:
        adata.raw = ad.AnnData(X=raw[0], var=pd.DataFrame(index=raw[1]), obs=obs)
    adata.write_h5ad(path)
    return path


def test_h5ad_obs_coords(tmp_path: Path) -> None:
    f = _h5ad(tmp_path / "a.h5ad", obs_cols={"center_x": [1.0, 2.0], "center_y": [3.0, 4.0]})
    sdata = cell_table(f, unit="um")
    table = sdata["table"]
    assert list(table.obs.index) == ["c1", "c2"]
    assert all(isinstance(i, str) for i in table.obs.index)
    assert list(table.var_names) == ["g1", "g2"]
    assert table.obsm["blank"].shape == (2, 1)
    assert table.uns["spatialdata_io"]["coordinates"]["x"] == "center_x"
    assert sdata["cells"].shape[0] == 2


def test_h5ad_obsm_spatial(tmp_path: Path) -> None:
    f = _h5ad(tmp_path / "a.h5ad", obsm={"spatial": np.array([[1.0, 3.0], [2.0, 4.0]])})
    sdata = cell_table(f, unit="um")
    uns = sdata["table"].uns["spatialdata_io"]["coordinates"]
    assert uns["x"] == "obsm:spatial[0]"
    assert uns["y"] == "obsm:spatial[1]"
    assert list(sdata["cells"].geometry.x) == [1.0, 2.0]
    assert list(sdata["cells"].geometry.y) == [3.0, 4.0]
    assert "spatial" not in sdata["table"].obsm
    sdata.write(tmp_path / "out.zarr")
    back = SpatialData.read(tmp_path / "out.zarr")
    assert back["table"].uns["spatialdata_io"]["coordinates"]["x"] == "obsm:spatial[0]"


def test_h5ad_obsm_x_spatial(tmp_path: Path) -> None:
    f = _h5ad(tmp_path / "a.h5ad", obsm={"X_spatial": np.array([[1.0, 3.0], [2.0, 4.0]])})
    sdata = cell_table(f, unit="um")
    assert sdata["table"].uns["spatialdata_io"]["coordinates"]["x"] == "obsm:X_spatial[0]"


def test_h5ad_no_coordinates_rejected(tmp_path: Path) -> None:
    f = _h5ad(tmp_path / "a.h5ad")
    with pytest.raises(ValueError, match="center_x"):
        cell_table(f, unit="um")


def test_h5ad_lognorm_rejected(tmp_path: Path) -> None:
    f = _h5ad(
        tmp_path / "a.h5ad",
        X=np.array([[0.5, 1.2, 0.0], [1.1, 0.0, 0.3]]),
        obs_cols={"center_x": [1.0, 2.0], "center_y": [3.0, 4.0]},
    )
    with pytest.raises(ValueError, match="non-integer"):
        cell_table(f, unit="um")


def test_h5ad_layer_counts(tmp_path: Path) -> None:
    f = _h5ad(
        tmp_path / "a.h5ad",
        X=np.array([[0.5, 1.2, 0.0], [1.1, 0.0, 0.3]]),
        obs_cols={"center_x": [1.0, 2.0], "center_y": [3.0, 4.0]},
        layers={"counts": np.array([[1, 2, 0], [3, 0, 4]])},
    )
    table = cell_table(f, unit="um", layer="counts")["table"]
    assert table.X.toarray().tolist() == [[1, 2], [3, 0]]


def test_h5ad_raw(tmp_path: Path) -> None:
    f = _h5ad(
        tmp_path / "a.h5ad",
        X=np.array([[0.5, 1.2, 0.0], [1.1, 0.0, 0.3]]),
        obs_cols={"center_x": [1.0, 2.0], "center_y": [3.0, 4.0]},
        raw=(np.array([[5, 6, 0, 1], [7, 0, 8, 2]]), ["r1", "r2", "r3", "blank-9"]),
    )
    table = cell_table(f, unit="um", layer="raw")["table"]
    assert list(table.var_names) == ["r1", "r2", "r3"]
    assert table.X.toarray().tolist() == [[5, 6, 0], [7, 0, 8]]
    assert list(table.obsm["blank"].columns) == ["blank-9"]


def test_h5ad_missing_layer(tmp_path: Path) -> None:
    f = _h5ad(
        tmp_path / "a.h5ad",
        obs_cols={"center_x": [1.0, 2.0], "center_y": [3.0, 4.0]},
        layers={"counts": np.array([[1, 2, 0], [3, 0, 4]])},
    )
    with pytest.raises(ValueError, match="nope.*counts"):
        cell_table(f, unit="um", layer="nope")
    with pytest.raises(ValueError, match="raw"):
        cell_table(f, unit="um", layer="raw")


def test_h5ad_with_coordinates_file(tmp_path: Path) -> None:
    f = _h5ad(tmp_path / "a.h5ad")
    coords = tmp_path / "coords.csv"
    coords.write_text("cell_id,center_x,center_y\nc1,1,2\nc2,3,4\nc3,5,6\n")
    sdata = cell_table(f, coords, unit="um")
    assert sdata["table"].uns["spatialdata_io"]["n_coordinates_unmatched"] == 1
    assert list(sdata["cells"].geometry.x) == [1.0, 3.0]
    other = tmp_path / "other.csv"
    other.write_text("cell_id,center_x,center_y\nc1,1,2\n")
    with pytest.raises(ValueError, match="no coordinates"):
        cell_table(f, other, unit="um")


def _mtx_files(tmp_path: Path, mat: np.ndarray) -> Path:
    mtx = tmp_path / "exp.mtx.gz"
    with gzip.open(mtx, "wb") as f:
        scipy.io.mmwrite(f, sp.coo_matrix(mat.T))  # type: ignore[arg-type]  # genes x cells
    for name, labels in (("features.tsv", GENES), ("barcodes.tsv", [f"c{i}" for i in range(len(mat))])):
        (tmp_path / name).write_text("\n".join(labels) + "\n")
    return mtx


def test_x_is_csr_on_every_path(csv: Path, tmp_path: Path) -> None:
    counts, coords, mat = _split_files(tmp_path)
    mtx = _mtx_files(tmp_path, mat)
    h5 = _h5ad(tmp_path / "a.h5ad", obsm={"spatial": np.array([[1.0, 3.0], [2.0, 4.0]])})
    tables = [
        cell_table(csv, unit="um")["table"],
        cell_table(counts, coords, unit="um", x="X", y="Y", cell_id="cells")["table"],
        cell_table(mtx, coords, unit="um", x="X", y="Y", cell_id="cells")["table"],
        cell_table(h5, unit="um")["table"],
    ]
    for table in tables:
        assert type(table.X) is sp.csr_matrix
    np.testing.assert_array_equal(tables[0].X.toarray(), [[1, 2], [3, 0], [0, 5]])  # type: ignore[union-attr]
    np.testing.assert_array_equal(tables[0].obsm["blank"], [[0], [1], [2]])


def test_wide_string_column_rejected(tmp_path: Path) -> None:
    rows = [f"{r},T{i}" for i, r in enumerate(ROWS)]
    with pytest.raises(ValueError, match="cell_type.*metadata_columns"):
        cell_table(_write(tmp_path / "s.csv", rows, HEADER + ",cell_type"), unit="um")


def test_wide_trailing_comma(tmp_path: Path) -> None:
    f = _write(tmp_path / "t.csv", [r + "," for r in ROWS], HEADER + ",")
    table = cell_table(f, unit="um")["table"]
    assert list(table.var_names) == ["GeneA", "GeneB"]
    assert list(table.obs.columns).count("fov") == 1
    assert not any(c.startswith("Unnamed") for c in table.obs.columns)


def test_wide_nan_count_rejected(tmp_path: Path) -> None:
    rows = [ROWS[0].replace(",1,2,0", ",1,,0"), *ROWS[1:]]
    with pytest.raises(ValueError, match="GeneB.*metadata_columns"):
        cell_table(_write(tmp_path / "n.csv", rows), unit="um")
