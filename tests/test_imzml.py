"""Tests for the imzML reader on small synthetic processed-mode files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from spatialdata import read_zarr
from spatialdata.transformations import get_transformation

pytest.importorskip("pyimzml")
from pyimzml.ImzMLWriter import ImzMLWriter

from spatialdata_io import imzml

# (x, y): peaks; pixel (2, 2) is not acquired
_SPECTRA = {
    (1, 1): ([100.0, 200.0, 300.0], [1.0, 5.0, 2.0]),
    (2, 1): ([100.0002, 200.0], [3.0, 4.0]),
    (1, 2): ([300.0, 450.0], [1.0, 0.5]),
}


def _write(path: Path, z: int = 1, extra_settings: str = "", extra_instrument: str = "") -> Path:
    with ImzMLWriter(str(path), mode="processed") as w:
        for (x, y), (mz, it) in _SPECTRA.items():
            w.addSpectrum(np.array(mz, dtype=float), np.array(it, dtype=np.float32), (x, y, z))
    text = path.read_text()
    marker = 'name="max count of pixels y"'
    line_end = text.index("/>", text.index(marker)) + 2
    text = text[:line_end] + extra_settings + text[line_end:]
    text = text.replace('<instrumentConfiguration id="IC1">', f'<instrumentConfiguration id="IC1">{extra_instrument}')
    path.write_text(text)
    return path


_PIXEL_SIZE = (
    '<cvParam cvRef="IMS" accession="IMS:1000046" name="pixel size x" value="50" unitName="micrometer"/>'
    '<cvParam cvRef="IMS" accession="IMS:1000047" name="pixel size y" value="50" unitName="micrometer"/>'
)
_ORBITRAP = '<cvParam cvRef="MS" accession="MS:1000484" name="orbitrap"/>'


def test_top_bins_header_pixel_size_and_round_trip(tmp_path: Path) -> None:
    path = _write(tmp_path / "s.imzML", extra_settings=_PIXEL_SIZE, extra_instrument=_ORBITRAP)
    sdata = imzml(path, n_features=3)
    table = sdata["table"]
    uns = table.uns["spatialdata_io"]
    assert (uns["ppm"], uns["ppm_source"], uns["analyser"]) == (5.0, "analyser", "orbitrap")
    assert uns["feature_space"] == "mz_top3_5ppm" and uns["imzml_mode"] == "processed"
    assert (uns["pixel_size_um"], uns["pixel_size_source"], uns["pitch_unknown"]) == (50.0, "imzml header", False)
    assert uns["n_pixels_empty"] == 0 and uns["mz_range"] == [100.0, 450.0]
    assert (uns["observation_unit"], uns["unit_size_um"]) == ("spot_or_bin", 50.0)
    # totals: 200 -> 9, 100 (+100.0002, 2 ppm) -> 4, 300 -> 3, 450 -> 0.5 dropped
    np.testing.assert_allclose(table.var["mz"], [100, 200, 300], rtol=5e-6)
    assert list(table.obs_names) == ["1_1", "2_1", "1_2"]
    np.testing.assert_allclose(table.X.toarray(), [[1, 5, 2], [3, 4, 0], [0, 0, 1]])
    assert table.X.dtype == np.float32
    np.testing.assert_allclose(table.obs["total_ion_current"], [8, 7, 1.5])  # all stored peaks, kept or not
    assert "tic" not in table.obs
    square = sdata.shapes["msi_pixels"].geometry.iloc[1]
    assert square.bounds == (1, 0, 2, 1)
    scale = get_transformation(sdata.shapes["msi_pixels"], "global").to_affine_matrix(("x", "y"), ("x", "y"))
    np.testing.assert_allclose(np.diag(scale)[:2], [50, 50])
    sdata.write(tmp_path / "x.zarr")
    assert read_zarr(tmp_path / "x.zarr")["table"].uns["spatialdata_io"]["feature_space"] == "mz_top3_5ppm"


def test_deposited_features_and_kwarg_pixel_size(tmp_path: Path) -> None:
    path = _write(tmp_path / "s.imzML")
    (tmp_path / "features.csv").write_text("mz,name\n200.001,PC\n450.0,X\n999,unseen\n")
    table = imzml(path, features=tmp_path / "features.csv", pixel_size_um=10.0)["table"]
    uns = table.uns["spatialdata_io"]
    assert (uns["feature_space"], uns["ppm"], uns["ppm_source"]) == ("deposited", 10.0, "default")
    assert (uns["pixel_size_um"], uns["pixel_size_source"]) == (10.0, "repository text")
    # 200.0 is 5 ppm from 200.001 (within 10 ppm); 100 and 300 match nothing
    np.testing.assert_allclose(table.X.toarray(), [[5, 0, 0], [4, 0, 0], [0, 0.5, 0]])


def test_pitch_unknown_and_3d_error(tmp_path: Path) -> None:
    sdata = imzml(_write(tmp_path / "s.imzML"), ppm=50)
    uns = sdata["table"].uns["spatialdata_io"]
    assert uns["pitch_unknown"] is True and uns["ppm_source"] == "kwarg"
    assert (uns["observation_unit"], uns["unit_size_um"]) == ("spot_or_bin", None)
    assert sdata.coordinate_systems == ["global"]
    with pytest.raises(ValueError, match="z planes"):
        imzml(_write(tmp_path / "z.imzML", z=2))
    with ImzMLWriter(str(tmp_path / "dup.imzML"), mode="processed") as w:
        for z in (1, 1):  # two planes on one grid without position z, as in Bruker 3D exports
            w.addSpectrum(np.array([100.0]), np.array([1.0], dtype=np.float32), (1, 1, z))
    with pytest.raises(ValueError, match="repeat an x/y position"):
        imzml(tmp_path / "dup.imzML")


def test_spectra_sharing_one_position_are_kept_unplaced(tmp_path: Path) -> None:
    # PXD070149: the Thermo converter writes the scans off the grid at x=1 y=1, between the pixels
    extra = [([100.0, 300.0], [2.0, 7.0]), ([200.0], [9.0]), ([450.0], [0.5])]
    with ImzMLWriter(str(tmp_path / "p.imzML"), mode="processed") as w:
        for i, ((x, y), (mz, it)) in enumerate(_SPECTRA.items()):
            w.addSpectrum(np.array(mz), np.array(it, dtype=np.float32), (x, y, 1))
            w.addSpectrum(np.array(extra[i][0]), np.array(extra[i][1], dtype=np.float32), (1, 1, 1))
    sdata = imzml(tmp_path / "p.imzML", ppm=50)
    table, unplaced = sdata["table"], sdata["spectra_without_position"]
    assert sorted(table.obs_names) == ["1_2", "2_1"]
    assert unplaced.obs["spectrum"].tolist() == [0, 1, 3, 5] and list(unplaced.var_names) == list(table.var_names)
    # no spectrum is dropped: every intensity is in one of the two tables
    written = sum(sum(it) for _, it in _SPECTRA.values()) + sum(sum(it) for _, it in extra)
    assert table.obs["total_ion_current"].sum() + unplaced.obs["total_ion_current"].sum() == pytest.approx(written)
    uns = table.uns["spatialdata_io"]
    assert uns["spectra_without_position"] == {"n": 4, "x": 1, "y": 1} and uns["n_pixels_empty"] == 0
    sdata.write(tmp_path / "p.zarr")
    assert read_zarr(tmp_path / "p.zarr")["spectra_without_position"].n_obs == 4
    with ImzMLWriter(str(tmp_path / "two.imzML"), mode="processed") as w:
        for x, y in [(1, 1), (1, 1), (2, 1), (2, 1), (1, 2)]:  # repeats at two positions stay refused
            w.addSpectrum(np.array([100.0]), np.array([1.0], dtype=np.float32), (x, y, 1))
    with pytest.raises(ValueError, match="repeat an x/y position"):
        imzml(tmp_path / "two.imzML")


def test_bruker_3d_positions(tmp_path: Path) -> None:
    path = tmp_path / "3d.imzML"
    # two planes on one 2 x 1 grid; SCiLS stores each voxel's µm position as 3DPosition userParams
    spectra = [
        ((1, 1), (5000, -2000, 0)),
        ((2, 1), (5400, -2000, 0)),
        ((1, 1), (5010, -2000, 1500)),
        ((2, 1), (5410, -2000, 1500)),
    ]
    with ImzMLWriter(str(path), mode="processed") as w:
        for (x, y), _ in spectra:
            w.addSpectrum(np.array([100.0]), np.array([1.0], dtype=np.float32), (x, y, 1))
    text = path.read_text()
    # put each userParam block after its position y cvParam
    rebuilt = ""
    for i, chunk in enumerate(text.split('name="position y"')):
        if i == 0:
            rebuilt += chunk
            continue
        end = chunk.index("/>") + 2
        params = "".join(
            f'<userParam name="3DPosition{a}" value="{v}"/>' for a, v in zip("XYZ", spectra[i - 1][1], strict=True)
        )
        rebuilt += 'name="position y"' + chunk[:end] + params + chunk[end:]
    path.write_text(rebuilt)

    sdata = imzml(path, ppm=5, n_features=1)

    assert sorted(sdata.shapes) == ["msi_pixels_z0", "msi_pixels_z1500"]
    table = sdata["table"]
    assert list(table.obs["z_um"]) == [0, 0, 1500, 1500] and table.n_obs == 4
    uns = table.uns["spatialdata_io"]
    assert (uns["pixel_size_um"], uns["pixel_size_source"], uns["n_planes"]) == (400.0, "3DPosition spacing", 2)
    assert (uns["observation_unit"], uns["unit_size_um"]) == ("spot_or_bin", 400.0)
    assert sdata.shapes["msi_pixels_z1500"].geometry.iloc[0].bounds == (4810, -2200, 5210, -1800)

    # a stated pixel size wins over the measured spacing, which is kept (MTBLS176 pancreas: 60 stated, 64.2 measured)
    stated = imzml(path, ppm=5, n_features=1, pixel_size_um=380)
    uns = stated["table"].uns["spatialdata_io"]
    assert (uns["pixel_size_um"], uns["pixel_size_source"], uns["measured_spacing_um"]) == (
        380,
        "repository text",
        400.0,
    )
    assert stated.shapes["msi_pixels_z1500"].geometry.iloc[0].bounds == (4820, -2190, 5200, -1810)


def test_drifting_ion_is_one_feature(tmp_path: Path) -> None:
    path = tmp_path / "d.imzML"
    with ImzMLWriter(str(path), mode="processed") as w:
        # the same ion 5 ppm apart in two pixels (neighbouring bins), and a second ion
        w.addSpectrum(np.array([798.5366, 900.0]), np.array([4.0, 1.0], dtype=np.float32), (1, 1, 1))
        w.addSpectrum(np.array([798.5406]), np.array([3.0], dtype=np.float32), (2, 1, 1))
    table = imzml(path, ppm=5, n_features=2)["table"]
    assert table.n_vars == 2
    np.testing.assert_allclose(table.X.toarray(), [[4, 1], [3, 0]])


def test_mz_below_one_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "m.imzML"
    with ImzMLWriter(str(path), mode="processed") as w:
        # m/z < 1 (e.g. zero-padded or calibration artefacts in MTBLS176) would give a negative log bin
        w.addSpectrum(np.array([0.5, 0.9999, 100.0]), np.array([9.0, 9.0, 1.0], dtype=np.float32), (1, 1, 1))
    table = imzml(path, ppm=5, n_features=5)["table"]
    np.testing.assert_allclose(table.var["mz"], [100], rtol=5e-6)
    np.testing.assert_allclose(table.X.toarray(), [[1]])


@pytest.mark.parametrize(
    ("header", "kwarg", "expected", "source"),
    [
        ("10000", None, 100.0, "imzml header (area->sqrt)"),
        ("1", 20.0, 20.0, "repository text"),
        ("1", None, None, None),
    ],
)
def test_header_pixel_size_area_and_placeholder(
    tmp_path: Path, header: str, kwarg: float | None, expected: float | None, source: str | None
) -> None:
    settings = _PIXEL_SIZE.replace('value="50"', f'value="{header}"')
    uns = imzml(_write(tmp_path / "s.imzML", extra_settings=settings), pixel_size_um=kwarg)["table"].uns[
        "spatialdata_io"
    ]
    assert (uns["pixel_size_um"], uns["pixel_size_source"], uns["pitch_unknown"]) == (
        expected,
        source,
        expected is None,
    )


def _dims(width_um: float) -> str:
    return f'<cvParam cvRef="IMS" accession="IMS:1000044" name="max dimension x" value="{width_um}" unitName="micrometer"/>'


@pytest.mark.parametrize(
    ("header", "width", "expected", "source"),
    [
        ("400", 40, 20.0, "imzml header (area->sqrt)"),  # 2 pixels over 40 µm: a 20 µm pixel written as its area
        ("20", 40, 20.0, "imzml header"),
        ("7", 40, None, "inconsistent"),
    ],
)
def test_header_pixel_size_checked_against_dimensions(
    tmp_path: Path, header: str, width: float, expected: float | None, source: str
) -> None:
    settings = _PIXEL_SIZE.replace('value="50"', f'value="{header}"') + _dims(width)
    path = _write(tmp_path / "s.imzML", extra_settings=settings)
    uns = imzml(path, pixel_size_um=33.0)["table"].uns["spatialdata_io"]
    assert (uns["pixel_size_um"], uns["pixel_size_source"]) == (expected, source)
    assert (uns["pixel_size_header"], uns["pixel_size_from_dimensions"]) == (float(header), 20.0)
    assert uns["pitch_unknown"] is (expected is None)


def test_placeholder_with_dimensions_uses_them(tmp_path: Path) -> None:
    settings = _PIXEL_SIZE.replace('value="50"', 'value="1.000000"') + _dims(40)
    uns = imzml(_write(tmp_path / "s.imzML", extra_settings=settings))["table"].uns["spatialdata_io"]
    assert (uns["pixel_size_um"], uns["pixel_size_source"]) == (20.0, "imzml dimensions")


@pytest.mark.parametrize(
    ("text", "expected", "source"),
    [
        (None, None, "ambiguous unitless header"),
        (25.0, 25.0, "repository text"),  # agrees with the header read as a side
        (5.0, 5.0, "repository text"),  # agrees with the header read as an area
        (30.0, None, "ambiguous unitless header"),
    ],
)
def test_unitless_header_without_dimensions(
    tmp_path: Path, text: float | None, expected: float | None, source: str
) -> None:
    unitless = '<cvParam cvRef="IMS" accession="IMS:1000046" name="pixel size" value="25"/>'
    uns = imzml(_write(tmp_path / "s.imzML", extra_settings=unitless), pixel_size_um=text)["table"].uns[
        "spatialdata_io"
    ]
    assert (uns["pixel_size_um"], uns["pixel_size_source"], uns["pitch_unknown"]) == (
        expected,
        source,
        expected is None,
    )


def test_mzml_without_namespace(tmp_path: Path) -> None:
    path = _write(tmp_path / "s.imzML", extra_settings=_PIXEL_SIZE)
    path.write_text(path.read_text().replace(' xmlns="http://psi.hupo.org/ms/mzml"', "", 1))
    table = imzml(path, n_features=3)["table"]
    assert table.n_obs == 3 and table.uns["spatialdata_io"]["pixel_size_um"] == 50.0


@pytest.mark.parametrize("mode", ["continuous", "processed"])
def test_memmap_path_matches_pyimzml(tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from spatialdata_io.readers import imzml as mod

    rng = np.random.default_rng(0)
    mz = np.linspace(100, 900, 400)
    path = tmp_path / f"{mode}.imzML"
    with ImzMLWriter(str(path), mode=mode) as w:
        for i in range(11):
            # spectrum 3: all zeros (continuous) or shorter (processed); batches of 3 below
            k = 400 if mode == "continuous" or i != 3 else 50
            it = rng.random(k).astype(np.float32) * (i != 3 or mode == "processed")
            w.addSpectrum(mz[:k], it, (i % 4 + 1, i // 4 + 1, 1))
    monkeypatch.setattr(mod, "_BATCH", 3)
    used = []
    fast_reader = mod._spectra_memmap
    monkeypatch.setattr(mod, "_spectra_memmap", lambda *a: used.append(1) or fast_reader(*a))
    fast = imzml(path, n_features=20, ppm=50)["table"]
    assert bool(used) == (mode == "continuous")  # processed mode stays on pyimzml
    monkeypatch.setattr(mod, "_spectra_memmap", lambda parser, ibd: mod._spectra(parser, None))
    slow = imzml(path, n_features=20, ppm=50)["table"]
    assert (fast.X != slow.X).nnz == 0 and fast.X.dtype == slow.X.dtype
    pd.testing.assert_frame_equal(fast.var, slow.var)
    pd.testing.assert_frame_equal(fast.obs, slow.obs)


def _flip_last_ibd_byte(ibd: Path) -> None:
    data = bytearray(ibd.read_bytes())
    data[-1] ^= 0xFF
    ibd.write_bytes(bytes(data))


def test_ibd_checksum_mismatch_alone_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    # a replaced .ibd (e.g. *_recal_M2, *_fix) whose imzML header was never updated decodes fine: kept, both hashes kept
    path = _write(tmp_path / "s.imzML", extra_settings=_PIXEL_SIZE)
    _flip_last_ibd_byte(tmp_path / "s.ibd")
    mismatch = imzml(path)["table"].uns["spatialdata_io"]["ibd_checksum_mismatch"]
    assert mismatch["algorithm"] == "sha1" and mismatch["declared"] != mismatch["ibd"]
    assert "the imzML declares" in caplog.text


def test_ibd_checksum_mismatch_that_fails_decoding_is_refused(tmp_path: Path) -> None:
    # MTBLS2960: the .ibd is not the one the imzML names, and spectra do not decode; 1 of 200 alone would be kept
    path = _write_many(tmp_path / "s.imzML", 200, 1)
    _flip_last_ibd_byte(tmp_path / "s.ibd")
    with pytest.raises(ValueError, match="does not decode with this .imzML"):
        imzml(path)


@pytest.mark.parametrize("kind", ["unpadded", "placeholder"])
def test_ibd_checksum_writer_quirks_are_accepted(tmp_path: Path, kind: str) -> None:
    # 51 of 972 deposited imzML pad only the first digest byte; 12 declare value="0"
    from spatialdata_io.readers.imzml import _check_ibd_checksum

    (tmp_path / "s.ibd").write_bytes(b"spectra")  # sha1 a1bbc6b19fb5f7015368195e...: byte 0x01 loses its zero
    value = "a1bbc6b19fb5f715368195eac56c776ba54141c" if kind == "unpadded" else "0"
    assert _check_ibd_checksum(tmp_path / "s.imzML", f'<cvParam accession="IMS:1000091" value="{value}"/>') is None
    assert (
        _check_ibd_checksum(tmp_path / "s.imzML", '<cvParam accession="IMS:1000091" value="1a2b"/>')["declared"]
        == "1a2b"
    )


def _write_many(path: Path, n: int, bad: int) -> Path:
    """``n`` spectra on a row; the first ``bad`` of them have decreasing m/z (a mis-decoded .ibd)."""
    with ImzMLWriter(str(path), mode="processed") as w:
        for i in range(n):
            mz = [300.0, 200.0, 100.0] if i < bad else [100.0, 200.0, 300.0]
            w.addSpectrum(np.array(mz), np.array([1.0, 2.0, 3.0], dtype=np.float32), (i + 1, 1, 1))
    text = path.read_text()
    marker = 'name="max count of pixels y"'
    line_end = text.index("/>", text.index(marker)) + 2
    path.write_text(text[:line_end] + _PIXEL_SIZE + text[line_end:])
    return path


def test_spectra_failing_the_decode_check(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with pytest.raises(ValueError, match="2 of 100 spectra have decreasing m/z"):
        imzml(_write_many(tmp_path / "bad.imzML", 100, 2))  # above 1%: refused
    sdata = imzml(_write_many(tmp_path / "ok.imzML", 200, 1))  # at or below 1%: kept, counted and logged
    assert sdata["table"].uns["spatialdata_io"]["n_spectra_undecodable"] == 1
    assert "1 of 200 spectra have decreasing m/z" in caplog.text


def test_mz_outside_the_declared_scan_window_fails_the_decode_check(tmp_path: Path) -> None:
    window = (
        '<cvParam cvRef="MS" accession="MS:1000501" name="scan window lower limit" value="150.0"/>'
        '<cvParam cvRef="MS" accession="MS:1000500" name="scan window upper limit" value="500.0"/>'
    )
    path = _write(tmp_path / "w.imzML", extra_settings=_PIXEL_SIZE + window)  # two spectra start at m/z 100
    with pytest.raises(ValueError, match="outside the declared scan window 150-500"):
        imzml(path)


def _write_tiled(path: Path, positions: list[tuple[float, float]]) -> Path:
    """PXD038114-style export: position x/y is the truncated stage coordinate, the real one is 3DPositionX/Y."""
    with ImzMLWriter(str(path), mode="processed") as w:
        for x, y in positions:
            w.addSpectrum(np.array([100.0, 200.0]), np.array([1.0, 2.0], dtype=np.float32), (int(x) + 8, int(y) + 2, 1))
    text, out, start = path.read_text(), [], 0
    for x, y in positions:  # one pair of userParams after each spectrum's position y
        i = text.index('name="position y"', start)
        j = text.index("/>", i) + 2
        out.append(
            text[start:j]
            + f'<userParam name="3DPositionX" value="{x:.6f}" /><userParam name="3DPositionY" value="{y:.6f}" />'
        )
        start = j
    path.write_text("".join(out) + text[start:])
    return path


def test_truncated_positions_are_placed_by_their_3dposition_grid(tmp_path: Path) -> None:
    pos = [(-7.15 + 0.05 * i, -1.05 + 0.05 * j) for j in range(3) for i in range(4)]  # 12 spectra, 1 integer position
    path = _write_tiled(tmp_path / "t.imzML", pos)
    uns = (table := imzml(path, n_features=2)["table"]).uns["spatialdata_io"]
    assert len(table) == 12 and uns["pitch_unknown"] and uns["pixel_size_source"] is None  # no unit: no pitch guessed
    assert uns["positions_source"] == "3DPositionX/Y (unit not stated)" and uns["position_step"] == pytest.approx(0.05)
    assert sorted(zip(table.obs["x"], table.obs["y"], strict=True)) == [
        (i, j) for i in range(1, 5) for j in range(1, 4)
    ]
    np.testing.assert_allclose(sorted(table.obs["x_pos"]), sorted(x for x, _ in pos))
    assert imzml(path, n_features=2, pixel_size_um=50)["table"].uns["spatialdata_io"]["pixel_size_um"] == 50

    off = _write_tiled(tmp_path / "off.imzML", [*pos[:-1], (pos[-1][0] + 0.02, pos[-1][1])])
    with pytest.raises(ValueError, match="not on one regular raster"):
        imzml(off, n_features=2)


def test_ibd_shorter_than_its_imzml_indexes_is_refused(tmp_path: Path) -> None:
    # sddb_e5kd42j6uj: the deposited .ibd is 8.2 GB, the .imzML points up to 13.2 GB
    path = _write(tmp_path / "s.imzML", extra_settings=_PIXEL_SIZE)
    ibd = tmp_path / "s.ibd"
    ibd.write_bytes(ibd.read_bytes()[:-100])
    with pytest.raises(ValueError, match="point past its end"):
        imzml(path)
