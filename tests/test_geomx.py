"""Tests for geomx() on synthetic inputs in the GeomxTools formats and in the layouts of real figshare/Mendeley deposits."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from spatialdata import read_zarr
from spatialdata.transformations import get_transformation

from spatialdata_io import geomx

# GeomxTools inst/extdata/DSP_NGS_Example_Data/DSP-1001250002642-A02.dcc @c1d0687, cut down
_DCC = """<Header>
FileVersion,0.01
SoftwareVersion,1.0.0
Date,2020-7-14
</Header>
<Scan_Attributes>
ID,{id}
Plate_ID,1001250002642
Well,{well}
</Scan_Attributes>
<NGS_Processing_Attributes>
SeqSetId,VH00121:3:AAAG2YWM5
tamperedIni,No
trimGaloreOpts," --hardtrim5 26 --dont_gzip"
Raw,646250
Trimmed,646250
Stitched,616150
Aligned,610390
umiQ30,0.9785
rtsQ30,0.9804
</NGS_Processing_Attributes>
<Code_Summary>
{counts}
</Code_Summary>
SOMEHASH100000000000
"""


def _pkc(path: Path, name: str, analyte: str, targets: list[tuple[str, str, list[str]]]) -> Path:
    """A PKC as in GeomxTools' example kits: RNA probes carry the RTS_ID, protein targets carry it themselves."""
    tl = []
    for display, code_class, rts in targets:
        if analyte == "Protein":
            tl.append(
                {
                    "GeneID": [1],
                    "AnalyteType": "Protein",
                    "DisplayName": display,
                    "SystematicName": [display],
                    "CodeClass": code_class,
                    "RTS_ID": rts[0],
                    "Probes": [{"ProbeID": f"DPROT_{display}", "Accession": ["P1"]}],
                }
            )
        else:
            probes = [
                {
                    "RTS_Seq": "GATTACTATGGT",
                    "DisplayName": f"{display}_{i:02d}",
                    "ProbeID": f"{display}:{i}",
                    "SystematicName": [display],
                    "RTS_ID": r,
                    "GeneID": [str(100 + i)],
                }
                for i, r in enumerate(rts, 1)
            ]
            tl.append({"AnalyteType": "RNA", "DisplayName": display, "CodeClass": code_class, "Probes": probes})
    d = {"Name": name, "Version": 1.0, "AnalyteType": analyte, "MinArea": 16000, "MinNuclei": 200}
    d |= {"Date": "200707", "ProbeGroups": [{"Name": "All", "Targets": [t[0] for t in targets]}], "Targets": tl}
    path.write_text(json.dumps(d, indent=2))
    return path


def _write_dcc(path: Path, sample: str, counts: list[tuple[str, int]], gz: bool = False) -> None:
    text = _DCC.format(id=sample, well=sample[-3:], counts="\n".join(f"{r},{c}" for r, c in counts))
    if gz:
        with gzip.open(path, "wt") as f:
            f.write(text)
    else:
        path.write_text(text)


def _ngs_kit(tmp_path: Path) -> tuple[Path, Path]:
    dcc = tmp_path / "dcc"
    dcc.mkdir()
    pkc = _pkc(
        tmp_path / "Hs_R_NGS_CTA_v1.0.pkc",
        "Cancer Transcriptome Atlas",
        "RNA",
        [
            ("HSPA2", "Endogenous01", ["RTS0039454", "RTS0039455"]),
            ("ZNF207", "Endogenous01", ["RTS0039458"]),
            ("Negative Probe", "Negative01", ["RTS0047618", "RTS0047619", "RTS0047620"]),
            ("UNSEEN", "Endogenous01", ["RTS0099999"]),
        ],
    )
    # A02: a duplicated probe row and a probe that's in no PKC; A03 gzipped
    _write_dcc(
        dcc / "DSP-1001250002642-A02.dcc",
        "DSP-1001250002642-A02",
        [("RTS0039454", 294), ("RTS0039455", 270), ("RTS0039458", 345), ("RTS0047618", 4), ("RTS0047619", 9)]
        + [("RTS0039455", 3), ("RTS0011111", 7)],
    )
    _write_dcc(
        dcc / "DSP-1001250002642-A03.dcc.gz",
        "DSP-1001250002642-A03",
        [("RTS0039454", 1), ("RTS0047620", 2)],
        gz=True,
    )
    return dcc, pkc


def test_dcc_pkc_worksheet_tables(tmp_path: Path) -> None:
    dcc, pkc = _ngs_kit(tmp_path)
    # lab worksheet as in GeomxTools' GxT_testing_labWorksheet.txt: a summary, then the AOIs after "Annotations"
    ws = tmp_path / "run_LabWorksheet.txt"
    ws.write_text(
        "Experiment Summary\nReadout group name\t04June2021_CTA\nNumber Of AOIs\t3\n\nAnnotations\n"
        "Sample_ID\tSlide name\tScan name\tPanel\tRoi\tSegment\tAoi\tArea\n"
        "DSP-1001250002642-A01\tNo Template Control\t\t\t\t\t\t\n"
        "DSP-1001250002642-A02\tslide1\tcw005 Slide1\tCTA\t1\tGeometric Segment\tGeometric Segment-aoi-001\t31318.7\n"
        "DSP-1001250002642-A03\tslide1\tcw005 Slide1\tCTA\t2\tGeometric Segment\tGeometric Segment-aoi-001\t2000\n"
    )

    sdata = geomx(dcc, pkc, ws)

    probe, table = sdata["table_probe"], sdata["table"]
    assert list(probe.obs_names) == ["DSP-1001250002642-A02", "DSP-1001250002642-A03"]
    # every PKC probe, then the DCC-only ones (the duplicate and the unannotated probe)
    assert list(probe.var_names[-2:]) == ["RTS0039455_dup1", "RTS0011111"]
    assert probe.n_vars == 7 + 2
    assert probe.var.loc["RTS0011111", "feature_types"] == "Unannotated Probe"
    assert probe.var.loc["RTS0039455_dup1", "Target"] == "HSPA2"
    np.testing.assert_array_equal(probe.X.sum(axis=1).A1, [294 + 270 + 345 + 4 + 9 + 3 + 7, 3])
    assert probe.uns["spatialdata_io"]["duplicate_probes"] == {"DSP-1001250002642-A02": 1}

    # targets: the SUM of their probes, integers; unannotated probes have no target
    assert list(table.var_names) == ["HSPA2", "ZNF207", "Negative Probe", "UNSEEN"]
    np.testing.assert_array_equal(table.X.toarray(), [[294 + 270 + 3, 345, 13, 0], [1, 0, 2, 0]])
    assert np.issubdtype(table.X.dtype, np.integer)
    assert list(table.var["n_probes"]) == [3, 1, 3, 1]  # HSPA2 counts its _dup1 row
    assert list(table.var["feature_types"]) == ["Gene Expression"] * 2 + ["Negative Control Probe", "Gene Expression"]
    assert list(table.var["is_control"]) == [False, False, True, False]

    obs = table.obs
    assert obs.loc["DSP-1001250002642-A02", "Area"] == "31318.7"  # verbatim
    assert ["Slide name", "Slide_name"] in [list(p) for p in table.uns["spatialdata_io"]["renamed_columns"]]
    assert obs.loc["DSP-1001250002642-A02", "dcc_trimGaloreOpts"] == " --hardtrim5 26 --dont_gzip"
    assert obs.loc["DSP-1001250002642-A03", "dcc_Raw"] == 646250
    assert obs.loc["DSP-1001250002642-A02", "dcc_unsectioned"] == "SOMEHASH100000000000"
    uns = table.uns["spatialdata_io"]
    assert (uns["observation_unit"], uns["unit_size_um"]) == ("region", None)
    assert uns["no_coordinates"] is True and "aois" not in sdata.shapes
    assert uns["annotation_join"]["n_matched"] == 2 and uns["pkc"]["Hs_R_NGS_CTA_v1.0.pkc"]["module"] == "Hs_R_NGS_CTA"
    assert uns["worksheet_header"][0] == "Experiment Summary"
    # the no-template control has no DCC: kept aside
    assert list(sdata["table_annotation_unmatched"].obs_names) == ["DSP-1001250002642-A01"]

    sdata.write(tmp_path / "x.zarr")
    back = read_zarr(tmp_path / "x.zarr")
    assert set(back.tables) == {"table", "table_probe", "table_annotation_unmatched"}


def test_protein_kit_xlsx_annotation_and_shapes(tmp_path: Path) -> None:
    dcc, rna = _ngs_kit(tmp_path)
    prot = _pkc(
        tmp_path / "Hs_P_NGS_Core_v1.0.pkc",
        "Human NGS Protein Core",
        "Protein",
        [("Ki-67", "Endogenous", ["RTS0020790"]), ("Rb IgG", "Negative", ["RTS0020791"])],
    )
    for name in ("DSP-1001250002642-A02.dcc",):
        text = (dcc / name).read_text().replace("</Code_Summary>", "RTS0020790,11\nRTS0020791,2\n</Code_Summary>")
        (dcc / name).write_text(text)
    # columns as in GeomxTools' DSP_Proteogenomics_Example_Data/Annotation.xlsx; GEO-style file names: match on the
    # DCC Scan_Attributes ID
    (dcc / "DSP-1001250002642-A02.dcc").rename(dcc / "GSM1_DSP-1001250002642-A02.dcc")
    (dcc / "DSP-1001250002642-A03.dcc.gz").rename(dcc / "GSM2_DSP-1001250002642-A03.dcc.gz")
    ann = pd.DataFrame(
        {
            "Sample_ID": ["DSP-1001250002642-A02", "DSP-1001250002642-A03"],
            "Segment_Type": ["Immune", "Tumor"],
            "ROI": [1, 1],
            "area": [6657.69, 20599.14],
            "nuclei": [97, 192],
            "X": [117.891, 117.893],
            "Y": [32.122, 32.14],
        }
    )
    ann.to_excel(tmp_path / "Annotation.xlsx", sheet_name="Annotations", index=False)

    sdata = geomx(
        dcc, [rna, prot], tmp_path / "Annotation.xlsx", annotation_sheet="Annotations", x="X", y="Y", area="area"
    )

    tp = sdata["table_protein"]
    assert list(tp.var_names) == ["Ki-67", "Rb IgG"]
    assert list(tp.var["feature_types"]) == ["Protein Expression", "Negative Control Probe"]
    np.testing.assert_array_equal(tp.X.toarray(), [[11, 2], [0, 0]])
    assert "Ki-67" not in sdata["table"].var_names
    assert sdata["table"].obs["Segment_Type"].tolist() == ["Immune", "Tumor"]
    assert sdata["table"].uns["spatialdata_io"]["annotation_join"]["matched_on"] == "dcc_ID"
    circles = sdata.shapes["aois"]
    assert circles.geometry.iloc[0].x == 117.891
    np.testing.assert_allclose(circles["radius"], np.sqrt(ann["area"] / np.pi))
    assert set(get_transformation(circles, get_all=True)) == {"aoi_coordinates"}  # no image frame stated
    np.testing.assert_array_equal(sdata["table_protein"].obs["instance_id"], [0, 1])
    sdata.write(tmp_path / "p.zarr")


def test_matrix_and_annotation_like_figshare_16818415(tmp_path: Path) -> None:
    # Annotation file_AOIs.txt (tab, an unnamed column) and Unnormalized_expression_data.csv (';' and decimal ',')
    (tmp_path / "Annotation file_AOIs.txt").write_text(
        "Scan name\t\tROI (label)\tSegment (Name/ Label)\tSegment tags\tROI_ID\tScan_ID\tTypes\tTissue\n"
        "ESD1_ESD2 CTA\t\t1\tVimentin+\tVimentin+\t1\tESD1_ESD2 CTA\tNormal\tESD1\n"
        "ESD1_ESD2 CTA\t\t1\tPanCK+\tPanCK+\t1\tESD1_ESD2 CTA\tNormal\tESD1\n"
        "ESD1_ESD2 CTA\t\t2\tPanCK+\tPanCK+\t2\tESD1_ESD2 CTA\tTransition\tESD1\n"
    )
    (tmp_path / "Unnormalized_expression_data.csv").write_text(
        "TargetName;ESD1_ESD2 CTA | 001 | PanCK+;ESD1_ESD2 CTA | 001 | Vimentin+\n"
        "CASP3;62,80944556;55,73870159\nNegative Probe;5,5;3,25\n"
    )
    pkc = _pkc(
        tmp_path / "k_v1.0.pkc",
        "k",
        "RNA",
        [("CASP3", "Endogenous01", ["R1"]), ("Negative Probe", "Negative01", ["R2"])],
    )

    sdata = geomx(
        pkc=pkc,
        counts=tmp_path / "Unnormalized_expression_data.csv",
        annotation=tmp_path / "Annotation file_AOIs.txt",
        matrix_sep=";",
        decimal=",",
        aoi_id="{Scan name} | {ROI (label):03d} | {Segment (Name/ Label)}",
    )

    t = sdata["table"]
    np.testing.assert_allclose(t.X.toarray(), [[62.80944556, 5.5], [55.73870159, 3.25]])
    assert list(t.var["is_control"]) == [False, True]
    uns = t.uns["spatialdata_io"]
    assert uns["counts_integer"] is False and uns["no_coordinates"] is True
    renamed = dict(map(tuple, uns["renamed_columns"]))
    assert renamed["Segment (Name/ Label)"] in t.obs.columns
    assert t.obs.loc["ESD1_ESD2 CTA | 001 | PanCK+", "Types"] == "Normal"
    assert renamed["Unnamed: 1"] in t.obs.columns  # the unnamed column is kept
    assert list(sdata["table_annotation_unmatched"].obs_names) == ["ESD1_ESD2 CTA | 002 | PanCK+"]
    assert "table_probe" not in sdata.tables
    sdata.write(tmp_path / "m.zarr")


def test_normalised_only_dsp_export_like_figshare_30505616(tmp_path: Path) -> None:
    # sheets and columns of All_Data_Sets_Normalization-to-House-Keeping-Genes.xlsx (DSP Analysis Suite export)
    path = tmp_path / "All_Data_Sets_Normalization-to-House-Keeping-Genes.xlsx"
    names = ["Grp3_MI-PATCH_2 | 001 | Cardiomyocyte", "Grp3_MI-PATCH_2 | 001 | Tissue"]
    seg = pd.DataFrame(
        {
            "SlideName": ["Grp3_MI-PATCH_2"] * 2,
            "ROILabel": ["001", "001"],
            "SegmentLabel": ["Cardiomyocyte", "Tissue"],
            "SegmentDisplayName": names,
            "AOISurfaceArea": [124840.912887, 253317.591904],
            "AOINucleiCount": [0, 1],
            "Region": ["Border", "Border"],  # clashes with the table's region key
            "ROICoordinateX": [11725, 11725],
            "ROICoordinateY": [11386, 11386],
        }
    )
    # a matrix AOI without a SegmentProperties row has no coordinates
    counts = pd.DataFrame(
        {"TargetName": ["Actb", "NegProbe-WTX"], names[0]: [28.07, 3.1], names[1]: [41.6, 2.2], "x | 002 | y": [1, 2]}
    )
    with pd.ExcelWriter(path) as w:
        seg.to_excel(w, sheet_name="SegmentProperties", index=False)
        counts.to_excel(w, sheet_name="TargetCountMatrix", index=False)

    sdata = geomx(
        layers={"normalized_housekeeping": path},
        matrix_sheet="TargetCountMatrix",
        annotation=path,
        annotation_sheet="SegmentProperties",
        aoi_id="SegmentDisplayName",
        x="ROICoordinateX",
        y="ROICoordinateY",
        aoi_radius=50,
    )

    t = sdata["table"]
    assert t.X is None and t.uns["spatialdata_io"]["no_raw_counts"] is True
    np.testing.assert_allclose(t.layers["normalized_housekeeping"], [[28.07, 3.1], [41.6, 2.2]])
    assert t.obs["AOINucleiCount"].tolist() == [0, 1]
    assert t.obs["Region_"].tolist() == ["Border", "Border"]
    assert sdata.shapes["aois"]["radius"].tolist() == [50, 50]
    assert list(sdata["table_no_coordinates"].obs_names) == ["x | 002 | y"]
    assert t.uns["spatialdata_io"]["n_no_coordinates"] == 1
    assert t.uns["spatialdata_io"]["radius_source"] == "stated"
    # no PKC: the NegProbe- name rule types the negative pool
    assert list(t.var["feature_types"]) == ["Gene Expression", "Negative Control Probe"]
    assert t.uns["spatialdata_io"]["negprobe_rule"]["n_targets"] == 1
    sdata.write(tmp_path / "n.zarr")


def test_negprobe_rule_without_pkc(tmp_path: Path) -> None:
    # figshare 16818415 names its pool "NegProbe" (no dash): it stays Gene Expression
    (tmp_path / "m.csv").write_text(
        "TargetName,s | 001 | a,s | 002 | a\nActb,5,6\nNegProbe-WTX,1,2\nNegProbe,3,4\nxNegProbe-WTX,0,1\n"
    )
    t = geomx(counts=tmp_path / "m.csv")["table"]
    assert t.var["feature_types"].tolist() == [
        "Gene Expression",
        "Negative Control Probe",
        "Gene Expression",
        "Gene Expression",
    ]
    assert t.var["is_control"].tolist() == [False, True, False, False]
    assert t.n_vars == 4  # nothing dropped
    assert dict(t.uns["spatialdata_io"]["negprobe_rule"]) == {
        "prefix": "NegProbe-",
        "applies_to": "targets that no PKC or deposited CodeClass types",
        "n_targets": 1,
    }


def test_negprobe_rule_leaves_pkc_and_codeclass_typing_alone(tmp_path: Path) -> None:
    # a PKC types every target: no rule, and a PKC "Endogenous" NegProbe- target stays Gene Expression
    (tmp_path / "m.csv").write_text("TargetName,s | 001 | a\nNegProbe-X,1\nNeg,2\n")
    pkc = _pkc(tmp_path / "k_v1.0.pkc", "k", "RNA", [("NegProbe-X", "Endogenous", ["R1"]), ("Neg", "Negative", ["R2"])])
    t = geomx(pkc=pkc, counts=tmp_path / "m.csv")["table"]
    assert t.var["is_control"].tolist() == [False, True]
    assert "negprobe_rule" not in t.uns["spatialdata_io"]
    # probe level without a CodeClass column: the rule types the target and its probes
    (tmp_path / "p.csv").write_text("ProbeName,TargetName,s | 001 | a\n1,GAL,6\n2,NegProbe-WTX,2\n3,NegProbe-WTX,1\n")
    sdata = geomx(counts=tmp_path / "p.csv", probe_level=True, feature_columns=["TargetName"])
    assert sdata["table"].var["is_control"].tolist() == [False, True]
    assert sdata["table_probe"].var["is_control"].tolist() == [False, True, True]
    assert sdata["table"].uns["spatialdata_io"]["negprobe_rule"]["n_targets"] == 1


def test_segment_map_like_figshare_30505616(tmp_path: Path) -> None:
    # TargetCountMatrix names a segment "Desmin" that SegmentProperties calls "Cardiomyocyte"
    (tmp_path / "ann.csv").write_text(
        "SegmentDisplayName,SegmentLabel,AOINucleiCount\n"
        "Grp3_MI-PATCH_2 | 001 | Cardiomyocyte,Cardiomyocyte,0\n"
        "Grp3_MI-PATCH_2 | 001 | Tissue,Tissue,1\n"
    )
    (tmp_path / "m.csv").write_text(
        "TargetName,Grp3_MI-PATCH_2 | 001 | Desmin,Grp3_MI-PATCH_2 | 001 | Tissue,Grp3_MI-PATCH_2 | 002 | Other\n"
        "Actb,1,2,3\n"
    )
    kw = {"counts": tmp_path / "m.csv", "annotation": tmp_path / "ann.csv", "aoi_id": "SegmentDisplayName"}

    plain = geomx(**kw)
    assert len(plain["table_annotation_unmatched"]) == 1  # default: the Desmin AOI misses its row
    assert "segment_deposited" not in plain["table"].obs and "segment_map" not in plain["table"].uns["spatialdata_io"]

    sdata = geomx(**kw, segment_map={"Desmin": "Cardiomyocyte"})
    t = sdata["table"]
    assert "table_annotation_unmatched" not in sdata.tables
    # the AOIs keep their deposited names; the Desmin AOI now carries the Cardiomyocyte row
    assert list(t.obs_names) == [
        "Grp3_MI-PATCH_2 | 001 | Desmin",
        "Grp3_MI-PATCH_2 | 001 | Tissue",
        "Grp3_MI-PATCH_2 | 002 | Other",
    ]
    assert t.obs["SegmentLabel"].tolist()[:2] == ["Cardiomyocyte", "Tissue"]
    assert t.obs["segment_deposited"].tolist() == ["Desmin", "Tissue", "Other"]
    smap = t.uns["spatialdata_io"]["segment_map"]
    assert dict(smap["map"]) == {"Desmin": "Cardiomyocyte"} and smap["n_aois"] == 1
    assert t.uns["spatialdata_io"]["annotation_join"]["n_matched"] == 2
    sdata.write(tmp_path / "s.zarr")
    back = read_zarr(tmp_path / "s.zarr")["table"]
    assert back.obs["segment_deposited"].tolist() == ["Desmin", "Tissue", "Other"]

    with pytest.raises(ValueError, match="name no AOI segment"):
        geomx(**kw, segment_map={"Vimentin": "x"})
    with pytest.raises(ValueError, match="needs `annotation`"):
        geomx(counts=tmp_path / "m.csv", segment_map={"Desmin": "Cardiomyocyte"})


def test_probe_level_matrix_like_mendeley_gmgcdhmmdx(tmp_path: Path) -> None:
    # InitialDataset.csv: probe rows with their metadata, then one column per AOI
    (tmp_path / "InitialDataset.csv").write_text(
        "ProbeName,ProbeDisplayName,TargetName,HUGOSymbol,AnalyteType,CodeClass,PR79_2_WTA | 001 | vessel,"
        "PR79_2_WTA | 004 | Full ROI\n"
        "45635,GAL_01,GAL,GAL,RNA,Endogenous,6,79\n"
        "1,NegProbe-WTX_01,NegProbe-WTX,,RNA,Negative,2,3\n"
        "2,NegProbe-WTX_02,NegProbe-WTX,,RNA,Negative,1,4\n"
    )
    sdata = geomx(
        counts=tmp_path / "InitialDataset.csv",
        probe_level=True,
        feature_columns=["ProbeDisplayName", "TargetName", "HUGOSymbol", "AnalyteType", "CodeClass"],
    )
    t, p = sdata["table"], sdata["table_probe"]
    assert list(t.var_names) == ["GAL", "NegProbe-WTX"]
    np.testing.assert_array_equal(t.X.toarray(), [[6, 3], [79, 7]])
    assert list(t.var["feature_types"]) == ["Gene Expression", "Negative Control Probe"]
    assert p.n_vars == 3 and p.var.loc["45635", "ProbeDisplayName"] == "GAL_01"
    assert t.uns["spatialdata_io"]["counts_integer"] is True


def test_ome_tiff_scales_aois_in_image_frame(tmp_path: Path) -> None:
    img = np.zeros((2, 32, 48), dtype=np.uint16)
    tifffile.imwrite(
        tmp_path / "slide.ome.tiff",
        img,
        ome=True,
        metadata={"axes": "CYX", "Channel": {"Name": ["DNA", "PanCK"]}, "PhysicalSizeX": 0.4, "PhysicalSizeY": 0.4},
    )
    pd.DataFrame({"AOI": ["a", "b"], "x": [10, 20], "y": [5, 6]}).to_csv(tmp_path / "ann.csv", index=False)
    pd.DataFrame({"a": [1], "b": [2]}, index=pd.Index(["G"], name="TargetName")).to_csv(tmp_path / "c.csv")

    sdata = geomx(
        counts=tmp_path / "c.csv",
        annotation=tmp_path / "ann.csv",
        image=tmp_path / "slide.ome.tiff",
        x="x",
        y="y",
        coordinates_in_image=True,
    )
    im = sdata.images["slide_image"]
    assert list(im.coords["c"].values) == ["DNA", "PanCK"] and im.shape == (2, 32, 48)
    for el in (im, sdata.shapes["aois"]):
        assert get_transformation(el, "global").scale.tolist() == pytest.approx([0.4, 0.4])
    assert sdata["table"].uns["spatialdata_io"]["um_per_px"] == pytest.approx(0.4)


def test_input_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Pass `dcc`"):
        geomx()
    with pytest.raises(ValueError, match="needs `pkc`"):
        geomx(tmp_path)
