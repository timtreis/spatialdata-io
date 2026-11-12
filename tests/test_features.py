import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse
from spatialdata import SpatialData, read_zarr
from spatialdata.models import TableModel, get_table_keys

from spatialdata_io.readers._utils._features import (
    BLANK_NAME_RULES,
    COSMX_NAME_RULES,
    feature_types_by_name,
    split_by_feature_type,
)

TYPES = [
    "Gene Expression",
    "Negative Control Probe",
    "Gene Expression",
    "Protein Expression",
    "Blank Codeword",
    "CRISPR Guide Capture",
    "Antibody Capture",
    "Genomic Control",
]


def _adata(types: list[str]) -> AnnData:
    n_obs, n_var = 4, len(types)
    x = sparse.csr_matrix(np.arange(n_obs * n_var, dtype=np.float32).reshape(n_obs, n_var))
    var = pd.DataFrame({"feature_types": types, "gene_ids": [f"ID{i}" for i in range(n_var)]})
    var.index = [f"f{i}" for i in range(n_var)]
    obs = pd.DataFrame({"cell_id": np.arange(n_obs), "region": pd.Categorical(["cells"] * n_obs)})
    obs.index = [f"c{i}" for i in range(n_obs)]
    adata = AnnData(x, obs=obs, var=var)
    adata.obsm["spatial"] = np.zeros((n_obs, 2))
    return TableModel.parse(adata, region="cells", region_key="region", instance_key="cell_id")


def _column(adata: AnnData, name: str) -> np.ndarray:
    return adata[:, name].X.toarray().ravel()


def test_split_keeps_every_feature() -> None:
    adata = _adata(TYPES)
    tables = split_by_feature_type(adata)

    assert set(tables) == {"table", "table_protein", "table_crispr_guide_capture"}
    assert tables["table"].var_names.tolist() == ["f0", "f1", "f2", "f4", "f7"]
    assert tables["table_protein"].var_names.tolist() == ["f3", "f6"]
    names = [n for t in tables.values() for n in t.var_names]
    assert len(names) == adata.n_vars == len(set(names))
    for t in tables.values():
        assert t.obs.equals(adata.obs)
        assert get_table_keys(t) == get_table_keys(adata)
        for n in t.var_names:
            np.testing.assert_array_equal(_column(t, n), _column(adata, n))
            assert t.var.loc[n, "feature_types"] == adata.var.loc[n, "feature_types"]


def test_drop_controls_only_touches_the_main_table(caplog: pytest.LogCaptureFixture) -> None:
    adata = _adata(TYPES)
    keep = split_by_feature_type(adata)
    with caplog.at_level(logging.INFO, logger="spatialdata"):
        drop = split_by_feature_type(adata, controls="drop")
    assert drop["table"].var_names.tolist() == ["f0", "f2"]
    assert (drop["table"].var["feature_types"] == "Gene Expression").all()
    assert "Dropped 3 control feature(s)" in caplog.text
    assert drop.keys() == keep.keys()
    for key in ("table_protein", "table_crispr_guide_capture"):
        assert drop[key].var_names.equals(keep[key].var_names)


def test_genes_only_is_returned_as_is() -> None:
    adata = _adata(["Gene Expression"] * 3)
    assert split_by_feature_type(adata)["table"] is adata
    assert split_by_feature_type(adata, controls="drop")["table"] is adata


def test_no_main_table_without_its_features() -> None:
    tables = split_by_feature_type(_adata(["Antibody Capture"] * 2))
    assert set(tables) == {"table_protein"}


def test_bad_controls_value() -> None:
    with pytest.raises(ValueError, match="'keep' or 'drop'"):
        split_by_feature_type(_adata(TYPES), controls="none")  # type: ignore[arg-type]


def test_name_prefix() -> None:
    tables = split_by_feature_type(_adata(TYPES), name="square_008um")
    assert set(tables) == {"square_008um", "square_008um_protein", "square_008um_crispr_guide_capture"}


def test_round_trip(tmp_path: Path) -> None:
    adata = _adata(TYPES)
    tables = split_by_feature_type(adata)
    SpatialData(tables=tables).write(tmp_path / "x.zarr")
    back = read_zarr(tmp_path / "x.zarr")
    for key, t in tables.items():
        assert back[key].var_names.equals(t.var_names)
        assert back[key].var["feature_types"].tolist() == t.var["feature_types"].tolist()
        np.testing.assert_array_equal(back[key].X.toarray(), t.X.toarray())


def test_cosmx_name_rules() -> None:
    names = ["NegPrb1", "Negative3", "NegativeAdd", "SystemControl12", "FalseCode5", "NEGR1", "PIGG", "ACTB"]
    assert feature_types_by_name(names, COSMX_NAME_RULES).tolist() == [
        "Negative Control Probe",
        "Negative Control Probe",
        "Negative Control Probe",
        "System Control",
        "System Control",
        "Gene Expression",
        "Gene Expression",
        "Gene Expression",
    ]


def test_blank_name_rules() -> None:
    names = ["Blank-1", "BLANK_0004", "blank", "Ccl2"]
    assert feature_types_by_name(names, BLANK_NAME_RULES).tolist() == ["Blank Codeword"] * 3 + ["Gene Expression"]
