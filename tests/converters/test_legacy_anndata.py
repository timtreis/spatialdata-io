from typing import Literal

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from anndata.tests.helpers import assert_equal
from scipy.sparse import csr_matrix
from spatialdata import SpatialData, get_element_instances
from spatialdata.datasets import blobs
from spatialdata.models import TableModel
from spatialdata.testing import assert_spatial_data_objects_are_identical
from spatialdata.transformations import get_transformation, set_transformation

from spatialdata_io.converters.legacy_anndata import (
    from_legacy_anndata,
    to_legacy_anndata,
)

BlobsTypes = Literal["blobs_labels", "blobs_circles", "blobs_polygons", "blobs_multipolygons"]


def blobs_annotating_element(name: BlobsTypes) -> SpatialData:
    sdata = blobs(length=50)
    if name == "blobs_labels":
        instance_id = get_element_instances(sdata[name]).tolist()
    else:
        instance_id = sdata[name].index.tolist()
    n = len(instance_id)
    new_table = AnnData(shape=(n, 0), obs={"region": [name for _ in range(n)], "instance_id": instance_id})
    new_table = TableModel.parse(new_table, region=name, region_key="region", instance_key="instance_id")
    del sdata["table"]
    sdata["table"] = new_table
    return sdata


def _adjust_table_for_idempotency_check(sdata: SpatialData, include_images: bool) -> SpatialData:
    """We need to adjust the coordinate systems from from_legacy_anndata() to perform the idempotency check."""
    if not include_images:
        # nothing needs to be adjusted
        return sdata
    CS_NAME = "blobs_multiscale_image_downscaled_hires"
    sdata = sdata.filter_by_coordinate_system(CS_NAME)
    sdata.rename_coordinate_systems({CS_NAME: "global"})
    t = get_transformation(sdata["locations"], "global")
    set_transformation(sdata["locations"], {"global": t}, set_all=True)
    IMG_NAME = "blobs_multiscale_image_hires_image"
    im = sdata.images[IMG_NAME]
    del sdata.images[IMG_NAME]
    sdata.images["blobs_multiscale_image"] = im
    return sdata


def idempotency_check_to_anndata(sdata0: SpatialData, include_images: bool) -> None:
    adata0 = to_legacy_anndata(sdata0, include_images=include_images)
    sdata1 = from_legacy_anndata(adata0)
    sdata1 = _adjust_table_for_idempotency_check(sdata1, include_images=include_images)
    adata2 = to_legacy_anndata(sdata1, include_images=include_images)
    assert_equal(adata0, adata2)


def idempotency_check_from_anndata(adata0: SpatialData, include_images: bool) -> None:
    sdata0 = from_legacy_anndata(adata0)
    sdata0 = _adjust_table_for_idempotency_check(sdata0, include_images=include_images)
    adata1 = to_legacy_anndata(sdata0, include_images=include_images)
    sdata1 = from_legacy_anndata(adata1)
    sdata1 = _adjust_table_for_idempotency_check(sdata1, include_images=include_images)
    assert_spatial_data_objects_are_identical(sdata0, sdata1)


@pytest.mark.parametrize("name", ["blobs_labels", "blobs_circles", "blobs_polygons", "blobs_multipolygons"])
@pytest.mark.parametrize("include_images", [False, True])
def test_bidirectional_conversion(name: BlobsTypes, include_images: bool) -> None:
    if include_images:
        pytest.skip(
            "include_images=True can't be tested because the bug "
            "https://github.com/scverse/spatialdata/issues/165 causes a large error"
        )
    sdata0 = blobs_annotating_element(name)
    adata0 = to_legacy_anndata(sdata0, include_images=include_images)
    sdata1 = from_legacy_anndata(adata0)
    sdata1 = _adjust_table_for_idempotency_check(sdata1, include_images=include_images)

    idempotency_check_to_anndata(sdata1, include_images=include_images)
    idempotency_check_from_anndata(adata0, include_images=include_images)


def _two_library_adata() -> AnnData:
    rng = np.random.default_rng(0)
    n_obs = {"A": 3, "B": 5}
    library = np.repeat(list(n_obs), list(n_obs.values()))
    adata = AnnData(
        X=csr_matrix(rng.integers(0, 50, size=(len(library), 4)).astype(np.float32)),
        obs=pd.DataFrame({"sample": pd.Categorical(library)}, index=[f"spot{i}" for i in range(len(library))]),
    )
    adata.obsm["spatial"] = rng.uniform(0, 100, size=(len(library), 2))
    adata.uns["spatial"] = {
        lib: {
            "images": {"hires": np.full((20, 30, 3), i, dtype=np.uint8)},
            "scalefactors": {"tissue_hires_scalef": 0.1 * (i + 1), "spot_diameter_fullres": 10.0 * (i + 1)},
        }
        for i, lib in enumerate(n_obs)
    }
    return adata


@pytest.mark.parametrize("library_id", ["A", "B"])
def test_from_legacy_anndata_library_id(library_id: str) -> None:
    adata = _two_library_adata()
    sdata = from_legacy_anndata(adata, library_id=library_id, library_key="sample")
    mask = (adata.obs["sample"] == library_id).to_numpy()
    i = ["A", "B"].index(library_id)
    cs = f"{library_id}_hires"

    assert list(sdata.images) == [cs]
    assert int(sdata.images[cs].data[0, 0, 0]) == i
    table = sdata["table"]
    assert table.obs_names.tolist() == adata.obs_names[mask].tolist()
    np.testing.assert_array_equal(table.X.toarray(), adata.X[mask].toarray())
    shapes = sdata["locations"]
    np.testing.assert_array_equal(np.column_stack([shapes.geometry.x, shapes.geometry.y]), adata.obsm["spatial"][mask])
    assert shapes["radius"].iloc[0] == 5.0 * (i + 1)
    assert set(get_transformation(shapes, get_all=True)) == {cs}
    np.testing.assert_allclose(get_transformation(shapes, to_coordinate_system=cs).scale, [0.1 * (i + 1)] * 2)
    assert list(adata.uns["spatial"]) == ["A", "B"]  # input not modified


def test_from_legacy_anndata_library_value() -> None:
    # obs holds a short sample name while uns['spatial'] is keyed by the longer Space Ranger run name
    adata = _two_library_adata()
    adata.uns["spatial"] = {f"run_{k}_GRCh38": v for k, v in adata.uns["spatial"].items()}
    sdata = from_legacy_anndata(adata, library_id="run_B_GRCh38", library_key="sample", library_value="B")
    assert list(sdata.images) == ["run_B_GRCh38_hires"]
    assert set(sdata["table"].obs["sample"]) == {"B"}
    assert sdata["table"].n_obs == 5


def test_from_legacy_anndata_library_id_errors() -> None:
    adata = _two_library_adata()
    with pytest.raises(ValueError, match="library_key"):
        from_legacy_anndata(adata, library_id="A")
    with pytest.raises(ValueError, match="not in adata.uns"):
        from_legacy_anndata(adata, library_id="C", library_key="sample")
    adata.obs["other"] = "x"
    with pytest.raises(ValueError, match="No observations"):
        from_legacy_anndata(adata, library_id="A", library_key="other")


def test_from_legacy_anndata_observation_unit() -> None:
    adata = _two_library_adata()
    for lib in adata.uns["spatial"].values():
        lib["metadata"] = {"chemistry_description": "Spatial 3' v1", "software_version": "spaceranger-1.3.0"}
    adata.var["gene_ids"] = [f"ENSG{i:011d}.1" for i in range(adata.n_vars)]
    table = from_legacy_anndata(adata, library_id="A", library_key="sample").tables["table"]
    uns = table.uns["spatialdata_io"]
    assert (uns["observation_unit"], uns["unit_size_um"]) == ("spot_or_bin", 55.0)
    assert table.var["gene_id"].iloc[0] == "ENSG00000000000" and set(table.var["genome_ref"]) == {"human"}
    # two libraries mixed, or no chemistry stated: the unit stays unset
    assert "spatialdata_io" not in from_legacy_anndata(adata).tables["table"].uns
    plain = _two_library_adata()
    assert "spatialdata_io" not in from_legacy_anndata(plain, library_id="A", library_key="sample").tables["table"].uns
