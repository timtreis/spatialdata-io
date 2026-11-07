from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from spatialdata_io.readers._utils._utils import _add_gene_id_and_genome


def _table(names: list[str], **columns: list[str]) -> ad.AnnData:
    return ad.AnnData(np.ones((2, len(names))), var=pd.DataFrame(columns, index=names))


def test_plain_and_versioned_ids_read_by_species() -> None:
    v = _add_gene_id_and_genome(_table(["TP53", "Trp53"], gene_ids=["ENSG00000141510.17", "ENSMUSG00000059552"])).var
    assert v["gene_id"].tolist() == ["ENSG00000141510", "ENSMUSG00000059552"]
    assert v["genome_ref"].tolist() == ["human", "mouse"]


def test_prefixed_combined_reference_keeps_what_was_deposited() -> None:
    ids = ["GRCh38_ENSG00000141510", "mm10___ENSMUSG00000059552"]
    t = _table(["GRCh38_TP53", "mm10___Trp53"], gene_ids=ids)
    X = t.X.copy()
    v = _add_gene_id_and_genome(t).var
    assert v["gene_id"].tolist() == ["ENSG00000141510", "ENSMUSG00000059552"]
    assert v["genome_ref"].tolist() == ["GRCh38", "mm10"]
    assert v["gene_ids"].tolist() == ids and t.var_names.tolist() == ["GRCh38_TP53", "mm10___Trp53"]
    assert np.array_equal(t.X, X)


def test_genome_column_wins_unless_unknown() -> None:
    v = _add_gene_id_and_genome(
        _table(["a", "b", "c"], gene_ids=["ENSG1", "GRCh38_ENSG2", "ENSMUSG3"], genome=["hg19", "unknown", "mm39"])
    ).var
    assert v["genome_ref"].tolist() == ["hg19", "GRCh38", "mm39"]


def test_symbols_retired_ids_and_pathogen_genes_get_no_id() -> None:
    v = _add_gene_id_and_genome(
        _table(["TP53", "OLD", "PBANKA_0100"], gene_ids=["TP53", "DEPRECATED_ENSG00000000001", "PBANKA_0100"])
    ).var
    assert v["gene_id"].isna().all()
    assert v["genome_ref"].tolist() == ["unknown", "unknown", "unknown"]


def test_no_gene_ids_column_reads_var_names() -> None:
    v = _add_gene_id_and_genome(_table(["hg38_ENSG00000141510", "ACTB"])).var
    assert v["gene_id"].tolist()[0] == "ENSG00000141510" and pd.isna(v["gene_id"].tolist()[1])
    assert v["genome_ref"].tolist() == ["hg38", "unknown"]
