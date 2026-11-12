from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from spatialdata._logging import logger
from spatialdata.models import TableModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

FEATURE_TYPES = "feature_types"
GENE_EXPRESSION = "Gene Expression"
PROTEIN_FEATURE_TYPES = ("Protein Expression", "Antibody Capture")
# control types seen in the data (10x `feature_type`, the name rules below); they stay in the main table
CONTROL_FEATURE_TYPES = (
    "Negative Control Probe",
    "Negative Control Codeword",
    "Genomic Control",
    "Unassigned Codeword",
    "Deprecated Codeword",
    "Blank Codeword",
    "System Control",
)
# feature types by name, for platforms without a type column
COSMX_NAME_RULES = (
    (re.compile(r"^(NegPrb|Negative)"), "Negative Control Probe"),  # also NegativeAdd
    (re.compile(r"^(SystemControl|FalseCode)"), "System Control"),
)
BLANK_NAME_RULES = ((re.compile(r"blank", re.IGNORECASE), "Blank Codeword"),)


def feature_types_by_name(
    names: Sequence[str] | pd.Index, rules: Sequence[tuple[re.Pattern[str], str]], default: str = GENE_EXPRESSION
) -> np.ndarray:
    """The type of each feature name: that of the first matching rule, else `default`.

    Parameters
    ----------
    names
        The feature names.
    rules
        ``(pattern, feature type)`` pairs, matched with :meth:`re.Pattern.search`, e.g. :data:`COSMX_NAME_RULES` or
        :data:`BLANK_NAME_RULES`.
    default
        The type of a name no rule matches.
    """
    return np.array([next((t for pattern, t in rules if pattern.search(str(n))), default) for n in names], dtype=object)


def _table_key(name: str, feature_type: str) -> str:
    """The table a feature type goes to: `name`, ``<name>_protein`` or ``<name>_<slug>``."""
    if feature_type == GENE_EXPRESSION or feature_type in CONTROL_FEATURE_TYPES:
        return name
    if feature_type in PROTEIN_FEATURE_TYPES:
        return f"{name}_protein"
    return f"{name}_" + re.sub(r"[^0-9a-z]", "_", feature_type.lower())


def _subset(adata: AnnData, keep: np.ndarray) -> AnnData:
    """``adata[:, keep]``, re-parsed with the region, region key and instance key of `adata` if it has them."""
    sub = adata[:, keep].copy()
    attrs = sub.uns.pop(TableModel.ATTRS_KEY, None)
    if attrs is None:
        return sub
    return TableModel.parse(
        sub,
        region=attrs[TableModel.REGION_KEY],
        region_key=attrs[TableModel.REGION_KEY_KEY],
        instance_key=attrs[TableModel.INSTANCE_KEY],
    )


def split_by_feature_type(
    adata: AnnData, *, controls: Literal["keep", "drop"] = "keep", name: str = "table"
) -> dict[str, AnnData]:
    """Split a table typed by ``var["feature_types"]`` into the main table and one table per other feature type.

    ``Gene Expression`` and the control types (:data:`CONTROL_FEATURE_TYPES`) stay in `name`, protein
    (``Protein Expression``, ``Antibody Capture``) goes to ``<name>_protein`` and any other type to
    ``<name>_<type>``, lowercased with every other character than ``a-z`` and ``0-9`` replaced by ``_``. Each keeps
    the ``obs``, ``obsm`` and, for a parsed table, the region and instance keys of `adata`. Nothing is dropped
    unless `controls` is ``"drop"``.

    Parameters
    ----------
    adata
        Every feature of the source, with ``var["feature_types"]``.
    controls
        ``"keep"`` keeps the controls in `name`; ``"drop"`` removes them, so `name` holds only ``Gene Expression``.
        The other tables are never affected.
    name
        The name of the main table.

    Returns
    -------
    The tables by name; `adata` itself is the main table when it has nothing to move or drop.
    """
    if controls not in ("keep", "drop"):
        raise ValueError(f"`controls` must be 'keep' or 'drop', got {controls!r}.")
    types = adata.var[FEATURE_TYPES].astype(str).to_numpy()
    keys = np.array([_table_key(name, t) for t in types], dtype=object)
    main = keys == name
    if controls == "drop":
        is_gene = types == GENE_EXPRESSION
        if n := int((main & ~is_gene).sum()):
            logger.info(f"Dropped {n} control feature(s) from {name!r} (controls='drop').")
        main &= is_gene
    tables = {name: adata if main.all() else _subset(adata, main)}
    for key in dict.fromkeys(keys[keys != name]):
        tables[str(key)] = _subset(adata, keys == key)
    return tables
