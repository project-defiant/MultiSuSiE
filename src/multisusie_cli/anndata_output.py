"""Write complete MultiSuSiE fits as AnnData objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from .models import RunParameters
from .preparation import PreparedLocus
from .runner import MultiSuSiEFit


def write_anndata(
    fit: MultiSuSiEFit,
    prepared: PreparedLocus,
    parameters: RunParameters,
    output: Path,
) -> None:
    """Write all modeled components and provenance to one H5AD file."""
    raw = fit.raw
    n_components = raw.alpha.shape[0]
    component_indices = list(range(n_components))
    purity = np.asarray(raw.sets[1], dtype=np.float32)
    coverage = np.asarray(raw.sets[2], dtype=np.float32)
    passing = np.asarray(raw.sets[3], dtype=bool)
    obs = pd.DataFrame(
        {
            "componentIndex": component_indices,
            "lbf": np.asarray(raw.lbf, dtype=np.float32),
            "KL": np.asarray(raw.KL, dtype=np.float32),
            "credibleSetCoverage": coverage,
            "credibleSetPurity": purity,
            "credibleSetPass": passing,
        },
        index=[f"component_{index}" for index in component_indices],
    )
    var = pd.DataFrame(
        {
            "variantId": prepared.variant_ids,
            "chromosome": prepared.chromosomes,
            "position": prepared.positions,
            "pip": np.asarray(raw.pip, dtype=np.float32),
            "pi": np.asarray(raw.pi, dtype=np.float32),
        },
        index=prepared.variant_ids,
    )
    populations = prepared.populations
    ancestry_names = [population.ancestry for population in populations]
    layers: dict[str, np.ndarray] = {
        f"mu__{population.ancestry}": np.asarray(raw.mu[index], dtype=np.float32)
        for index, population in enumerate(populations)
    }
    for left_index, left in enumerate(ancestry_names):
        for right_index, right in enumerate(ancestry_names):
            layers[f"mu2__{left}__{right}"] = np.asarray(
                raw.mu2[left_index, right_index], dtype=np.float32
            )
    adata = ad.AnnData(
        X=np.asarray(raw.alpha, dtype=np.float32),
        obs=obs,
        var=var,
        layers=layers,
    )
    adata.obsm["V"] = np.asarray(raw.V, dtype=np.float32).T
    adata.varm["coef"] = np.asarray(raw.coef, dtype=np.float32).T
    adata.varm["coefSd"] = np.asarray(raw.coef_sd, dtype=np.float32).T
    adata.varm["variantPresent"] = np.asarray(
        [population.variant_present for population in populations], dtype=bool
    ).T
    adata.uns.update(
        {
            "runId": prepared.run_id,
            "fineMappingLocusSetId": prepared.fine_mapping_locus_set_id,
            "ancestries": ancestry_names,
            "studyIds": [population.study_id for population in populations],
            "populationSizes": [population.sample_size for population in populations],
            "rho": parameters.rho,
            "methodParameters": parameters.model_dump(mode="json"),
            "converged": fit.converged,
            "niter": int(raw.niter),
            "sigma2": _json_value(raw.sigma2),
            "ER2": _json_value(raw.ER2),
            "elbo": _json_value(raw.elbo),
            "inputMode": "z-score",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output)


def _json_value(value: Any) -> Any:
    """Convert NumPy scalars/arrays into AnnData-compatible metadata."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
