"""Public behavior tests for the extended AnnData result."""

from pathlib import Path

import anndata as ad
import numpy as np

from multisusie_cli.models import RunParameters
from multisusie_cli.preparation import PopulationArrays, PreparedLocus
from multisusie_cli.runner import run_multisusie
from multisusie_cli.anndata_output import write_anndata


def test_write_anndata_preserves_complete_fit_and_provenance(tmp_path: Path) -> None:
    prepared = PreparedLocus(
        run_id="run-1",
        fine_mapping_locus_set_id="set-1",
        variant_ids=["1_10_A_G", "1_20_C_T", "1_30_G_A"],
        chromosomes=["1", "1", "1"],
        positions=[10, 20, 30],
        populations=[
            PopulationArrays(
                ancestry="EUR",
                study_id="study-eur",
                sample_size=1000,
                variant_present=np.array([True, True, True]),
                z_scores=np.array([8.0, 0.1, 0.2], dtype=np.float32),
                ld_matrix=np.eye(3, dtype=np.float32),
            )
        ],
    )
    parameters = RunParameters(L=2, max_iter=30)
    fit = run_multisusie(prepared, parameters)
    output = tmp_path / "nested" / "extended_results.h5ad"

    write_anndata(fit, prepared, parameters, output)
    result = ad.read_h5ad(output)

    assert result.shape == (2, 3)
    assert result.var["variantId"].tolist() == prepared.variant_ids
    assert result.obs["componentIndex"].tolist() == [0, 1]
    assert "mu__EUR" in result.layers
    assert "mu2__EUR__EUR" in result.layers
    assert result.obsm["V"].shape == (2, 1)
    assert result.varm["coef"].shape == (3, 1)
    assert result.varm["variantPresent"].dtype == bool
    assert result.uns["runId"] == "run-1"
    assert result.uns["fineMappingLocusSetId"] == "set-1"
    assert result.uns["methodParameters"]["rho"] == 0.75
    assert "credibleSetPurityMinR2" in result.obs.columns
    assert "credibleSetPurity" not in result.obs.columns
    assert result.uns["purityMinR2Threshold"] == parameters.purity_min_r2
    assert result.uns["lowMemoryMode"] is False


def test_write_anndata_keeps_diagnostic_output_when_purity_is_missing(
    tmp_path: Path,
) -> None:
    prepared = PreparedLocus(
        run_id="run-1",
        fine_mapping_locus_set_id="set-1",
        variant_ids=["1_10_A_G", "1_20_C_T", "1_30_G_A"],
        chromosomes=["1", "1", "1"],
        positions=[10, 20, 30],
        populations=[
            PopulationArrays(
                ancestry="EUR",
                study_id="study-eur",
                sample_size=1000,
                variant_present=np.array([True, True, True]),
                z_scores=np.array([8.0, 0.1, 0.2], dtype=np.float32),
                ld_matrix=np.eye(3, dtype=np.float32),
            )
        ],
    )
    parameters = RunParameters(L=2, max_iter=30)
    fit = run_multisusie(prepared, parameters)
    fit.raw.sets = (fit.raw.sets[0], None, None, fit.raw.sets[3])
    fit.passing_component_indices = []
    output = tmp_path / "diagnostic.h5ad"

    write_anndata(fit, prepared, parameters, output)
    result = ad.read_h5ad(output)

    assert result.obs["credibleSetPurityMinR2"].isna().all()
    assert result.obs["credibleSetCoverage"].isna().all()
    assert not result.obs["credibleSetPass"].any()
