"""Tests for the application boundary before numerical implementation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from multisusie_cli.models import RunInputs, RunParameters, StudyMetadata


def test_study_metadata_requires_positive_sample_size() -> None:
    with pytest.raises(ValidationError):
        StudyMetadata(studyId="study", ancestry="EUR", sampleSize=0)


def test_run_parameters_use_scalar_rho() -> None:
    parameters = RunParameters()

    assert parameters.rho == 0.75
    assert parameters.purity_min_r2 == 0.01
    assert "rho_matrix" not in RunParameters.model_fields


@pytest.mark.parametrize("value", [0.0, 1.0, -0.1, 1.1])
def test_run_parameters_require_strictly_bounded_purity_threshold(value: float) -> None:
    with pytest.raises(ValidationError):
        RunParameters(purity_min_r2=value)


def test_stats_model_records_purity_filtering_counts() -> None:
    from multisusie_cli.models import MultiSuSiEStats

    stats = MultiSuSiEStats(
        runId="run",
        fineMappingLocusSetId="set",
        status="SUCCESS",
        converged=True,
        purityMinR2Threshold=0.01,
        nModeledComponents=10,
        nPurityPassingComponents=4,
        nPurityFilteredComponents=6,
    )

    assert stats.purityMinR2Threshold == 0.01
    assert stats.nPurityFilteredComponents == 6


def test_run_inputs_require_existing_files(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="does not exist"):
        RunInputs(
            fine_mapping_locus_set=tmp_path / "locus.parquet",
            multi_ancestry_pairwise_ld=tmp_path / "ld.parquet",
            study_metadata=tmp_path / "metadata.jsonl",
            run_id="run",
            fine_mapping_locus_set_id="locus-set",
        )
