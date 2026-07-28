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
    assert "rho_matrix" not in RunParameters.model_fields


def test_run_inputs_require_existing_files(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="does not exist"):
        RunInputs(
            fine_mapping_locus_set=tmp_path / "locus.parquet",
            multi_ancestry_pairwise_ld=tmp_path / "ld.parquet",
            study_metadata=tmp_path / "metadata.jsonl",
            run_id="run",
            fine_mapping_locus_set_id="locus-set",
        )
