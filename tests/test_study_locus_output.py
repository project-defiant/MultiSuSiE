"""Public behavior tests for Gentropy StudyLocus conversion."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from multisusie_cli.models import RunParameters
from multisusie_cli.preparation import PopulationArrays, PreparedLocus
from multisusie_cli.runner import run_multisusie
from multisusie_cli.study_locus_output import write_study_locus


def test_write_study_locus_emits_one_row_per_passing_component(tmp_path: Path) -> None:
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
                betas=np.array([0.8, 0.01, 0.02], dtype=np.float32),
                standard_errors=np.array([0.1, 0.1, 0.1], dtype=np.float32),
            )
        ],
    )
    parameters = RunParameters(L=2, max_iter=30)
    fit = run_multisusie(prepared, parameters)
    output = tmp_path / "nested" / "study_locus.parquet"

    write_study_locus(fit, prepared, parameters, output)
    result = pl.read_parquet(output)

    assert result.height == 1
    row = result.to_dicts()[0]
    assert row["finemappingMethod"] == "MultiSuSiE"
    assert row["studyId"] == "study-eur"
    assert row["variantId"] == "1_10_A_G"
    assert row["qualityControls"] == []
    assert row["credibleSetIndex"] == 0
    assert row["locus"][0]["variantId"] == "1_10_A_G"
    assert row["locus"][0]["posteriorProbability"] == 1.0
    assert row["beta"] == pytest.approx(0.8)
    assert row["standardError"] == pytest.approx(0.1)
    assert row["pValueExponent"] < 0
