"""Public behavior tests for the MultiSuSiE execution boundary."""

import numpy as np

from multisusie_cli.models import RunParameters
from multisusie_cli.preparation import PopulationArrays, PreparedLocus
from multisusie_cli.runner import FitQualityError, run_multisusie


def test_runner_returns_a_converged_reportable_fit() -> None:
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

    fit = run_multisusie(prepared, RunParameters(L=2, max_iter=30))

    assert fit.converged is True
    assert fit.passing_component_indices == [0]
    assert fit.raw.variant_ids == prepared.variant_ids


def test_runner_rejects_non_converged_fit() -> None:
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

    with np.testing.assert_raises_regex(FitQualityError, "did not converge"):
        run_multisusie(prepared, RunParameters(L=2, max_iter=1))
