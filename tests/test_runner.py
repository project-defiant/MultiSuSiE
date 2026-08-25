"""Public behavior tests for the MultiSuSiE execution boundary."""

import numpy as np
from types import SimpleNamespace

from multisusie_cli.models import RunParameters
from multisusie_cli.preparation import PopulationArrays, PreparedLocus
from multisusie_cli.runner import run_multisusie
import multisusie_cli.runner as runner_module


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


def test_runner_returns_non_converged_fit_for_reporting() -> None:
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

    fit = run_multisusie(prepared, RunParameters(L=2, max_iter=1))

    assert fit.converged is False


def test_runner_treats_missing_purity_as_no_reportable_components(monkeypatch) -> None:
    prepared = PreparedLocus(
        run_id="run-1",
        fine_mapping_locus_set_id="set-1",
        variant_ids=["1_10_A_G"],
        chromosomes=["1"],
        positions=[10],
        populations=[],
    )
    raw = SimpleNamespace(
        converged=True,
        sets=(None, None, None, None),
        alpha=np.zeros((2, 1), dtype=np.float32),
    )
    monkeypatch.setattr(
        runner_module.MultiSuSiE,
        "multisusie_rss",
        lambda **_: raw,
    )

    fit = run_multisusie(prepared, RunParameters())

    assert fit.converged is True
    assert fit.passing_component_indices == []
