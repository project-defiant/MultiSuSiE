"""Public behavior tests for the MultiSuSiE execution boundary."""

import numpy as np
import pytest
from types import SimpleNamespace

from multisusie_cli.models import RunParameters
from multisusie_cli.preparation import PopulationArrays, PreparedLocus
from multisusie_cli.runner import FitQualityError, run_multisusie
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


def test_runner_reports_internal_numerical_assertion_as_fit_quality_error(
    monkeypatch,
) -> None:
    # Regression test: a population's residual-variance estimate going
    # negative (LD/summary-statistics mismatch) corrupts a matrix used in
    # MultiSuSiE's log-Bayes-factor computation, which raises a bare
    # AssertionError deep inside the vendored numerics. That must be reported
    # as a fit-quality problem with this locus, not crash the whole run.
    prepared = PreparedLocus(
        run_id="run-1",
        fine_mapping_locus_set_id="set-1",
        variant_ids=["1_10_A_G"],
        chromosomes=["1"],
        positions=[10],
        populations=[],
    )

    def _raise_assertion(**_):
        assert False, "logdet_Ainv_plus_Q_sign > 0"

    monkeypatch.setattr(
        runner_module.MultiSuSiE,
        "multisusie_rss",
        _raise_assertion,
    )

    with pytest.raises(FitQualityError):
        run_multisusie(prepared, RunParameters())
