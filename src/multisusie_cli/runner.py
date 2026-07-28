"""Execution boundary for the existing MultiSuSiE numerical implementation."""

from __future__ import annotations

from typing import Any

import MultiSuSiE
import numpy as np
from pydantic import BaseModel, ConfigDict

from .models import RunParameters
from .preparation import PreparedLocus


class FitQualityError(RuntimeError):
    """Raised when a MultiSuSiE fit cannot be reported."""


class MultiSuSiEFit(BaseModel):
    """Validated summary of a successful, reportable numerical fit."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    raw: Any
    converged: bool
    passing_component_indices: list[int]


def run_multisusie(prepared: PreparedLocus, parameters: RunParameters) -> MultiSuSiEFit:
    """Run MultiSuSiE and reject fits that cannot be reported."""
    populations = prepared.populations
    raw = MultiSuSiE.multisusie_rss(
        R_list=[population.ld_matrix for population in populations],
        z_list=[population.z_scores for population in populations],
        population_sizes=[population.sample_size for population in populations],
        rho=parameters.rho,
        L=parameters.L,
        scaled_prior_variance=parameters.scaled_prior_variance,
        pop_spec_standardization=parameters.pop_spec_standardization,
        estimate_residual_variance=parameters.estimate_residual_variance,
        estimate_prior_variance=parameters.estimate_prior_variance,
        estimate_prior_method=parameters.estimate_prior_method,
        pop_spec_effect_priors=parameters.pop_spec_effect_priors,
        iter_before_zeroing_effects=parameters.iter_before_zeroing_effects,
        prior_tol=parameters.prior_tol,
        max_iter=parameters.max_iter,
        tol=parameters.tol,
        coverage=parameters.coverage,
        min_abs_corr=parameters.min_abs_corr,
        float_type=np.float32,
        low_memory_mode=parameters.low_memory_mode,
        single_population_mac_thresh=0,
        multi_population_maf_thresh=0,
        variant_ids=prepared.variant_ids,
    )
    converged = bool(raw.converged)
    if not converged:
        raise FitQualityError("MultiSuSiE fit did not converge")
    passing_component_indices = _passing_component_indices(raw)
    if not passing_component_indices:
        raise FitQualityError("MultiSuSiE fit has no passing credible sets")
    return MultiSuSiEFit(
        raw=raw,
        converged=converged,
        passing_component_indices=passing_component_indices,
    )


def _passing_component_indices(raw: Any) -> list[int]:
    """Extract passing credible-set component indices from a library result."""
    try:
        passing = np.asarray(raw.sets[3], dtype=bool)
    except (AttributeError, IndexError, TypeError) as error:
        raise FitQualityError(
            "MultiSuSiE result has no credible-set pass mask"
        ) from error
    return [index for index, is_passing in enumerate(passing.tolist()) if is_passing]
