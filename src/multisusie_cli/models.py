"""Pydantic models for the MultiSuSiE application boundary."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Literal


class StudyMetadata(BaseModel):
    """Metadata required to construct one MultiSuSiE population input."""

    model_config = ConfigDict(extra="forbid")

    studyId: str = Field(min_length=1)
    ancestry: str = Field(min_length=1)
    sampleSize: int = Field(gt=0)


class RunParameters(BaseModel):
    """User-configurable MultiSuSiE parameters exposed by the CLI."""

    model_config = ConfigDict(extra="forbid")

    rho: float = Field(default=0.75, ge=0, lt=1)
    L: int = Field(default=10, gt=0)
    scaled_prior_variance: float = Field(default=0.2, gt=0)
    pop_spec_standardization: bool = True
    estimate_residual_variance: bool = True
    estimate_prior_variance: bool = True
    estimate_prior_method: str = "early_EM"
    pop_spec_effect_priors: bool = True
    iter_before_zeroing_effects: int = Field(default=5, ge=0)
    prior_tol: float = Field(default=1e-9, gt=0)
    max_iter: int = Field(default=100, gt=0)
    tol: float = Field(default=1e-3, gt=0)
    coverage: float = Field(default=0.95, gt=0, le=1)
    min_abs_corr: float = Field(default=0, ge=0, le=1)
    low_memory_mode: bool = False


class RunInputs(BaseModel):
    """Validated command inputs for one fine-mapping locus set."""

    model_config = ConfigDict(extra="forbid")

    fine_mapping_locus_set: Path
    multi_ancestry_pairwise_ld: Path
    study_metadata: Path
    run_id: str = Field(min_length=1)
    fine_mapping_locus_set_id: str = Field(min_length=1)

    @field_validator(
        "fine_mapping_locus_set", "multi_ancestry_pairwise_ld", "study_metadata"
    )
    @classmethod
    def input_must_exist(cls, value: Path) -> Path:
        """Reject missing input paths before any numerical work starts."""
        if not value.exists():
            raise ValueError(f"Input path does not exist: {value}")
        return value


class MultiSuSiEStats(BaseModel):
    """Machine-readable status emitted for every MultiSuSiE invocation."""

    model_config = ConfigDict(extra="forbid")

    runId: str
    fineMappingLocusSetId: str
    status: Literal["SUCCESS", "NON_CONVERGED", "FAILED"]
    converged: bool | None = None
    niter: int | None = None
    nReportableComponents: int | None = None
    reason: str | None = None
