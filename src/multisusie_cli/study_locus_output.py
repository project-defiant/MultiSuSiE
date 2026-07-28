"""Convert reportable MultiSuSiE fits to Gentropy StudyLocus rows."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict
from scipy.stats import norm

from .models import RunParameters
from .preparation import PreparedLocus
from .runner import MultiSuSiEFit


class StudyLocusVariant(BaseModel):
    """Nested variant contract for one MultiSuSiE credible set."""

    model_config = ConfigDict(extra="forbid")

    variantId: str
    posteriorProbability: float
    is95CredibleSet: bool
    is99CredibleSet: None = None
    pValueMantissa: float
    pValueExponent: int
    beta: float
    standardError: float
    logBF: None = None
    r2Overall: None = None


class StudyLocusRecord(BaseModel):
    """Flat Gentropy-compatible parent row for one passing component."""

    model_config = ConfigDict(extra="forbid")

    studyLocusId: str
    studyType: None = None
    variantId: str
    chromosome: str
    position: int
    region: None = None
    studyId: str
    beta: float
    zScore: float
    pValueMantissa: float
    pValueExponent: int
    effectAlleleFrequencyFromSource: None = None
    standardError: float
    subStudyDescription: None = None
    qualityControls: list[str]
    finemappingMethod: str
    credibleSetIndex: int
    credibleSetlog10BF: float
    purityMeanR2: None = None
    purityMinR2: float | None
    locusStart: int
    locusEnd: int
    sampleSize: None = None
    ldSet: None = None
    locus: list[StudyLocusVariant]
    confidence: None = None
    isTransQtl: None = None


def write_study_locus(
    fit: MultiSuSiEFit,
    prepared: PreparedLocus,
    parameters: RunParameters,
    output: Path,
) -> None:
    """Write one flat Parquet row per passing joint credible-set component."""
    records = [
        _component_record(fit, prepared, parameters, component_index)
        for component_index in fit.passing_component_indices
    ]
    if not records:
        raise ValueError("Cannot write StudyLocus output without passing components")
    output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([record.model_dump() for record in records]).write_parquet(output)


def _component_record(
    fit: MultiSuSiEFit,
    prepared: PreparedLocus,
    parameters: RunParameters,
    component_index: int,
) -> StudyLocusRecord:
    raw = fit.raw
    component_variant_indices = [
        int(index) for index in np.asarray(raw.sets[0][component_index]).tolist()
    ]
    if not component_variant_indices:
        raise ValueError(f"Passing component has no variants: {component_index}")
    lead_index = _lead_index(raw, prepared, component_variant_indices, component_index)
    fixed_effect = _fixed_effect_statistics(prepared, component_variant_indices)
    lead_stats = fixed_effect[lead_index]
    nested = [
        StudyLocusVariant(
            variantId=prepared.variant_ids[index],
            posteriorProbability=float(raw.alpha[component_index, index]),
            is95CredibleSet=True,
            pValueMantissa=stats["pValueMantissa"],
            pValueExponent=stats["pValueExponent"],
            beta=stats["beta"],
            standardError=stats["standardError"],
        )
        for index, stats in fixed_effect.items()
        if index in component_variant_indices
    ]
    purity = float(raw.sets[1][component_index])
    purity_min_r2 = purity * purity if math.isfinite(purity) else None
    study_ids = sorted(population.study_id for population in prepared.populations)
    study_locus_id = hashlib.md5(
        f"{prepared.fine_mapping_locus_set_id}|{prepared.variant_ids[lead_index]}|MultiSuSiE".encode()
    ).hexdigest()
    return StudyLocusRecord(
        studyLocusId=study_locus_id,
        variantId=prepared.variant_ids[lead_index],
        chromosome=prepared.chromosomes[lead_index],
        position=prepared.positions[lead_index],
        studyId="|".join(study_ids),
        beta=lead_stats["beta"],
        zScore=lead_stats["zScore"],
        pValueMantissa=lead_stats["pValueMantissa"],
        pValueExponent=lead_stats["pValueExponent"],
        standardError=lead_stats["standardError"],
        qualityControls=[],
        finemappingMethod="MultiSuSiE",
        credibleSetIndex=component_index,
        credibleSetlog10BF=float(raw.lbf[component_index]) / math.log(10),
        purityMinR2=purity_min_r2,
        locusStart=min(prepared.positions),
        locusEnd=max(prepared.positions),
        locus=nested,
    )


def _lead_index(
    raw: Any,
    prepared: PreparedLocus,
    indices: list[int],
    component_index: int,
) -> int:
    component_alpha = np.asarray(raw.alpha)[component_index]
    return min(
        indices,
        key=lambda index: (
            -float(component_alpha[index]),
            _p_value(_fixed_effect_statistics(prepared, [index])[index]),
            prepared.variant_ids[index],
        ),
    )


def _fixed_effect_statistics(
    prepared: PreparedLocus, indices: list[int]
) -> dict[int, dict[str, float | int]]:
    statistics: dict[int, dict[str, float | int]] = {}
    for index in indices:
        contributions = [
            (float(population.betas[index]), float(population.standard_errors[index]))
            for population in prepared.populations
            if population.variant_present[index]
            and population.betas is not None
            and population.standard_errors is not None
        ]
        if not contributions:
            raise ValueError(f"No fixed-effect statistics for variant index: {index}")
        weights = np.asarray([1 / (se * se) for _, se in contributions])
        beta = float(
            np.sum(weights * np.asarray([beta for beta, _ in contributions]))
            / np.sum(weights)
        )
        standard_error = float(np.sqrt(1 / np.sum(weights)))
        z_score = beta / standard_error
        mantissa, exponent = _scientific_p_value(z_score)
        statistics[index] = {
            "beta": beta,
            "standardError": standard_error,
            "zScore": z_score,
            "pValueMantissa": mantissa,
            "pValueExponent": exponent,
        }
    return statistics


def _scientific_p_value(z_score: float) -> tuple[float, int]:
    log10_p = (math.log(2) + norm.logsf(abs(z_score))) / math.log(10)
    exponent = math.floor(log10_p)
    return 10 ** (log10_p - exponent), exponent


def _p_value(statistics: dict[str, float | int]) -> float:
    """Recover a p-value from its Gentropy mantissa/exponent representation."""
    return float(statistics["pValueMantissa"]) * 10 ** int(statistics["pValueExponent"])
