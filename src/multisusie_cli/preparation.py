"""Prepare Gentropy parquet inputs for the MultiSuSiE numerical API."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from .models import RunInputs, StudyMetadata


class PopulationArrays(BaseModel):
    """Aligned arrays for one ancestry in the MultiSuSiE input contract."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    ancestry: str
    study_id: str
    sample_size: int = Field(gt=0)
    variant_present: np.ndarray
    z_scores: np.ndarray
    ld_matrix: np.ndarray
    betas: np.ndarray | None = None
    standard_errors: np.ndarray | None = None


class PreparedLocus(BaseModel):
    """Deterministically ordered numerical inputs for one locus set."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    run_id: str
    fine_mapping_locus_set_id: str
    variant_ids: list[str]
    chromosomes: list[str]
    positions: list[int]
    populations: list[PopulationArrays]


def _parquet_input(path: Path) -> Path | str:
    """Return a file path or parquet-only glob for a dataset directory."""
    return str(path / "**" / "*.parquet") if path.is_dir() else path


def prepare_inputs(inputs: RunInputs) -> PreparedLocus:
    """Read, validate, and align locus, metadata, and pairwise-LD inputs."""
    locus = pl.read_parquet(_parquet_input(inputs.fine_mapping_locus_set))
    metadata = _read_metadata(inputs.study_metadata)
    _validate_locus_set(locus, inputs.fine_mapping_locus_set_id, metadata)

    variant_rows = _variant_rows(locus)
    ordered_variants = _ordered_variants(variant_rows)
    variant_ids = [row["variant_id"] for row in ordered_variants]
    variant_index = {variant_id: index for index, variant_id in enumerate(variant_ids)}
    chromosomes = [row["chromosome"] for row in ordered_variants]
    positions = [row["position"] for row in ordered_variants]
    ld = pl.read_parquet(_parquet_input(inputs.multi_ancestry_pairwise_ld))
    _validate_ld_columns(ld)

    populations = [
        _prepare_population(
            metadata_row,
            variant_rows,
            variant_ids,
            variant_index,
            ld.filter(pl.col("ancestry") == metadata_row.ancestry),
        )
        for metadata_row in metadata
    ]
    return PreparedLocus(
        run_id=inputs.run_id,
        fine_mapping_locus_set_id=inputs.fine_mapping_locus_set_id,
        variant_ids=variant_ids,
        chromosomes=chromosomes,
        positions=positions,
        populations=populations,
    )


def _read_metadata(path: Path) -> list[StudyMetadata]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    metadata = [StudyMetadata.model_validate(record) for record in records]
    if not metadata:
        raise ValueError("Study metadata must contain at least one row")
    if len({row.studyId for row in metadata}) != len(metadata):
        raise ValueError("Study metadata contains duplicate studyId values")
    if len({row.ancestry for row in metadata}) != len(metadata):
        raise ValueError("Study metadata must contain one study per ancestry")
    return sorted(metadata, key=lambda row: row.studyId)


def _validate_locus_set(
    locus: pl.DataFrame, locus_set_id: str, metadata: list[StudyMetadata]
) -> None:
    required = {"fineMappingLocusSetId", "studyId", "locus"}
    missing = required - set(locus.columns)
    if missing:
        raise ValueError(f"FineMappingLocusSet is missing columns: {sorted(missing)}")
    observed_ids = set(locus.get_column("fineMappingLocusSetId").drop_nulls())
    if observed_ids != {locus_set_id}:
        raise ValueError(
            "FineMappingLocusSet contains unexpected fineMappingLocusSetId values: "
            f"{sorted(observed_ids)}"
        )
    observed_studies = set(locus.get_column("studyId").drop_nulls())
    metadata_studies = {row.studyId for row in metadata}
    if observed_studies != metadata_studies:
        raise ValueError(
            "Study metadata and FineMappingLocusSet studyId values differ: "
            f"metadata={sorted(metadata_studies)}, loci={sorted(observed_studies)}"
        )


def _variant_rows(locus: pl.DataFrame) -> dict[str, dict[str, dict[str, float | str]]]:
    rows: dict[str, dict[str, dict[str, float | str]]] = {}
    for row in locus.select("studyId", "locus").iter_rows(named=True):
        study_id = str(row["studyId"])
        if row["locus"] is None:
            raise ValueError(f"Study locus has no variants: {study_id}")
        study_rows = rows.setdefault(study_id, {})
        for variant in row["locus"]:
            variant_id = str(variant["variantId"])
            if variant_id in study_rows:
                raise ValueError(f"Duplicate variant in study locus: {variant_id}")
            beta = variant.get("beta")
            standard_error = variant.get("standardError")
            if beta is None or standard_error is None:
                raise ValueError(
                    f"Missing beta or standardError for {study_id}/{variant_id}"
                )
            if not math.isfinite(float(beta)) or not math.isfinite(
                float(standard_error)
            ):
                raise ValueError(
                    f"Non-finite beta or standardError for {study_id}/{variant_id}"
                )
            if float(standard_error) <= 0:
                raise ValueError(
                    f"Non-positive standardError for {study_id}/{variant_id}"
                )
            chromosome, position = _parse_variant_id(variant_id)
            study_rows[variant_id] = {
                "variant_id": variant_id,
                "chromosome": chromosome,
                "position": position,
                "beta": float(beta),
                "standard_error": float(standard_error),
            }
    return rows


def _ordered_variants(
    variant_rows: dict[str, dict[str, dict[str, float | str]]],
) -> list[dict[str, Any]]:
    variants = {
        variant_id: variant
        for study_rows in variant_rows.values()
        for variant_id, variant in study_rows.items()
    }
    ordered = sorted(
        variants.values(),
        key=lambda row: (
            str(row["chromosome"]),
            int(row["position"]),
            str(row["variant_id"]),
        ),
    )
    chromosomes = {str(row["chromosome"]) for row in ordered}
    if len(chromosomes) != 1:
        raise ValueError(
            f"Fine-mapping locus set spans chromosomes: {sorted(chromosomes)}"
        )
    return ordered


def _prepare_population(
    metadata: StudyMetadata,
    variant_rows: dict[str, dict[str, dict[str, float | str]]],
    variant_ids: list[str],
    variant_index: dict[str, int],
    ld: pl.DataFrame,
) -> PopulationArrays:
    study_variants = variant_rows[metadata.studyId]
    present = np.zeros(len(variant_ids), dtype=bool)
    z_scores = np.full(len(variant_ids), np.nan, dtype=np.float32)
    for variant_id, variant in study_variants.items():
        index = variant_index[variant_id]
        present[index] = True
        z_scores[index] = float(variant["beta"]) / float(variant["standard_error"])
    ld_matrix = _build_ld_matrix(ld, variant_ids, variant_index, present)
    if not present.any():
        raise ValueError(f"No usable variants for ancestry: {metadata.ancestry}")
    return PopulationArrays(
        ancestry=metadata.ancestry,
        study_id=metadata.studyId,
        sample_size=metadata.sampleSize,
        variant_present=present,
        z_scores=z_scores,
        ld_matrix=ld_matrix,
        betas=np.asarray(
            [
                float(study_variants[variant_id]["beta"])
                if variant_id in study_variants
                else np.nan
                for variant_id in variant_ids
            ],
            dtype=np.float32,
        ),
        standard_errors=np.asarray(
            [
                float(study_variants[variant_id]["standard_error"])
                if variant_id in study_variants
                else np.nan
                for variant_id in variant_ids
            ],
            dtype=np.float32,
        ),
    )


def _build_ld_matrix(
    ld: pl.DataFrame,
    variant_ids: list[str],
    variant_index: dict[str, int],
    present: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((len(variant_ids), len(variant_ids)), dtype=np.float32)
    np.fill_diagonal(matrix, 1.0)
    seen: dict[tuple[str, str], float] = {}
    for row in ld.iter_rows(named=True):
        first = str(row["variantIdI"])
        second = str(row["variantIdJ"])
        if first not in variant_index or second not in variant_index:
            continue
        value = float(row["r"])
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError(f"Invalid LD value for {first}/{second}: {value}")
        key: tuple[str, str] = (first, second) if first <= second else (second, first)
        previous = seen.get(key)
        if previous is not None and not math.isclose(previous, value, rel_tol=1e-6):
            raise ValueError(f"Conflicting LD values for {first}/{second}")
        seen[key] = value
        first_index = variant_index[first]
        second_index = variant_index[second]
        matrix[first_index, second_index] = value
        matrix[second_index, first_index] = value
    absent = ~present
    matrix[absent, :] = 0
    matrix[:, absent] = 0
    return matrix


def _validate_ld_columns(ld: pl.DataFrame) -> None:
    required = {"ancestry", "variantIdI", "variantIdJ", "r"}
    missing = required - set(ld.columns)
    if missing:
        raise ValueError(
            f"MultiAncestryPairwiseLD is missing columns: {sorted(missing)}"
        )


def _parse_variant_id(variant_id: str) -> tuple[str, int]:
    fields = variant_id.split("_")
    if len(fields) < 2:
        raise ValueError(
            f"Cannot parse chromosome and position from variantId: {variant_id}"
        )
    try:
        return fields[0], int(fields[1])
    except ValueError as error:
        raise ValueError(
            f"Cannot parse position from variantId: {variant_id}"
        ) from error
