"""Tests for deterministic MultiSuSiE input preparation."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from multisusie_cli.models import RunInputs
from multisusie_cli.preparation import prepare_inputs


def _write_inputs(tmp_path: Path) -> RunInputs:
    locus = tmp_path / "locus.parquet"
    pl.DataFrame(
        {
            "fineMappingLocusSetId": ["set-1", "set-1"],
            "studyId": ["study-b", "study-a"],
            "locus": [
                [
                    {"variantId": "1_20_A_G", "beta": 2.0, "standardError": 1.0},
                    {"variantId": "1_10_C_T", "beta": 1.0, "standardError": 2.0},
                ],
                [{"variantId": "1_20_A_G", "beta": 3.0, "standardError": 1.0}],
            ],
        },
        schema={
            "fineMappingLocusSetId": pl.String,
            "studyId": pl.String,
            "locus": pl.List(
                pl.Struct(
                    {
                        "variantId": pl.String,
                        "beta": pl.Float64,
                        "standardError": pl.Float64,
                    }
                )
            ),
        },
    ).write_parquet(locus)
    ld = tmp_path / "ld.parquet"
    pl.DataFrame(
        {
            "ancestry": ["EUR", "EUR", "EUR", "AFR", "AFR", "AFR"],
            "variantIdI": [
                "1_10_C_T",
                "1_10_C_T",
                "1_20_A_G",
                "1_10_C_T",
                "1_10_C_T",
                "1_20_A_G",
            ],
            "variantIdJ": [
                "1_10_C_T",
                "1_20_A_G",
                "1_20_A_G",
                "1_10_C_T",
                "1_20_A_G",
                "1_20_A_G",
            ],
            "r": [1.0, 0.5, 1.0, 1.0, 0.5, 1.0],
        }
    ).write_parquet(ld)
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"studyId": "study-a", "ancestry": "EUR", "sampleSize": 1000},
                {"studyId": "study-b", "ancestry": "AFR", "sampleSize": 900},
            ]
        )
        + "\n"
    )
    return RunInputs(
        fine_mapping_locus_set=locus,
        multi_ancestry_pairwise_ld=ld,
        study_metadata=metadata,
        run_id="run-1",
        fine_mapping_locus_set_id="set-1",
    )


def test_prepare_inputs_orders_variants_and_masks_absent_ancestry(
    tmp_path: Path,
) -> None:
    prepared = prepare_inputs(_write_inputs(tmp_path))

    assert prepared.variant_ids == ["1_10_C_T", "1_20_A_G"]
    assert [population.ancestry for population in prepared.populations] == [
        "EUR",
        "AFR",
    ]
    np.testing.assert_allclose(
        prepared.populations[0].z_scores, [np.nan, 3.0], equal_nan=True
    )
    np.testing.assert_allclose(prepared.populations[1].z_scores, [0.5, 2.0])
    np.testing.assert_allclose(prepared.populations[0].ld_matrix, [[0, 0], [0, 1]])
    np.testing.assert_allclose(prepared.populations[1].ld_matrix, [[1, 0.5], [0.5, 1]])


def test_prepare_inputs_reads_spark_parquet_dataset_with_metadata_files(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    dataset = tmp_path / "ld_dataset"
    dataset.mkdir()
    inputs.multi_ancestry_pairwise_ld.rename(dataset / "part-00000.snappy.parquet")
    (dataset / "._SUCCESS.crc").touch()
    inputs = inputs.model_copy(update={"multi_ancestry_pairwise_ld": dataset})

    prepared = prepare_inputs(inputs)

    assert prepared.variant_ids == ["1_10_C_T", "1_20_A_G"]


def test_prepare_inputs_fails_on_missing_summary_statistics(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    locus = pl.read_parquet(inputs.fine_mapping_locus_set)
    locus = locus.with_columns(
        pl.when(pl.col("studyId") == "study-a")
        .then(pl.lit([{"variantId": "1_20_A_G", "beta": None, "standardError": 1.0}]))
        .otherwise(pl.col("locus"))
        .alias("locus")
    )
    locus.write_parquet(inputs.fine_mapping_locus_set)

    with pytest.raises(ValueError, match="Missing beta"):
        prepare_inputs(inputs)
