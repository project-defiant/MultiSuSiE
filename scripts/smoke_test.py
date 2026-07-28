"""Run the public CLI against a tiny synthetic locus set."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from multisusie_cli.cli import app


def main() -> None:
    """Create inputs, run the CLI, and validate both result artifacts."""
    with tempfile.TemporaryDirectory(prefix="multisusie-smoke-") as directory:
        root = Path(directory)
        locus = root / "locus.parquet"
        pl.DataFrame(
            {
                "fineMappingLocusSetId": ["smoke-set"],
                "studyId": ["smoke-study"],
                "locus": [
                    [
                        {"variantId": "1_10_A_G", "beta": 3.0, "standardError": 1.0},
                        {"variantId": "1_20_C_T", "beta": 0.1, "standardError": 1.0},
                        {"variantId": "1_30_G_A", "beta": 0.2, "standardError": 1.0},
                    ]
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
        ld = root / "ld.parquet"
        pl.DataFrame(
            {
                "ancestry": ["EUR"] * 5,
                "variantIdI": [
                    "1_10_A_G",
                    "1_10_A_G",
                    "1_20_C_T",
                    "1_20_C_T",
                    "1_30_G_A",
                ],
                "variantIdJ": [
                    "1_10_A_G",
                    "1_20_C_T",
                    "1_20_C_T",
                    "1_30_G_A",
                    "1_30_G_A",
                ],
                "r": [1.0, 0.0, 1.0, 0.0, 1.0],
            }
        ).write_parquet(ld)
        metadata = root / "metadata.jsonl"
        metadata.write_text(
            json.dumps(
                {"studyId": "smoke-study", "ancestry": "EUR", "sampleSize": 1000}
            )
            + "\n"
        )
        study_locus = root / "study_locus.parquet"
        extended = root / "fit.h5ad"
        result = CliRunner().invoke(
            app,
            [
                "--fine-mapping-locus-set",
                str(locus),
                "--multi-ancestry-pairwise-ld",
                str(ld),
                "--study-metadata",
                str(metadata),
                "--run-id",
                "smoke-run",
                "--fine-mapping-locus-set-id",
                "smoke-set",
                "--study-locus-output",
                str(study_locus),
                "--extended-results-output",
                str(extended),
                "--L",
                "2",
                "--max-iter",
                "30",
            ],
        )
        if result.exit_code != 0:
            raise RuntimeError(result.output) from result.exception
        if not study_locus.is_file() or not extended.is_file():
            raise RuntimeError("Smoke test did not produce both output artifacts")


if __name__ == "__main__":
    main()
