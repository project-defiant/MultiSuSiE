"""End-to-end tests for the MultiSuSiE command-line application."""

from pathlib import Path

from anndata import read_h5ad
from typer.testing import CliRunner

from multisusie_cli import cli as cli_module
from multisusie_cli.models import RunInputs
from test_preparation import _write_inputs

app = cli_module.app
runner = CliRunner()


def _arguments(inputs: RunInputs, tmp_path: Path) -> list[str]:
    return [
        "--fine-mapping-locus-set",
        str(inputs.fine_mapping_locus_set),
        "--multi-ancestry-pairwise-ld",
        str(inputs.multi_ancestry_pairwise_ld),
        "--study-metadata",
        str(inputs.study_metadata),
        "--run-id",
        inputs.run_id,
        "--fine-mapping-locus-set-id",
        inputs.fine_mapping_locus_set_id,
        "--study-locus-output",
        str(tmp_path / "results" / "study_locus.parquet"),
        "--extended-results-output",
        str(tmp_path / "results" / "fit.h5ad"),
        "--L",
        "2",
        "--max-iter",
        "30",
    ]


def test_cli_runs_fit_and_writes_both_outputs(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)

    result = runner.invoke(app, _arguments(inputs, tmp_path))

    assert result.exit_code == 0, result.output
    assert (tmp_path / "results" / "study_locus.parquet").is_file()
    extended = read_h5ad(tmp_path / "results" / "fit.h5ad")
    assert extended.uns["runId"] == "run-1"
    assert extended.n_obs == 2


def test_cli_does_not_publish_outputs_when_fit_is_not_reportable(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    study_locus = output_dir / "study_locus.parquet"
    extended = output_dir / "fit.h5ad"

    arguments = _arguments(inputs, tmp_path)
    arguments[arguments.index("30")] = "1"
    result = runner.invoke(app, arguments)

    assert result.exit_code != 0
    assert not study_locus.exists()
    assert not extended.exists()
    assert not list(output_dir.glob(".*.tmp*"))


def test_cli_rolls_back_both_outputs_when_second_publish_fails(
    tmp_path: Path, monkeypatch
) -> None:
    inputs = _write_inputs(tmp_path)
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    study_locus = output_dir / "study_locus.parquet"
    extended = output_dir / "fit.h5ad"
    study_locus.write_bytes(b"previous study locus")
    extended.write_bytes(b"previous extended result")
    original_replace = cli_module._replace

    def fail_extended_publish(source: Path, target: Path) -> None:
        if target == extended:
            raise OSError("simulated second publication failure")
        original_replace(source, target)

    monkeypatch.setattr(cli_module, "_replace", fail_extended_publish)
    result = runner.invoke(app, _arguments(inputs, tmp_path))

    assert result.exit_code == 1
    assert study_locus.read_bytes() == b"previous study locus"
    assert extended.read_bytes() == b"previous extended result"
    assert not list(output_dir.glob(".*.tmp*"))
