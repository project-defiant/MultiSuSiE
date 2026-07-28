"""Typer command-line interface for the MultiSuSiE application layer."""

from pathlib import Path
from uuid import uuid4

import typer
from loguru import logger

from .anndata_output import write_anndata
from .models import RunInputs, RunParameters
from .preparation import PreparedLocus, prepare_inputs
from .runner import FitQualityError, MultiSuSiEFit, run_multisusie
from .study_locus_output import write_study_locus

app = typer.Typer(no_args_is_help=True)


@app.command()
def run(
    fine_mapping_locus_set: Path = typer.Option(..., exists=True),
    multi_ancestry_pairwise_ld: Path = typer.Option(..., exists=True),
    study_metadata: Path = typer.Option(..., exists=True),
    run_id: str = typer.Option(...),
    fine_mapping_locus_set_id: str = typer.Option(...),
    study_locus_output: Path = typer.Option(...),
    extended_results_output: Path = typer.Option(...),
    rho: float = typer.Option(0.75, min=0, max=0.999999),
    L: int = typer.Option(10, min=1),
    scaled_prior_variance: float = typer.Option(0.2, min=0.0000001),
    pop_spec_standardization: bool = typer.Option(True),
    estimate_residual_variance: bool = typer.Option(True),
    estimate_prior_variance: bool = typer.Option(True),
    estimate_prior_method: str = typer.Option("early_EM"),
    pop_spec_effect_priors: bool = typer.Option(True),
    iter_before_zeroing_effects: int = typer.Option(5, min=0),
    prior_tol: float = typer.Option(1e-9, min=0.000000000001),
    max_iter: int = typer.Option(100, min=1),
    tol: float = typer.Option(1e-3, min=0.000000000001),
    coverage: float = typer.Option(0.95, min=0.000001, max=1),
    min_abs_corr: float = typer.Option(0, min=0, max=1),
    low_memory_mode: bool = typer.Option(False),
) -> None:
    """Run MultiSuSiE for one fine-mapping locus set."""
    inputs = RunInputs(
        fine_mapping_locus_set=fine_mapping_locus_set,
        multi_ancestry_pairwise_ld=multi_ancestry_pairwise_ld,
        study_metadata=study_metadata,
        run_id=run_id,
        fine_mapping_locus_set_id=fine_mapping_locus_set_id,
    )
    parameters = RunParameters(
        rho=rho,
        L=L,
        scaled_prior_variance=scaled_prior_variance,
        pop_spec_standardization=pop_spec_standardization,
        estimate_residual_variance=estimate_residual_variance,
        estimate_prior_variance=estimate_prior_variance,
        estimate_prior_method=estimate_prior_method,
        pop_spec_effect_priors=pop_spec_effect_priors,
        iter_before_zeroing_effects=iter_before_zeroing_effects,
        prior_tol=prior_tol,
        max_iter=max_iter,
        tol=tol,
        coverage=coverage,
        min_abs_corr=min_abs_corr,
        low_memory_mode=low_memory_mode,
    )
    logger.info(
        "Validated MultiSuSiE inputs for run_id={} locus_set_id={} with {}",
        inputs.run_id,
        inputs.fine_mapping_locus_set_id,
        parameters.model_dump_json(),
    )
    try:
        prepared = prepare_inputs(inputs)
        fit = run_multisusie(prepared, parameters)
        _write_outputs_atomically(
            fit=fit,
            prepared=prepared,
            parameters=parameters,
            study_locus_output=study_locus_output,
            extended_results_output=extended_results_output,
        )
    except (FitQualityError, OSError, ValueError) as error:
        logger.error("MultiSuSiE run failed: {}", error)
        raise typer.Exit(code=1) from error
    logger.info(
        "MultiSuSiE completed for run_id={} locus_set_id={} with {} reportable components",
        inputs.run_id,
        inputs.fine_mapping_locus_set_id,
        len(fit.passing_component_indices),
    )


def _write_outputs_atomically(
    *,
    fit: MultiSuSiEFit,
    prepared: PreparedLocus,
    parameters: RunParameters,
    study_locus_output: Path,
    extended_results_output: Path,
) -> None:
    """Write both outputs before replacing any user-visible result."""
    temporary_study_locus = _temporary_path(study_locus_output)
    temporary_extended_results = _temporary_path(extended_results_output)
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        write_study_locus(fit, prepared, parameters, temporary_study_locus)
        write_anndata(fit, prepared, parameters, temporary_extended_results)
        study_locus_output.parent.mkdir(parents=True, exist_ok=True)
        extended_results_output.parent.mkdir(parents=True, exist_ok=True)
        for output in (study_locus_output, extended_results_output):
            if output.exists():
                backup = _temporary_path(output)
                _replace(output, backup)
                backups.append((output, backup))
        _replace(temporary_study_locus, study_locus_output)
        published.append(study_locus_output)
        _replace(temporary_extended_results, extended_results_output)
        published.append(extended_results_output)
    except BaseException:
        for output in published:
            output.unlink(missing_ok=True)
        for output, backup in reversed(backups):
            backup.replace(output)
        raise
    finally:
        temporary_study_locus.unlink(missing_ok=True)
        temporary_extended_results.unlink(missing_ok=True)
        for _, backup in backups:
            backup.unlink(missing_ok=True)


def _temporary_path(output: Path) -> Path:
    """Return a unique sibling path so writers retain their expected suffix."""
    return output.with_name(f".{output.name}.{uuid4().hex}.tmp{output.suffix}")


def _replace(source: Path, target: Path) -> None:
    """Keep filesystem publication injectable for failure-path tests."""
    source.replace(target)
