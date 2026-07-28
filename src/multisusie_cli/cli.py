"""Typer command-line interface for the MultiSuSiE application layer."""

from pathlib import Path

import typer
from loguru import logger

from .models import RunInputs, RunParameters

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
    typer.echo("Numerical execution is not implemented yet", err=True)
    raise typer.Exit(code=2)
