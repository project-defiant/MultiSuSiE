import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


@pytest.fixture(autouse=True)
def fixed_test_seeds():
    random.seed(0)
    np.random.seed(0)


@dataclass
class SyntheticData:
    geno_list: list[np.ndarray]
    y_list: list[np.ndarray]
    beta_hat_list: list[np.ndarray]
    se_list: list[np.ndarray]
    r_list: list[np.ndarray]
    n_list: list[int]
    vary_list: list[float]
    common: dict


@pytest.fixture(scope="module")
def synthetic_data() -> SyntheticData:
    geno_yri = np.loadtxt("example_data/geno_YRI.txt")
    geno_ceu = np.loadtxt("example_data/geno_CEU.txt")
    geno_jpt = np.loadtxt("example_data/geno_JPT.txt")
    geno_list = [geno_ceu, geno_yri, geno_jpt]

    beta_yri = np.zeros(40)
    beta_ceu = np.zeros(40)
    beta_jpt = np.zeros(40)
    beta_yri[10] = 0.75
    beta_ceu[10] = 1
    beta_jpt[10] = 0.5
    beta_yri[3] = 0.5
    beta_ceu[3] = 0.5
    beta_jpt[3] = 0.5
    beta_yri[38] = 1
    beta_ceu[38] = 0
    beta_jpt[38] = 0
    beta_list = [beta_yri, beta_ceu, beta_jpt]

    rng = np.random.default_rng(1)
    y_list = [geno.dot(beta) + rng.standard_normal(geno.shape[0]) for geno, beta in zip(geno_list, beta_list)]
    y_list = [y - np.mean(y) for y in y_list]

    xty_list = [geno.T.dot(y) for geno, y in zip(geno_list, y_list)]
    xtx_diag_list = [np.diagonal(geno.T.dot(geno)) for geno in geno_list]
    with np.errstate(divide="ignore", invalid="ignore"):
        beta_hat_list = [xty / xtx_diag for xty, xtx_diag in zip(xty_list, xtx_diag_list)]

    n_list = [geno.shape[0] for geno in geno_list]
    residuals_list = [np.expand_dims(y, 1) - (geno * beta) for y, geno, beta in zip(y_list, geno_list, beta_hat_list)]
    ssr_list = [np.sum(resid ** 2, axis=0) for resid in residuals_list]
    se_list = [np.sqrt(ssr / ((n - 2) * xtx)) for ssr, n, xtx in zip(ssr_list, n_list, xtx_diag_list)]

    with np.errstate(divide="ignore", invalid="ignore"):
        r_list = [np.corrcoef(geno, rowvar=False) for geno in geno_list]

    vary_list = [np.var(y, ddof=1) for y in y_list]
    rho = np.array([[1, 0.75, 0.75], [0.75, 1, 0.75], [0.75, 0.75, 1]])
    common = dict(
        rho=rho,
        L=10,
        scaled_prior_variance=0.2,
        min_abs_corr=0,
        float_type=np.float64,
        estimate_prior_method="EM",
        pop_spec_effect_priors=False,
        iter_before_zeroing_effects=0,
    )

    return SyntheticData(
        geno_list=geno_list,
        y_list=y_list,
        beta_hat_list=beta_hat_list,
        se_list=se_list,
        r_list=r_list,
        n_list=n_list,
        vary_list=vary_list,
        common=common,
    )
