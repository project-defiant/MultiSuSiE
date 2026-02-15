import numpy as np
import pytest

import MultiSuSiE


def _make_two_pop_data(seed=7):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=(40, 6))
    x2 = rng.normal(size=(35, 6))
    x1[:, 0] = 2.0
    x2[:, 0] = 2.0
    beta = np.array([0.0, 0.8, 0.0, -0.5, 0.0, 0.0])
    y1 = x1.dot(beta) + rng.normal(scale=0.5, size=x1.shape[0])
    y2 = x2.dot(beta) + rng.normal(scale=0.5, size=x2.shape[0])
    return [x1, x2], [y1, y2]


def test_missing_y_rows_are_removed_consistently():
    x_list, y_list = _make_two_pop_data()
    rho = np.eye(2)
    kwargs = dict(
        rho=rho,
        L=4,
        scaled_prior_variance=0.2,
        standardize=True,
        float_type=np.float64,
        estimate_prior_method="EM",
        pop_spec_effect_priors=False,
        iter_before_zeroing_effects=0,
        min_abs_corr=0,
        max_iter=60,
    )

    y_with_nan = [y.copy() for y in y_list]
    y_with_nan[0][5] = np.nan
    mask = ~np.isnan(y_with_nan[0])
    fit_nan = MultiSuSiE.multisusie(
        X_list=[x.copy() for x in x_list],
        Y_list=y_with_nan,
        **kwargs,
    )

    fit_filtered = MultiSuSiE.multisusie(
        X_list=[x_list[0][mask].copy(), x_list[1].copy()],
        Y_list=[y_list[0][mask].copy(), y_list[1].copy()],
        **kwargs,
    )

    np.testing.assert_allclose(fit_nan.pip, fit_filtered.pip, rtol=0, atol=1e-10)


def test_constant_column_has_zero_pip_when_standardized():
    x_list, y_list = _make_two_pop_data()
    fit = MultiSuSiE.multisusie(
        X_list=[x.copy() for x in x_list],
        Y_list=[y.copy() for y in y_list],
        rho=np.eye(2),
        L=4,
        standardize=True,
        min_abs_corr=0,
        float_type=np.float64,
        estimate_prior_method="EM",
        pop_spec_effect_priors=False,
        iter_before_zeroing_effects=0,
        max_iter=60,
    )
    assert np.isclose(fit.pip[0], 0.0)
    assert np.allclose(fit.coef[:, 0], 0.0)


def test_individual_input_shape_validation():
    x = np.random.default_rng(11).normal(size=(20, 4))
    y = np.random.default_rng(12).normal(size=20)

    with pytest.raises(AssertionError):
        MultiSuSiE.multisusie(
            X_list=[x, x],
            Y_list=[y],
            rho=np.eye(2),
            L=2,
            min_abs_corr=0,
        )

    with pytest.raises(AssertionError):
        MultiSuSiE.multisusie(
            X_list=[x],
            Y_list=[y[:-1]],
            rho=np.array([[1.0]]),
            L=2,
            min_abs_corr=0,
        )


def test_individual_multisusie_mutates_x_when_low_memory_true():
    x_list, y_list = _make_two_pop_data(seed=21)
    x_in = [x.copy() for x in x_list]
    y_in = [y.copy() for y in y_list]
    x_before = [x.copy() for x in x_in]

    MultiSuSiE.multisusie(
        X_list=x_in,
        Y_list=y_in,
        rho=np.eye(2),
        L=4,
        scaled_prior_variance=0.2,
        standardize=True,
        low_memory_mode=True,
        min_abs_corr=0,
        float_type=np.float64,
        estimate_prior_method="EM",
        pop_spec_effect_priors=False,
        iter_before_zeroing_effects=0,
        max_iter=60,
    )

    any_changed = any(np.max(np.abs(before - after)) > 1e-10 for before, after in zip(x_before, x_in))
    assert any_changed
