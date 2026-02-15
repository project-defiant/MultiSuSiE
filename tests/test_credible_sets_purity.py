from types import SimpleNamespace

import numpy as np
import pytest

import MultiSuSiE
import MultiSuSiE.susiepy_ss as susiepy_ss


def _expected_purity_for_cs(cs, r_list):
    if len(cs) == 0:
        return np.nan
    r_sub = [r[np.ix_(cs, cs)] for r in r_list]
    abs_meta_r = np.maximum.reduce([np.abs(r) for r in r_sub])
    return float(np.min(abs_meta_r))


def _make_reasonable_filtering_case(seed=4):
    rng = np.random.default_rng(seed)
    p = 6
    n1, n2 = 120, 110
    cov = np.array(
        [
            [1, 0.4, 0.2, 0, 0, 0],
            [0.4, 1, 0.3, 0, 0, 0],
            [0.2, 0.3, 1, 0, 0, 0],
            [0, 0, 0, 1, 0.5, 0.1],
            [0, 0, 0, 0.5, 1, 0.2],
            [0, 0, 0, 0.1, 0.2, 1],
        ],
        dtype=float,
    )
    chol = np.linalg.cholesky(cov)
    x1 = rng.normal(size=(n1, p)).dot(chol.T)
    x2 = rng.normal(size=(n2, p)).dot(chol.T)
    beta = np.array([0.15, 0.14, 0.0, 0.0, 0.0, 0.0], dtype=float)
    y1 = x1.dot(beta) + rng.normal(scale=1.2, size=n1)
    y2 = x2.dot(beta) + rng.normal(scale=1.2, size=n2)
    y1 -= y1.mean()
    y2 -= y2.mean()
    x_list = [x1, x2]
    y_list = [y1, y2]
    rho = np.array([[1, 0.8], [0.8, 1]], dtype=float)
    common = dict(
        rho=rho,
        L=4,
        scaled_prior_variance=0.2,
        estimate_prior_method="EM",
        pop_spec_effect_priors=False,
        iter_before_zeroing_effects=0,
        float_type=np.float64,
    )

    xty = [x.T.dot(y) for x, y in zip(x_list, y_list)]
    xtx_diag = [np.diag(x.T.dot(x)) for x in x_list]
    with np.errstate(divide="ignore", invalid="ignore"):
        b_list = [xy / xd for xy, xd in zip(xty, xtx_diag)]
    n_list = [n1, n2]
    residuals = [np.expand_dims(y, 1) - (x * b) for x, y, b in zip(x_list, y_list, b_list)]
    ssr = [np.sum(r ** 2, axis=0) for r in residuals]
    s_list = [np.sqrt(s / ((n - 2) * xd)) for s, n, xd in zip(ssr, n_list, xtx_diag)]
    r_list = [np.corrcoef(x, rowvar=False) for x in x_list]
    vary_list = [np.var(y, ddof=1) for y in y_list]
    return x_list, y_list, b_list, s_list, r_list, vary_list, n_list, common


def test_rss_low_memory_without_recover_r_skips_purity(synthetic_data):
    kwargs = dict(synthetic_data.common)
    kwargs["min_abs_corr"] = 0
    fit = MultiSuSiE.multisusie_rss(
        b_list=[b.copy() for b in synthetic_data.beta_hat_list],
        s_list=[s.copy() for s in synthetic_data.se_list],
        R_list=[r.copy() for r in synthetic_data.r_list],
        varY_list=synthetic_data.vary_list,
        population_sizes=synthetic_data.n_list,
        low_memory_mode=True,
        recover_R=False,
        single_population_mac_thresh=0,
        **kwargs,
    )
    purity = np.asarray(fit.sets[1], dtype=float)
    assert np.all(np.isnan(purity))


def test_rss_low_memory_with_recover_r_computes_purity(synthetic_data):
    kwargs = dict(synthetic_data.common)
    kwargs["min_abs_corr"] = 0
    fit = MultiSuSiE.multisusie_rss(
        b_list=[b.copy() for b in synthetic_data.beta_hat_list],
        s_list=[s.copy() for s in synthetic_data.se_list],
        R_list=[r.copy() for r in synthetic_data.r_list],
        varY_list=synthetic_data.vary_list,
        population_sizes=synthetic_data.n_list,
        low_memory_mode=True,
        recover_R=True,
        single_population_mac_thresh=0,
        **kwargs,
    )
    purity = np.asarray(fit.sets[1], dtype=float)
    assert np.any(np.isfinite(purity))


def test_individual_calculate_purity_flag_controls_purity_output(synthetic_data):
    fit_no_purity = MultiSuSiE.multisusie(
        X_list=[x.copy() for x in synthetic_data.geno_list],
        Y_list=[y.copy() for y in synthetic_data.y_list],
        standardize=False,
        calculate_purity=False,
        **synthetic_data.common,
    )
    purity_no = np.asarray(fit_no_purity.sets[1], dtype=float)
    assert np.all(np.isnan(purity_no))

    fit_purity = MultiSuSiE.multisusie(
        X_list=[x.copy() for x in synthetic_data.geno_list],
        Y_list=[y.copy() for y in synthetic_data.y_list],
        standardize=False,
        calculate_purity=True,
        **synthetic_data.common,
    )
    purity_yes = np.asarray(fit_purity.sets[1], dtype=float)
    assert np.any(np.isfinite(purity_yes))


def test_get_purity_x_respects_n_purity():
    cs = np.array([0, 1, 2], dtype=int)
    r = np.eye(3)
    purity_all = susiepy_ss.get_purity_x(cs=cs, R_list=[r], min_abs_cor=0, n_purity=3, X_list=None)
    purity_subsampled = susiepy_ss.get_purity_x(cs=cs, R_list=[r], min_abs_cor=0, n_purity=1, X_list=None)
    assert np.isclose(purity_all, 0.0)
    assert np.isclose(purity_subsampled, 1.0)


def test_susie_get_cs_filters_by_min_abs_corr_only_when_calculating_purity():
    s = SimpleNamespace(
        alpha=np.array([[0.6, 0.4, 0.0]], dtype=float),
        V=np.array([[0.1]], dtype=float),
    )
    r = np.array(
        [
            [1.0, 0.1, 0.0],
            [0.1, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    cs_with_purity = susiepy_ss.susie_get_cs(
        s=s,
        R_list=[r],
        coverage=0.95,
        min_abs_corr=0.5,
        dedup=True,
        n_purity=100,
        calculate_purity=True,
        X_list=None,
    )
    assert bool(cs_with_purity[3][0]) is False

    cs_without_purity = susiepy_ss.susie_get_cs(
        s=s,
        R_list=[r],
        coverage=0.95,
        min_abs_corr=0.5,
        dedup=True,
        n_purity=100,
        calculate_purity=False,
        X_list=None,
    )
    assert bool(cs_without_purity[3][0]) is True


@pytest.mark.parametrize("method", ["rss", "individual"])
def test_end_to_end_purity_values_match_direct_recomputation(synthetic_data, method):
    kwargs = dict(synthetic_data.common)
    kwargs["min_abs_corr"] = 0
    if method == "rss":
        fit = MultiSuSiE.multisusie_rss(
            b_list=[b.copy() for b in synthetic_data.beta_hat_list],
            s_list=[s.copy() for s in synthetic_data.se_list],
            R_list=[r.copy() for r in synthetic_data.r_list],
            varY_list=synthetic_data.vary_list,
            population_sizes=synthetic_data.n_list,
            single_population_mac_thresh=0,
            low_memory_mode=False,
            recover_R=False,
            **kwargs,
        )
    else:
        fit = MultiSuSiE.multisusie(
            X_list=[x.copy() for x in synthetic_data.geno_list],
            Y_list=[y.copy() for y in synthetic_data.y_list],
            standardize=False,
            calculate_purity=True,
            n_purity=10_000,
            **kwargs,
        )

    cs_list, purity_list, _, include_mask = fit.sets
    for cs, purity, include in zip(cs_list, purity_list, include_mask):
        if include:
            expected = _expected_purity_for_cs(cs, synthetic_data.r_list)
            if np.isnan(expected):
                assert np.isnan(purity)
            else:
                assert np.isclose(float(purity), expected, atol=1e-10)


@pytest.mark.parametrize("method", ["rss", "individual"])
def test_end_to_end_purity_filtering_matches_threshold(synthetic_data, method):
    # Use a threshold above the maximum attainable purity (1.0) so the test
    # deterministically exercises purity-based filtering.
    threshold = 1.01
    kwargs = dict(synthetic_data.common)
    kwargs["min_abs_corr"] = threshold
    if method == "rss":
        fit = MultiSuSiE.multisusie_rss(
            b_list=[b.copy() for b in synthetic_data.beta_hat_list],
            s_list=[s.copy() for s in synthetic_data.se_list],
            R_list=[r.copy() for r in synthetic_data.r_list],
            varY_list=synthetic_data.vary_list,
            population_sizes=synthetic_data.n_list,
            single_population_mac_thresh=0,
            low_memory_mode=False,
            recover_R=False,
            **kwargs,
        )
    else:
        fit = MultiSuSiE.multisusie(
            X_list=[x.copy() for x in synthetic_data.geno_list],
            Y_list=[y.copy() for y in synthetic_data.y_list],
            standardize=False,
            calculate_purity=True,
            n_purity=10_000,
            **kwargs,
        )

    _, purity_list, _, include_mask = fit.sets
    purity_arr = np.asarray(purity_list, dtype=float)
    include_arr = np.asarray(include_mask, dtype=bool)
    finite = np.isfinite(purity_arr)
    assert np.any(finite)
    assert np.any((~include_arr) & finite)
    assert np.all(include_arr[finite] == (purity_arr[finite] >= threshold))


@pytest.mark.parametrize("method", ["rss", "individual"])
def test_end_to_end_purity_filtering_activates_for_reasonable_threshold(method):
    x_list, y_list, b_list, s_list, r_list, vary_list, n_list, common = _make_reasonable_filtering_case()

    if method == "rss":
        baseline = MultiSuSiE.multisusie_rss(
            b_list=[b.copy() for b in b_list],
            s_list=[s.copy() for s in s_list],
            R_list=[r.copy() for r in r_list],
            varY_list=vary_list,
            population_sizes=n_list,
            single_population_mac_thresh=0,
            low_memory_mode=False,
            recover_R=False,
            min_abs_corr=0,
            **common,
        )
    else:
        baseline = MultiSuSiE.multisusie(
            X_list=[x.copy() for x in x_list],
            Y_list=[y.copy() for y in y_list],
            standardize=False,
            calculate_purity=True,
            n_purity=10_000,
            min_abs_corr=0,
            **common,
        )

    baseline_purity = np.asarray(baseline.sets[1], dtype=float)
    finite = baseline_purity[np.isfinite(baseline_purity)]
    assert finite.size > 0
    min_finite = float(np.min(finite))
    assert 0 < min_finite < 1

    threshold = min_finite + 0.05
    assert 0 < threshold < 1

    if method == "rss":
        fit = MultiSuSiE.multisusie_rss(
            b_list=[b.copy() for b in b_list],
            s_list=[s.copy() for s in s_list],
            R_list=[r.copy() for r in r_list],
            varY_list=vary_list,
            population_sizes=n_list,
            single_population_mac_thresh=0,
            low_memory_mode=False,
            recover_R=False,
            min_abs_corr=threshold,
            **common,
        )
    else:
        fit = MultiSuSiE.multisusie(
            X_list=[x.copy() for x in x_list],
            Y_list=[y.copy() for y in y_list],
            standardize=False,
            calculate_purity=True,
            n_purity=10_000,
            min_abs_corr=threshold,
            **common,
        )

    purity = np.asarray(fit.sets[1], dtype=float)
    include = np.asarray(fit.sets[3], dtype=bool)
    finite = np.isfinite(purity)
    assert np.any((~include) & finite)
    assert np.all(include[finite] == (purity[finite] >= threshold))
