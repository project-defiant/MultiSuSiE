import numpy as np
import pytest

from MultiSuSiE import susiepy
from MultiSuSiE import susiepy_ss


def _direct_log_bf_single_predictor(x, y, sigma2, prior_var):
    n = y.shape[0]
    eye = np.eye(n)

    sigma0 = sigma2 * eye
    sigma1 = sigma2 * eye + prior_var * np.outer(x, x)

    sign0, logdet0 = np.linalg.slogdet(sigma0)
    sign1, logdet1 = np.linalg.slogdet(sigma1)
    assert sign0 > 0 and sign1 > 0

    quad0 = y.dot(np.linalg.solve(sigma0, y))
    quad1 = y.dot(np.linalg.solve(sigma1, y))

    loglik0 = -0.5 * (n * np.log(2 * np.pi) + logdet0 + quad0)
    loglik1 = -0.5 * (n * np.log(2 * np.pi) + logdet1 + quad1)
    return loglik1 - loglik0


def _stacked_sigma0(residual_variance, n_list):
    total_n = int(np.sum(n_list))
    sigma0 = np.zeros((total_n, total_n), dtype=float)
    start = 0
    for s2, n in zip(residual_variance, n_list):
        sigma0[start : start + n, start : start + n] = s2 * np.eye(n)
        start += n
    return sigma0


def _design_for_variant(x_list, variant_idx):
    total_n = int(np.sum([x.shape[0] for x in x_list]))
    k = len(x_list)
    z = np.zeros((total_n, k), dtype=float)
    start = 0
    for pop_idx, x in enumerate(x_list):
        n = x.shape[0]
        z[start : start + n, pop_idx] = x[:, variant_idx]
        start += n
    return z


def _direct_multi_pop_log_bf_for_variant(
    x_list, y_list, residual_variance, rho, prior_variance, variant_idx
):
    y = np.concatenate(y_list)
    n_total = y.shape[0]

    sigma0 = _stacked_sigma0(residual_variance, [x.shape[0] for x in x_list])
    z = _design_for_variant(x_list, variant_idx)
    a = rho * np.sqrt(np.outer(prior_variance, prior_variance))
    sigma1 = sigma0 + z.dot(a).dot(z.T)

    sign0, logdet0 = np.linalg.slogdet(sigma0)
    sign1, logdet1 = np.linalg.slogdet(sigma1)
    assert sign0 > 0 and sign1 > 0

    quad0 = y.dot(np.linalg.solve(sigma0, y))
    quad1 = y.dot(np.linalg.solve(sigma1, y))

    loglik0 = -0.5 * (n_total * np.log(2 * np.pi) + logdet0 + quad0)
    loglik1 = -0.5 * (n_total * np.log(2 * np.pi) + logdet1 + quad1)
    return loglik1 - loglik0


def _direct_multi_pop_posterior_for_variant(
    x_list, y_list, residual_variance, rho, prior_variance, variant_idx
):
    y = np.concatenate(y_list)
    z = _design_for_variant(x_list, variant_idx)
    a = rho * np.sqrt(np.outer(prior_variance, prior_variance))
    sigma0 = _stacked_sigma0(residual_variance, [x.shape[0] for x in x_list])
    sigma_y = sigma0 + z.dot(a).dot(z.T)
    cov_by = a.dot(z.T)
    inv_sigma_y_y = np.linalg.solve(sigma_y, y)
    post_mean = cov_by.dot(inv_sigma_y_y)
    post_covar = a - cov_by.dot(np.linalg.solve(sigma_y, cov_by.T))
    post_second = post_covar + np.outer(post_mean, post_mean)
    return post_mean, post_second


@pytest.mark.parametrize("prior_var", [1e-6, 0.05, 0.2, 1.0])
def test_single_population_compute_lbf_matches_direct_bayesian_linear_regression(
    prior_var,
):
    rng = np.random.default_rng(2026)
    n = 30
    p = 7
    sigma2 = 1.3

    x = rng.normal(size=(n, p))
    x[:, 0] = 0.0
    y = rng.normal(size=n)

    xty = x.T.dot(y)
    xtx = x.T.dot(x)
    x_l2 = np.array([np.diag(xtx)], dtype=np.float64)

    x_l2_vec = np.einsum("ij,ij->j", x, x)
    lbf_individual = susiepy.compute_lbf_1pop(
        V=prior_var,
        X_std=x,
        Y=y,
        X_l2=x_l2_vec,
        residual_variance=sigma2,
        return_moments=False,
        float_type=np.float64,
    )
    lbf_summary = susiepy_ss.compute_lbf(
        V=np.array([prior_var], dtype=np.float64),
        XTY_list=[xty.astype(np.float64)],
        XTX_list=[xtx.astype(np.float64)],
        X_l2_arr=x_l2,
        rho=np.array([[1.0]], dtype=np.float64),
        residual_variance=np.array([sigma2], dtype=np.float64),
        return_moments=False,
        float_type=np.float64,
    )

    lbf_direct = np.array(
        [
            _direct_log_bf_single_predictor(x[:, j], y, sigma2, prior_var)
            for j in range(p)
        ],
        dtype=float,
    )

    np.testing.assert_allclose(lbf_individual, lbf_direct, rtol=0, atol=1e-10)
    np.testing.assert_allclose(lbf_summary, lbf_direct, rtol=0, atol=1e-10)
    np.testing.assert_allclose(lbf_individual, lbf_summary, rtol=0, atol=1e-12)
    assert np.isclose(lbf_direct[0], 0.0)


@pytest.mark.parametrize(
    "prior_variance",
    [
        np.array([0.25, 0.15], dtype=np.float64),
        np.array([0.0, 0.15], dtype=np.float64),
    ],
)
def test_multipop_compute_lbf_matches_direct_bayesian_linear_regression(prior_variance):
    rng = np.random.default_rng(2028)
    n_list = [14, 11]
    p = 5
    residual_variance = np.array([1.1, 0.9], dtype=np.float64)
    rho = np.array([[1.0, 0.55], [0.55, 1.0]], dtype=np.float64)

    x_list = [rng.normal(size=(n, p)) for n in n_list]
    x_list[0][:, 0] = 0.0
    x_list[1][:, 0] = 0.0
    y_list = [rng.normal(size=n) for n in n_list]

    x_l2_arr = np.array([np.einsum("ij,ij->j", x, x) for x in x_list], dtype=np.float64)
    xty_list = [x.T.dot(y) for x, y in zip(x_list, y_list)]
    xtx_list = [x.T.dot(x) for x in x_list]

    lbf_individual = susiepy.compute_lbf(
        V=prior_variance,
        Y_list=y_list,
        X_std_list=x_list,
        X_l2_arr=x_l2_arr,
        rho=rho,
        residual_variance=residual_variance,
        return_moments=False,
        float_type=np.float64,
    )
    lbf_summary = susiepy_ss.compute_lbf(
        V=prior_variance,
        XTY_list=xty_list,
        XTX_list=xtx_list,
        X_l2_arr=x_l2_arr,
        rho=rho,
        residual_variance=residual_variance,
        return_moments=False,
        float_type=np.float64,
    )
    lbf_direct = np.array(
        [
            _direct_multi_pop_log_bf_for_variant(
                x_list=x_list,
                y_list=y_list,
                residual_variance=residual_variance,
                rho=rho,
                prior_variance=prior_variance,
                variant_idx=j,
            )
            for j in range(p)
        ],
        dtype=float,
    )

    np.testing.assert_allclose(lbf_individual, lbf_direct, rtol=0, atol=1e-10)
    np.testing.assert_allclose(lbf_summary, lbf_direct, rtol=0, atol=1e-10)
    np.testing.assert_allclose(lbf_individual, lbf_summary, rtol=0, atol=1e-12)
    assert np.isclose(lbf_direct[0], 0.0)


def test_single_population_posterior_moments_match_direct_gaussian_conditioning():
    rng = np.random.default_rng(2029)
    n = 26
    p = 6
    sigma2 = np.array([1.4], dtype=np.float64)
    prior_variance = np.array([0.2], dtype=np.float64)
    rho = np.array([[1.0]], dtype=np.float64)

    x = rng.normal(size=(n, p))
    y = rng.normal(size=n)

    x_list = [x]
    y_list = [y]
    x_l2_arr = np.array([np.einsum("ij,ij->j", x, x)], dtype=np.float64)
    xty_list = [x.T.dot(y)]
    xtx_list = [x.T.dot(x)]

    _, post_mean_indiv, post_mean2_indiv = susiepy.compute_lbf(
        V=prior_variance,
        Y_list=y_list,
        X_std_list=x_list,
        X_l2_arr=x_l2_arr,
        rho=rho,
        residual_variance=sigma2,
        return_moments=True,
        float_type=np.float64,
    )
    _, post_mean_ss, post_mean2_ss = susiepy_ss.compute_lbf(
        V=prior_variance,
        XTY_list=xty_list,
        XTX_list=xtx_list,
        X_l2_arr=x_l2_arr,
        rho=rho,
        residual_variance=sigma2,
        return_moments=True,
        float_type=np.float64,
    )

    direct_means = []
    direct_second = []
    for j in range(p):
        m, s2 = _direct_multi_pop_posterior_for_variant(
            x_list=x_list,
            y_list=y_list,
            residual_variance=sigma2,
            rho=rho,
            prior_variance=prior_variance,
            variant_idx=j,
        )
        direct_means.append(m[0])
        direct_second.append(s2[0, 0])

    direct_means = np.array(direct_means, dtype=float)
    direct_second = np.array(direct_second, dtype=float)
    np.testing.assert_allclose(post_mean_indiv[0], direct_means, rtol=0, atol=1e-10)
    np.testing.assert_allclose(post_mean_ss[0], direct_means, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        post_mean2_indiv[0, 0], direct_second, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(post_mean2_ss[0, 0], direct_second, rtol=0, atol=1e-10)


def test_multipopulation_posterior_moments_match_direct_gaussian_conditioning():
    rng = np.random.default_rng(2030)
    n_list = [13, 11]
    p = 5
    residual_variance = np.array([1.2, 0.8], dtype=np.float64)
    prior_variance = np.array([0.25, 0.15], dtype=np.float64)
    rho = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)

    x_list = [rng.normal(size=(n, p)) for n in n_list]
    y_list = [rng.normal(size=n) for n in n_list]
    x_l2_arr = np.array([np.einsum("ij,ij->j", x, x) for x in x_list], dtype=np.float64)
    xty_list = [x.T.dot(y) for x, y in zip(x_list, y_list)]
    xtx_list = [x.T.dot(x) for x in x_list]

    _, post_mean_indiv, post_mean2_indiv = susiepy.compute_lbf(
        V=prior_variance,
        Y_list=y_list,
        X_std_list=x_list,
        X_l2_arr=x_l2_arr,
        rho=rho,
        residual_variance=residual_variance,
        return_moments=True,
        float_type=np.float64,
    )
    _, post_mean_ss, post_mean2_ss = susiepy_ss.compute_lbf(
        V=prior_variance,
        XTY_list=xty_list,
        XTX_list=xtx_list,
        X_l2_arr=x_l2_arr,
        rho=rho,
        residual_variance=residual_variance,
        return_moments=True,
        float_type=np.float64,
    )

    for j in range(p):
        m_direct, s2_direct = _direct_multi_pop_posterior_for_variant(
            x_list=x_list,
            y_list=y_list,
            residual_variance=residual_variance,
            rho=rho,
            prior_variance=prior_variance,
            variant_idx=j,
        )
        np.testing.assert_allclose(post_mean_indiv[:, j], m_direct, rtol=0, atol=1e-10)
        np.testing.assert_allclose(post_mean_ss[:, j], m_direct, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            post_mean2_indiv[:, :, j], s2_direct, rtol=0, atol=1e-10
        )
        np.testing.assert_allclose(
            post_mean2_ss[:, :, j], s2_direct, rtol=0, atol=1e-10
        )
