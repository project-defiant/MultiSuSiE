import numpy as np
import pytest

import MultiSuSiE


@pytest.mark.parametrize("method", ["individual", "rss"])
def test_elbo_is_monotone_in_restricted_coordinate_ascent_regime(synthetic_data, method):
    kwargs = dict(synthetic_data.common)
    kwargs.update(
        estimate_residual_variance=False,
        estimate_prior_variance=False,
        estimate_prior_method=None,
        iter_before_zeroing_effects=999,
        max_iter=40,
        tol=1e-12,
        min_abs_corr=0,
    )

    if method == "individual":
        fit = MultiSuSiE.multisusie(
            X_list=[x.copy() for x in synthetic_data.geno_list],
            Y_list=[y.copy() for y in synthetic_data.y_list],
            standardize=False,
            **kwargs,
        )
    else:
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

    assert len(fit.elbo) == fit.niter
    assert np.all(np.isfinite(fit.elbo))
    delta = np.diff(fit.elbo)
    assert np.all(delta >= -1e-12)
    if method == "rss":
        assert fit.niter == kwargs["max_iter"]
        assert fit.converged is False
    else:
        assert fit.niter < kwargs["max_iter"]
        assert fit.converged is True
