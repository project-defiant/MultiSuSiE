import numpy as np

import MultiSuSiE


def test_rss_float32_and_float64_are_close(synthetic_data):
    base_kwargs = dict(
        b_list=[b.copy() for b in synthetic_data.beta_hat_list],
        s_list=[s.copy() for s in synthetic_data.se_list],
        R_list=[r.copy() for r in synthetic_data.r_list],
        varY_list=synthetic_data.vary_list,
        population_sizes=synthetic_data.n_list,
        single_population_mac_thresh=0,
        low_memory_mode=False,
        rho=synthetic_data.common["rho"],
        L=synthetic_data.common["L"],
        scaled_prior_variance=synthetic_data.common["scaled_prior_variance"],
        min_abs_corr=synthetic_data.common["min_abs_corr"],
        estimate_prior_method=synthetic_data.common["estimate_prior_method"],
        pop_spec_effect_priors=synthetic_data.common["pop_spec_effect_priors"],
        iter_before_zeroing_effects=synthetic_data.common[
            "iter_before_zeroing_effects"
        ],
    )

    fit64 = MultiSuSiE.multisusie_rss(float_type=np.float64, **base_kwargs)
    fit32 = MultiSuSiE.multisusie_rss(float_type=np.float32, **base_kwargs)

    np.testing.assert_allclose(fit64.pip, fit32.pip, rtol=0, atol=1e-5)
