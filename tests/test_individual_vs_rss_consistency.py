import numpy as np

import MultiSuSiE


def test_individual_matches_summary_statistics(synthetic_data):
    ss_fit = MultiSuSiE.multisusie_rss(
        b_list=synthetic_data.beta_hat_list,
        s_list=synthetic_data.se_list,
        R_list=synthetic_data.r_list,
        varY_list=synthetic_data.vary_list,
        population_sizes=synthetic_data.n_list,
        low_memory_mode=False,
        single_population_mac_thresh=0,
        **synthetic_data.common,
    )
    indiv_fit = MultiSuSiE.multisusie(
        X_list=synthetic_data.geno_list,
        Y_list=synthetic_data.y_list,
        standardize=False,
        **synthetic_data.common,
    )
    np.testing.assert_allclose(ss_fit.pip, indiv_fit.pip, rtol=0, atol=1e-10)
