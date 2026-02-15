import numpy as np

import MultiSuSiE


def test_low_memory_true_mutates_r_inputs(synthetic_data):
    r_list = [r.copy() for r in synthetic_data.r_list]
    r_before = [r.copy() for r in r_list]

    MultiSuSiE.multisusie_rss(
        b_list=[b.copy() for b in synthetic_data.beta_hat_list],
        s_list=[s.copy() for s in synthetic_data.se_list],
        R_list=r_list,
        varY_list=synthetic_data.vary_list,
        population_sizes=synthetic_data.n_list,
        single_population_mac_thresh=0,
        low_memory_mode=True,
        **synthetic_data.common,
    )

    any_changed = any(np.nanmax(np.abs(before - after)) > 1e-10 for before, after in zip(r_before, r_list))
    assert any_changed
