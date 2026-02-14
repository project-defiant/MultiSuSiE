import numpy as np

import MultiSuSiE


def test_low_memory_false_does_not_mutate_inputs(synthetic_data):
    r_list_copy = [r.copy() for r in synthetic_data.r_list]
    beta_hat_copy = [b.copy() for b in synthetic_data.beta_hat_list]
    se_copy = [s.copy() for s in synthetic_data.se_list]

    MultiSuSiE.multisusie_rss(
        b_list=synthetic_data.beta_hat_list,
        s_list=synthetic_data.se_list,
        R_list=synthetic_data.r_list,
        varY_list=synthetic_data.vary_list,
        population_sizes=synthetic_data.n_list,
        single_population_mac_thresh=5,
        low_memory_mode=False,
        **synthetic_data.common,
    )

    for r_before, r_after in zip(r_list_copy, synthetic_data.r_list):
        assert np.nanmax(np.abs(r_before - r_after)) < 1e-10
        assert (np.isnan(r_before) == np.isnan(r_after)).all()

    for b_before, b_after in zip(beta_hat_copy, synthetic_data.beta_hat_list):
        assert np.nanmax(np.abs(b_before - b_after)) < 1e-10
        assert (np.isnan(b_before) == np.isnan(b_after)).all()

    for s_before, s_after in zip(se_copy, synthetic_data.se_list):
        assert np.nanmax(np.abs(s_before - s_after)) < 1e-10
        assert (np.isnan(s_before) == np.isnan(s_after)).all()


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
