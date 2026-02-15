import numpy as np

import MultiSuSiE


def test_recover_r_from_xtx_roundtrip(synthetic_data):
    r_list_copy = [r.copy() for r in synthetic_data.r_list]
    xtx_list = []
    for i in range(len(r_list_copy)):
        MultiSuSiE.recover_XTX_and_XTY(
            synthetic_data.beta_hat_list[i],
            synthetic_data.se_list[i],
            r_list_copy[i],
            synthetic_data.vary_list[i],
            synthetic_data.n_list[i],
        )
        xtx_list.append(r_list_copy[i])

    x_l2_arr = np.array([np.diag(xtx) for xtx in xtx_list])
    for xtx, x_l2 in zip(xtx_list, x_l2_arr):
        MultiSuSiE.recover_R_from_XTX(xtx, x_l2)

    for r, xtx in zip(synthetic_data.r_list, xtx_list):
        assert np.nanmax(np.abs(r - xtx)) < 1e-10


def test_recover_xtx_xty_from_z_matches_standardized_data(synthetic_data):
    z_list = [
        b / s for b, s in zip(synthetic_data.beta_hat_list, synthetic_data.se_list)
    ]
    with np.errstate(divide="ignore", invalid="ignore"):
        geno_std_list = [
            geno / np.std(geno, axis=0, ddof=1) for geno in synthetic_data.geno_list
        ]
    y_std_list = [y / np.std(y, ddof=1) for y in synthetic_data.y_list]
    xtx_std_list = [geno.T.dot(geno) for geno in geno_std_list]
    xty_std_list = [geno.T.dot(y) for geno, y in zip(geno_std_list, y_std_list)]

    for i in range(len(z_list)):
        xtx, xty = MultiSuSiE.recover_XTX_and_XTY_from_Z(
            z=z_list[i],
            R=np.copy(synthetic_data.r_list[i]),
            n=synthetic_data.n_list[i],
            float_type=np.float64,
        )
        assert np.nanmax(np.abs(np.nan_to_num(xtx, 0) - xtx_std_list[i])) < 1e-10
        assert np.nanmax(np.abs(np.nan_to_num(xty, 0) - xty_std_list[i])) < 1e-10
