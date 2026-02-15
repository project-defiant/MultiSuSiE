import numpy as np
import pytest

import MultiSuSiE


def _minimal_inputs():
    r = np.eye(2)
    return {
        "R_list": [r],
        "population_sizes": [100],
        "b_list": [np.array([0.1, 0.2])],
        "s_list": [np.array([0.05, 0.07])],
        "z_list": [np.array([2.0, 3.0])],
        "varY_list": [1.0],
    }


def test_raises_if_both_z_and_b_s_are_provided():
    d = _minimal_inputs()
    with pytest.raises(
        ValueError,
        match="provide either \\(b_list and s_list\\) or z_list, but not both",
    ):
        MultiSuSiE.multisusie_rss(
            R_list=d["R_list"],
            population_sizes=d["population_sizes"],
            b_list=d["b_list"],
            s_list=d["s_list"],
            z_list=d["z_list"],
            varY_list=d["varY_list"],
        )


def test_raises_if_neither_z_nor_b_s_is_provided():
    d = _minimal_inputs()
    with pytest.raises(
        ValueError,
        match="provide either \\(b_list and s_list\\) or z_list, but not both",
    ):
        MultiSuSiE.multisusie_rss(
            R_list=d["R_list"],
            population_sizes=d["population_sizes"],
        )


def test_raises_if_b_without_s_or_vary():
    d = _minimal_inputs()
    with pytest.raises(
        ValueError,
        match="if b_list is provided, s_list and varY_list must also be provided",
    ):
        MultiSuSiE.multisusie_rss(
            R_list=d["R_list"],
            population_sizes=d["population_sizes"],
            b_list=d["b_list"],
        )


def test_raises_if_population_sizes_missing():
    d = _minimal_inputs()
    with pytest.raises(ValueError, match="population_sizes must be provided"):
        MultiSuSiE.multisusie_rss(
            R_list=d["R_list"],
            population_sizes=None,
            b_list=d["b_list"],
            s_list=d["s_list"],
            varY_list=d["varY_list"],
        )
