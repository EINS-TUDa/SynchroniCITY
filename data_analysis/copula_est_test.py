import pytest
from pytest import approx, fixture
from copula_syncest_gaus import most_likely_tau as mlt_gaus
from copula_syncest_arch import most_likely_tau as mlt_arch
import numpy as np
from timeit import timeit


def _test_est(estim_func):
    for ui in np.linspace(0.1, 0.9, 9):
        u = [ui, ui]
        print(ui)
        assert estim_func(u) == approx(1, abs=3e-2)
    assert estim_func([.1, .9]) == approx(0, abs=1e-2)
    assert estim_func([.2, .8]) == approx(0, abs=1e-2)
    assert estim_func([0.1, 0.3, 0.9]) == approx(0, abs=1e-2)
    assert estim_func([0.1, 0.3, 0.9, .6, .3, .1, .9]) == approx(0, abs=1e-2)
    assert estim_func([0.1, 0.3, 0.9, .6, .3, .1, .6, .7, .8, .9, .1, .2]) == approx(0, abs=1e-2)
    for n in range(3, 30):
        print(n)
        for ui in np.linspace(0.1, 0.9, 9):
            u = [ui] * n
            print(ui)
            assert estim_func(u) == approx(1, abs=3e-2)


def test_gaussian():
    _test_est(mlt_gaus)


def test_gumbel():
    _test_est(lambda u: mlt_arch(u, "gumbel"))


def test_clayton():
    _test_est(lambda u: mlt_arch(u, "clayton"))


def test_frank():
    _test_est(lambda u: mlt_arch(u, "frank"))


def testf_time():
    print("\n --- Run Time --- ")
    u = [.2, .3, .4, .5, .4, .3, .2]
    print("gaussian", timeit(lambda: mlt_gaus(u), number=100))
    print("gumbel", timeit(lambda: mlt_arch(u, "gumbel"), number=100))
    print("clayton", timeit(lambda: mlt_arch(u, "clayton"), number=100))
    print("frank", timeit(lambda: mlt_arch(u, "frank"), number=100))
