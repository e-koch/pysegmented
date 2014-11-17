
'''
Test Suite
'''

from lm_seg import Lm_Seg

import numpy as np
import numpy.testing as test

np.random.seed(12345)


def lmseg_test():
    x = np.linspace(0, 10, 100)

    y = 2 + 2*x*(x < 5) + (5*x - 15)*(x >= 5) + np.random.normal(0, 0.1, 100)

    model = Lm_Seg(x, y, 3)
    model.fit_model(tol=2e-2)

    test.assert_approx_equal(5.0, model.brk, significant=2)
    test.assert_allclose([2.0, 5.0], model.slopes, rtol=0.1)
