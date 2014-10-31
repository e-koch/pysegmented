
'''
Port of the R seg.lm.fit function
'''

import statsmodels.api as sm
import numpy as np
import warnings


def lm_seg(y, x, brk, tol=1, iter_max=100, h_step=0.1,
           epsil_0=10):
    '''
    Fit a segmented model with OLS
    '''

    if brk > np.max(x) or brk < np.min(x):
        raise ValueError("brk is outside the range.")

    # Fit a normal linear model to the data
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    init_lm = model.fit(y=y, x=x)

    epsil = epsil_0

    # Before we get into the loop, make sure that this was a bad fit
    if epsil_0 < tol:
        warnings.warning('Initial epsilon is smaller than tolerance. \
                         The tolerance should be set smaller.')
        return init_lm

    # Sum of residuals
    dev_0 = np.sum(init_lm.resid**2.)

    # Count
    it = 0

    # Now loop through and minimize the residuals by changing where the
    # breaking point is.
    while epsil > tol:
        U = np.max(y - brk, axis=0)
        V = deriv_max(y, brk)

        X_all = np.vstack([x, U, V])
        X_all = sm.add_constant(X_all)

        model = sm.OLS(y, X_all)
        fit = model.fit()

        beta = fit.param[2]  # Get coef
        gamma = fit.param[3]  # Get coef

        brk += (h_step * gamma) / beta

        # How to handle this??
        if brk > np.max(x) or brk < np.min(x):
            pass

        dev_1 = np.sum(fit.resid**2.)

        epsil = (dev_1 - dev_0) / (dev_0 + 1e-3)

        it += 1

        if it > iter_max:
            break
            warnings.warning("Max iterations reached. \
                             Result may not be minimized.")

    # With the break point hopefully found, return the fit

    return fit


def deriv_max(a, b, pow=1):
    if pow == 1:
        dum = -1 * np.ones(a.shape)
        dum[a < b] = 0
        return dum
    else:
        return -pow * np.max(a - b, axis=0) ** (pow-1)
