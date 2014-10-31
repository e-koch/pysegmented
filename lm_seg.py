
'''
Port of the R seg.lm.fit function
'''

import statsmodels.api as sm
import numpy as np
import warnings


def lm_seg(y, x, brk, tol=1e-2, iter_max=100, h_step=2.0,
           epsil_0=10, verbose=True):
    '''
    Fit a segmented model with OLS
    '''

    if not (x > brk).any():
        raise ValueError("brk is outside the range.")

    # Fit a normal linear model to the data
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const)
    init_lm = model.fit(y=y, x=x)

    if verbose:
        print init_lm.summary()

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
    while np.abs(epsil) > tol:
        U = (x - brk) * (x > brk)
        V = deriv_max(x, brk)

        X_all = np.vstack([x, U, V]).T
        X_all = sm.add_constant(X_all)

        model = sm.OLS(y, X_all)
        fit = model.fit()

        if verbose:
            print "Iteration: %s/%s" % (it+1, iter_max)
            print fit.summary()
            print "Break Point: " + str(brk)
            print "Epsilon: " + str(epsil)

        beta = fit.params[2]  # Get coef
        gamma = fit.params[3]  # Get coef

        # Adjust the break point
        brk += (h_step * gamma) / beta

        # How to handle this??
        # if not (x > brk).any():
        #     pass

        dev_1 = np.sum(fit.resid**2.)

        epsil = (dev_1 - dev_0) / (dev_0 + 1e-3)

        it += 1

        if it > iter_max:
            break
            warnings.warning("Max iterations reached. \
                             Result may not be minimized.")

    # With the break point hopefully found, do a final good fit
    U = (x - brk) * (x > brk)
    V = deriv_max(x, brk)

    X_all = np.vstack([x, U, V]).T
    X_all = sm.add_constant(X_all)

    model = sm.OLS(y, X_all)
    fit = model.fit()

    brk_err = brk_errs(fit.params, fit.cov_params())
    return fit, (brk, brk_err)


def deriv_max(a, b, pow=1):
    if pow == 1:
        dum = -1 * np.ones(a.shape)
        dum[a < b] = 0
        return dum
    else:
        return -pow * np.max(a - b, axis=0) ** (pow-1)


def brk_errs(params, cov):
    '''
    Given the covariance matrix of the fits, calculate the standard
    error on the break.
    '''

    # Var gamma
    term1 = cov[3, 3]

    # Var beta * (beta/gamma)^2
    term2 = cov[2, 2] * (params[3]/params[2])**2.

    # Correlation b/w gamma and beta
    term3 = 2 * cov[3, 2] * (params[3]/params[2])

    return np.sqrt(term1 + term2 + term3)
