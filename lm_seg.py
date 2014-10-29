
'''
Port of the R seg.lm.fit function
'''

from statsmodels import OLS
import numpy as np
import warning

def lm_seg(y, x, Z, brk, tol=1, iter_max=100, h_step=0.1):
    '''
    Fit a segmented model with OLS
    '''

    c1 = (Z <= brk)
    c2 = (Z >= brk)

    if np.sum(c1 + c2, axis=None) == 0:
        raise ValueError("brk is outside the range.")

    # Fit a normal linear model to the data
    init_lm = OLS.fit(y=y, x=x)

    # Get the epsilon value from the fit
    epsil_0 = None
    epsil = None

    # Before we get into the loop, make sure that this was a bad fit
    if epsil_0 < tol:
        warning.warning('Initial epsilon is smaller than tolerance. \
                         Data may not have a break, or the tolerance \
                         should be smaller.')
        return init_lm

    # Sum of residuals
    dev_0 = None #np.sum()

    # Count
    it = 0

    # Now loop through and minimize the residuals by changing where the
    # breaking point is.
    while epsil > tol:
        U = np.max(Z - brk, axis=0)
        V = deriv_max(Z, brk)

        X_all = np.vstack([x, U, V])

        fit = OLS.fit(y=y, x=X_all)

        dev_1 = None  # np.sum()

        epsil = (dev_1 - dev_0) / (dev_0 + 1e-3)

        it += 1

        if it > iter_max:
            break

        beta = None  # Get coef
        gamma = None  # Get coef

        brk += (h_step * gamma) / beta



def deriv_max(a, b, pow=1):
    if pow == 1:
        dum = -1 * np.ones(a.shape)
        dum[a < b] = 0
        return dum
    else:
        return -pow * np.max(a - b, axis=0) ** (pow-1)
