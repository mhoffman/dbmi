#!/usr/bin/env python

import numpy as np
import scipy.signal

def first_moment(x, y):
    """Return the first moment (mean) of given distribution
    """
    return np.trapz(x*y, x) / np.trapz(y,x)

def second_moment(x, y):
    """Return centered second moment of given distribution"""

    return np.trapz((x - first_moment(x, y))**2 * y, x)/ np.trapz(y, x)

def width(x, y):
    """"Return width of given distribution using the semilelliptical model
    """
    return 4*np.sqrt(second_moment(x, y))

def get_abc(x, y, N_d=10.):
    """Return parameters of semi-elliptical model for given distribution.
    The area under the curve is normalized to N_d = 10.
    """
    E_d = first_moment(x, y)
    W_d = width(x, y)
    return W_d / 2, 4*N_d/W_d/np.pi, E_d

def ellipsis(x, a=1., b=1., c=0.):
    """Return (semi-)ellipsis parametrized by
        a: half-width
        b: height
        c: center
    """
    y = b*np.sqrt(1-((x-c)/a)**2)
    y[np.isnan(y)] = 0.
    return y

def get_ellipsis_and_hilbert(x, y, N_d=10.):
    a, b, c = get_abc(x, y, N_d=N_d)
    ellipsis_model = ellipsis(x, a=a, b=b, c=c)
    hilbert_transform = np.imag(scipy.signal.hilbert(ellipsis_model))
    return ellipsis_model, hilbert_transform

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt


    x = np.arange(-10, 10, .1)
    noise = np.random.random(x.shape) * .1
    y = ellipsis(x, 6, 1, -2.4) + noise
    raw_hilbert = np.imag(scipy.signal.hilbert(y))
    el, H = get_ellipsis_and_hilbert(x, y)

    plt.xlim((-10., 10))
    plt.plot(x, y, 'k-', label='raw signal')
    plt.plot(x, raw_hilbert, 'k--', label='raw Hilbert')

    plt.plot(x, el, 'r-', label='semi-elliptic model')
    plt.plot(x, H, 'r--', label='elliptic Hilbert transform')

    leg = plt.legend(loc='best')
    leg.get_frame().set_alpha(.5)

    plt.savefig('semielliptic_model_test.pdf', bbox_inches='tight')


