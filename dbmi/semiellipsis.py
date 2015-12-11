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

def get_abc(x, y, N_d=10., self_consistent=False):
    """Return parameters of semi-elliptical model for given distribution.
    The area under the curve is normalized to N_d = 10.
    """
    tol = 1e-7

    E_d = first_moment(x, y)
    W_d = width(x, y)

    if not self_consistent:
        return W_d / 2, 4*N_d/W_d/np.pi, E_d

    a, b, c = get_abc(x, y)
    while True:
        a1, b1, c1 = get_abc(np.ma.masked_outside(x, c-a, c+a),
                          np.ma.masked_outside(y, c-a, c+a),)
        if np.abs(b1 - b) < tol:
            break
        a, b, c = a1, b1, c1

    return a, b, c


def ellipsis(x, a=1., b=1., c=0.):
    """Return (semi-)ellipsis parametrized by
        a: half-width
        b: height
        c: center
    """
    y = b*np.sqrt(1-((x-c)/a)**2)
    y[np.isnan(y)] = 0.
    return y

def get_ellipsis_and_hilbert(x, y, N_d=10., self_consistent=False):
    a, b, c = get_abc(x, y, N_d=N_d, self_consistent=self_consistent)
    ellipsis_model = ellipsis(x, a=a, b=b, c=c)
    hilbert_transform = np.imag(scipy.signal.hilbert(ellipsis_model))
    return ellipsis_model, hilbert_transform

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt


    x = np.arange(-10, 10, .1)
    noise = np.random.random(x.shape) * .3
    y = ellipsis(x, 6, 1, -2.4) + noise

    for self_consistent in [True, False]:

        raw_hilbert = np.imag(scipy.signal.hilbert(y))
        el, H = get_ellipsis_and_hilbert(x, y, N_d=10, self_consistent=self_consistent)
        plt.plot(x, el, '-', label='semi-elliptic model {self_consistent}'.format(**locals()))
        plt.plot(x, H, '--', label='elliptic Hilbert transform {self_consistent}'.format(**locals()))

    plt.xlim((-20., 20))
    plt.plot(x, y, 'k-', label='raw signal')
    plt.plot(x, raw_hilbert, 'k--', label='raw Hilbert')


    leg = plt.legend(loc='best')
    leg.get_frame().set_alpha(.5)

    plt.savefig('semielliptic_model_test.pdf', bbox_inches='tight')


