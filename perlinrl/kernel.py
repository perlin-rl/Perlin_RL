import numpy as np
import scipy


def rbf(l=1.0):
    return se(sig=1.414, l=l)


def se(sig=1.414, l=1.0):
    def func(xa, xb):
        sq = scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean') / -2*l
        return (sig**2)*np.exp(sq)
    return func


def brown(f=0.02, eps=0.00000001):
    def func(xa, xb):
        m = []
        for i in range(xa.shape[0]):
            l = []
            for j in range(xb.shape[0]):
                l.append(min(xa[i][0], xb[j][0]))
            m.append(l)

        return (np.array(m) + eps)*f
    return func


def pink(sig=1.414):
    def func(xa, xb):
        d = scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
        m = (d == 1)*1 + (d == 0)*sig
        return m
    return func
