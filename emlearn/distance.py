
import os.path
import os

import numpy

from . import common

"""

References
https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/covariance/_empirical_covariance.py#L297

https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/covariance/_elliptic_envelope.py#L149

"""

from sklearn.mixture._gaussian_mixture import _compute_log_det_cholesky
from sklearn.utils.extmath import row_norms
np = numpy


def squared_mahalanobis_distance(x1, x2, precision):
    """    
    @precision is the inverted covariance matrix

    computes (x1 - x2).T * VI * (x1 - x2)
    where VI is the precision matrix, the inverse of the covariance matrix

    Loosely based on the scikit-learn implementation,
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neighbors/_dist_metrics.pyx
    """

    distance = 0.0
    size = x1.shape[0]
    temp = numpy.zeros(shape=size) 

    assert x1.shape == x2.shape
    assert precision.shape[0] == precision.shape[1]
    assert size == precision.shape[0]

    print(x1.shape, x1.shape, precision.shape)

    # precompute x1-x2, used twice
    for i in range(size):
        temp[i] = x1[i] - x2[i]

    for i in range(size):
        accumulate = 0
        for j in range(size):
            accumulate += precision[i, j] * temp[j]
        distance += accumulate * temp[i]

    return distance


class Wrapper:
    def __init__(self, estimator, classifier, dtype='float'):
        self.dtype = dtype

        precision = estimator.get_precision()
        self._means = estimator.location_.copy()
        self._precision = precision
        self._offset = estimator.offset_

    def mahalanobis(self, X):
        def dist(x):
            return squared_mahalanobis_distance(x, self._means, precision=self._precision)

        return numpy.array([ dist(x) for x in X ])

    def predict(self, X):
        def predict_one(d):
            dist = -d
            dd = dist - self._offset
            is_inlier = 1 if dd > 0 else -1
            return is_inlier

        distances = self.mahalanobis(X)
        return numpy.array([predict_one(d) for d in distances])


    def save(self, name=None, file=None):
        if name is None:
            if file is None:
                raise ValueError('Either name or file must be provided')
            else:
                name = os.path.splitext(os.path.basename(file))[0]

        code = "" # Implement
        if file:
            with open(file, 'w') as f:
                f.write(code)

        raise NotImplementedError("TODO implement save()")
        return code

