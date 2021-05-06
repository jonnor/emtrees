
import os.path
import os

import numpy

from . import common

# Ref 
"""
References

https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/mixture/_gaussian_mixture.py#L380

https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/mixture/_base.py

log probability is used
implementation depends on covariance type. Can be known at fit time

Looks like log_det can be known at fit time
one constant per component

Stores
means_ means of each component
weights_ weights of each component

precisions_cholesky_
which is a Cholesky decomposition of the precision, the inverse of the covariance matrix

BaysianGaussianMixture
https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/mixture/_bayesian_mixture.py

Seem to reuse _estimate_log_gaussian_prob from GMM
Has an additional term log_lambda
does not seem to depend on X?

"""

class Wrapper:
    def __init__(self, estimator, classifier, dtype='float'):
        self.dtype = dtype

    def predict(self, X):
        predictions = numpy.zeros(shape=X.shape)
        return predictions

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

