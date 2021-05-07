
import os.path
import os

import numpy

from . import common

# Ref 
"""
References

https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/mixture/_gaussian_mixture.py#L380

https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/mixture/_base.py

_estimate_log_gaussian_prob

log probability is used
implementation depends on covariance type. Can be known at fit time

Looks like log_det can be known at fit time
one constant per component

Stores
means_ means of each component
weights_ weights of each component

precisions_cholesky_
which is a Cholesky decomposition of the precision, the inverse of the covariance matrix

for full. components, features, features
for spherical. components
for diag. components, features
for tied, features, features 

BaysianGaussianMixture
https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/mixture/_bayesian_mixture.py

Seem to reuse _estimate_log_gaussian_prob from GMM
Has an additional term log_lambda
does not seem to depend on X?


https://www.vlfeat.org/api/gmm.html
implements only diagonal covariance matrix

https://github.com/vlfeat/vlfeat/blob/master/vl/gmm.c#L712

"""

class Wrapper:
    def __init__(self, estimator, classifier, dtype='float'):
        self.dtype = dtype


        n_components, n_features = estimator.means_.shape
        print("est shape", n_components, n_features)
        covariance_type = estimator.covariance_type
        precisions_chol = estimator.precisions_cholesky_

        from sklearn.mixture._gaussian_mixture import _compute_log_det_cholesky

        log_det = _compute_log_det_cholesky(
            precisions_chol, covariance_type, n_features)

        #print("log_det", log_det.shape)
        #print("means", estimator.means_.shape)
        #print("prec", precisions_chol.shape)

        self._log_det = log_det
        self._means = estimator.means_.copy()
        self._covariance_type = covariance_type
        self._precisions_col = precisions_chol
        self._weights = estimator.weights_

    def predict_proba(self, X):
        from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob
        predictions = _estimate_log_gaussian_prob(X, self._means,
                                    self._precisions_col, self._covariance_type)

        predictions += numpy.log(self._weights)

        return predictions

    def predict(self, X):
        probabilities = self.predict_proba(X)
        predictions = numpy.argmax(probabilities, axis=1)
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

