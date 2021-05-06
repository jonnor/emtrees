
import os.path
import os

import numpy

from . import common


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

