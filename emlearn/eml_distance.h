
/*
Mahalanobis distance

computes (x1 - x2).T * VI * (x1 - x2)
where VI is the precision matrix, the inverse of the covariance matrix
*/
float
eml_mahalanobis_distance_squared(float *x1, float *x2, float *precision, int n_features)
{
    float distance = 0.0

    for (int i=0; i<n_features; i++) {
        accumulate = 0;
        for (int j=0; j<n_features; j++) {
            accumulate += precision[i, j] * (x1[j] - x2[j]);
        }
        distance += accumulate * (x1[i] - x2[i]);
    }

    return distance;
}

typedef struct _EmlEllipticCurve {
    int n_features;
    float decision_boundary;
    float *means; // shape: n_features
    float *precision; // shape: n_features*n_features
} EmlEllipticCurve;

int
eml_elliptic_curve_predict(EmlEllipticCurve *self, float *features, int n_features)
{
    EML_PRECONDITION(n_features == self.n_features);

    float dist = eml_mahalanobis_distance_squared(features, n_features);

    
}

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
