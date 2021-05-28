

#ifndef EML_MIXTURE_H
#define EML_MIXTURE_H

#include "eml_common.h"
#include "eml_fixedpoint.h"

#ifndef EML_MAX_CLASSES
#define EML_MAX_CLASSES 10
#endif


// numpy.log2(2*numpy.pi)
#define EML_LOG2_2PI 2.651496129472319f

typedef struct _EmlMixtureModel {
   int32_t n_components;
   int32_t n_features;
   float *weigths;
   float *means;
   float *log_dets;
} EmlMixtureModel;


bool
eml_dot_product(float *a, float *b, int n)
{
    float sum = 0; 
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
 
    return sum;
}


    float out[EML_MAX_CLASSES];

/*

    for (int c=0; c<n_components; c++) {

        log_prob = model->means[c]

        out[c] = -0.5 * (n_features * EML_LOG2_2PI + log_prob ) + log_det[c];
    }



    if covariance_type == 'full':
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

*/

int32_t
eml_mixture_predict_proba(EmlMixtureModel *model,
                        const float values[], int32_t values_length,
                        float *probabilities)
{

   EML_PRECONDITION(model, -EmlUninitialized);
   EML_PRECONDITION(values, -EmlUninitialized);
   EML_PRECONDITION(model->n_components > 0, -EmlUninitialized);



   return ;
}



#endif // EML_MIXTURE_H
