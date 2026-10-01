/* PCA detector (k components) with int8 weights, float32 arithmetic, for Cortex-M.
   Same computation as PCAModule(int8_weights=True) (src/ogw/embedded.py):
     xn = x / norm[p] ;  f = xn - mu ;  z = V f ;  r = f - V^T z ;
     res[p] = ||r_p||² / ||xn_p||² / res_scale[p] ; 0 if the path is empty ;
     indicator = max_p res[p].
   Two passes over the signal: in RAM, only the input and k + NT scalars. */
#include <stdint.h>
#include "pca_data.h"

float pca_indicator(const float *x, uint8_t *faulty)
{
    float z[PCA_K] = {0};
    for (int p = 0; p < PCA_NT; ++p) {                  /* pass 1: energy, projection */
        const float *xp = x + p * PCA_L;
        const int8_t *mu = pca_q_mu + p * PCA_L;
        const float inv = 1.0f / pca_norm[p], smu = pca_s_mu[p];
        float e = 0.0f;
        for (int t = 0; t < PCA_L; ++t) {
            e += xp[t] * xp[t];
            const float f = xp[t] * inv - smu * mu[t];
            for (int k = 0; k < PCA_K; ++k)
                z[k] += f * pca_q_V[k * PCA_D + p * PCA_L + t];
        }
        faulty[p] = e < 0.3f * pca_energy[p];
    }
    for (int k = 0; k < PCA_K; ++k)
        z[k] *= pca_s_V[k] * pca_s_V[k];               /* z = s V q . f, then reconstruction s q . z */
    float best = 0.0f;
    for (int p = 0; p < PCA_NT; ++p) {                  /* pass 2: residual per path */
        if (faulty[p])
            continue;
        const float *xp = x + p * PCA_L;
        const int8_t *mu = pca_q_mu + p * PCA_L;
        const float inv = 1.0f / pca_norm[p], smu = pca_s_mu[p];
        float num = 0.0f, den = 0.0f;
        for (int t = 0; t < PCA_L; ++t) {
            const float xn = xp[t] * inv, f = xn - smu * mu[t];
            float rec = 0.0f;
            for (int k = 0; k < PCA_K; ++k)
                rec += z[k] * pca_q_V[k * PCA_D + p * PCA_L + t];
            const float r = f - rec;
            num += r * r;
            den += xn * xn;
        }
        const float res = num / (den > 1e-12f ? den : 1e-12f) / pca_res_scale[p];
        if (res > best)
            best = res;
    }
    return best;
}
