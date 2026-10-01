#include <stdio.h>
#include <stdlib.h>
#include "pca_data.h"
#include "timer.h"

float pca_indicator(const float *x, uint8_t *faulty);

int main(void)
{
    uint8_t faulty[PCA_NT];
    for (int i = 0; i < PCA_NE; ++i) {
        timer_start();
        const uint32_t t0 = timer_read();
        const float s = pca_indicator(pca_inputs + i * PCA_D, faulty);
        const uint32_t t1 = timer_read();
        int n = 0;
        for (int p = 0; p < PCA_NT; ++p)
            n += faulty[p];
        printf("PCA %d indicator=%.7g ticks=%lu empty_paths=%d\n", i, (double)s, (unsigned long)(t1 - t0), n);
    }
    exit(0);
}
