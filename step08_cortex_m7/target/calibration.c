/* Loop of known length: N iterations of 16 instructions (14 nop, subs, bne). */
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

int main(void)
{
    const uint32_t N = 1000000u;
    uint32_t n = N;
    timer_start();
    uint32_t t0 = timer_read();
    __asm volatile(
        "1:\n"
        " nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n nop\n"
        " subs %0, %0, #1\n"
        " bne 1b\n"
        : "+r"(n) : : "cc");
    uint32_t t1 = timer_read();
    printf("CALIBRATION instructions=%lu ticks=%lu\n", (unsigned long)(16u * N), (unsigned long)(t1 - t0));
    exit(0);
}
