/* Instruction counter on QEMU run with -icount shift=0 (1 instruction = 1 ns of
   virtual time), read from the CMSDK Timer0 of the MPS2 board (0x40000000).
   The instructions per tick factor is calibrated by step08_cortex_m7/build.py on a
   loop of known length, then applied to the measurements. */
#ifndef TIMER_H
#define TIMER_H
#include <stdint.h>
#define TIMER0_CTRL   (*(volatile uint32_t *)0x40000000)
#define TIMER0_VALUE  (*(volatile uint32_t *)0x40000004)
#define TIMER0_RELOAD (*(volatile uint32_t *)0x40000008)

static inline void timer_start(void)
{
    TIMER0_CTRL = 0;
    TIMER0_RELOAD = 0xFFFFFFFFu;
    TIMER0_VALUE = 0xFFFFFFFFu;
    TIMER0_CTRL = 1;
}

static inline uint32_t timer_read(void) { return 0xFFFFFFFFu - TIMER0_VALUE; }   /* elapsed ticks */
#endif
