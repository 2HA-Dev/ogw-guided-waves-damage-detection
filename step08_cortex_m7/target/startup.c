/* Minimal Cortex-M7 startup: vector table, floating point unit enabled, then _start
   (newlib crt0 with semihosting: stack, .bss zeroed, constructors, main). */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

extern uint32_t _stack_top;
extern void _start(void);

void Reset_Handler(void)
{
    /* CP10 and CP11 in full access, before any floating point instruction */
    *(volatile uint32_t *)0xE000ED88 |= (0xFu << 20);
    __asm volatile("dsb\n isb");
    _start();
    for (;;) {}
}

void Default_Handler(void) { for (;;) {} }

__attribute__((section(".vectors"), used))
const void *vectors[16] = {
    &_stack_top, (void *)Reset_Handler, (void *)Default_Handler, (void *)Default_Handler,
    (void *)Default_Handler, (void *)Default_Handler, (void *)Default_Handler, 0, 0, 0, 0,
    (void *)Default_Handler, (void *)Default_Handler, 0, (void *)Default_Handler, (void *)Default_Handler};

#ifdef __cplusplus
}
#endif
