/* Emulated Cortex-M7 (QEMU mps2-an500): 4 MB of code at 0x0, 4 MB of data at 0x20000000.
   Actual sizes are read from the binary (arm-none-eabi-size), not from this map. */
MEMORY
{
  FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
  RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}
ENTRY(Reset_Handler)
_stack_top = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
  .vectors : { KEEP(*(.vectors)) } > FLASH
  .text : { *(.text*) KEEP(*(.init)) KEEP(*(.fini)) } > FLASH
  .rodata : { *(.rodata*) KEEP(*(.nn_data*)) } > FLASH      /* network weights (Aidge) */
  .test_inputs : { KEEP(*(.test_inputs*)) } > FLASH         /* test measurements, outside the model */
  .ARM.extab : { *(.ARM.extab* .gnu.linkonce.armextab.*) } > FLASH
  .ARM.exidx : { __exidx_start = .; *(.ARM.exidx*) __exidx_end = .; } > FLASH
  .preinit_array : { PROVIDE_HIDDEN(__preinit_array_start = .); KEEP(*(.preinit_array*)); PROVIDE_HIDDEN(__preinit_array_end = .); } > FLASH
  .init_array : { PROVIDE_HIDDEN(__init_array_start = .); KEEP(*(SORT(.init_array.*))); KEEP(*(.init_array*)); PROVIDE_HIDDEN(__init_array_end = .); } > FLASH
  .fini_array : { PROVIDE_HIDDEN(__fini_array_start = .); KEEP(*(.fini_array*)); PROVIDE_HIDDEN(__fini_array_end = .); } > FLASH
  /* QEMU loads the ELF directly: .data is placed in RAM without a copy from Flash */
  .data : { *(.data*) } > RAM
  .bss (NOLOAD) : { __bss_start__ = .; *(.bss*) *(COMMON) __bss_end__ = .; } > RAM
  end = .;
  __end__ = .;
  _end = .;
}
