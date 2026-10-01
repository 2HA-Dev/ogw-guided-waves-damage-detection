"""Step 8: the detectors on an emulated Cortex-M7 (QEMU mps2-an500).

Prerequisites: xPack tools in arm_tools/ (arm-none-eabi-gcc 15.2.1 toolchain, QEMU 9.2.4);
Step 5 Aidge exports (cnn2d_16_int8_nchw, cnn1d_f32_ncw).
For each firmware: compilation (-mcpu=cortex-m7, -O3), sizes read from the
binary (Flash = code + constants; RAM = data + .bss, excluding the stack), execution under
QEMU with -icount shift=0 (1 instruction = 1 virtual ns), number of instructions
per inference read from a timer and calibrated on a loop of known length.
Outputs compared with the reference (Python for the PCA, host program for Aidge).
The instruction counts are an exact measurement of the emulator; the durations in ms
are only an estimate (1 instruction per cycle assumed), not a measurement on a board.
Run (from the repository root): .venv/bin/python step08_cortex_m7/build.py
"""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import torch

from ogw.anomaly import PCADetector
from ogw.data import load_meta, load_signals
from ogw.embedded import PCAModule

R = Path(__file__).resolve().parents[1]
TARGET = R / "step08_cortex_m7" / "target"
OUT = R / "results" / "step08_cortex_m7"
BUILD = OUT / "build"
GCC = next((R / "arm_tools").glob("xpack-arm-none-eabi-gcc-*")) / "bin"
QEMU = next((R / "arm_tools").glob("xpack-qemu-arm-*")) / "bin" / "qemu-system-arm"
ARCH = ["-mcpu=cortex-m7", "-mthumb", "-mfpu=fpv5-d16", "-mfloat-abi=hard"]
OPT = ["-O3", "-ffast-math", "-ffunction-sections", "-fdata-sections"]
LINK = ["--specs=rdimon.specs", f"-T{TARGET / 'link.ld'}", "-Wl,--gc-sections", "-Wl,--no-warn-rwx-segments"]
F_MHZ = 480            # clock of a common Cortex-M7 (STM32H7), for the order of magnitude in ms
S5 = R / "results" / "step05_aidge_export" / "exports"


def compile_elf(name, sources, includes=(), cxx=False):
    elf = BUILD / f"{name}.elf"
    cc = str(GCC / ("arm-none-eabi-g++" if cxx else "arm-none-eabi-gcc"))
    extra = ["-std=c++17", "-fno-exceptions", "-fno-rtti"] if cxx else ["-std=c11"]
    cmd = [cc, *ARCH, *OPT, *extra, f"-I{TARGET}", *[f"-I{i}" for i in includes], str(TARGET / "startup.c"),
           *map(str, sources), *LINK, "-o", str(elf)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])
    return elf


def sizes(elf, symbols_outside_model=()):
    out = subprocess.run([str(GCC / "arm-none-eabi-size"), "-A", str(elf)], capture_output=True, text=True).stdout
    s = {l.split()[0]: int(l.split()[1]) for l in out.splitlines() if l.startswith(".")}
    outside = 0
    if symbols_outside_model:
        nm = subprocess.run([str(GCC / "arm-none-eabi-nm"), "-S", "--size-sort", str(elf)],
                            capture_output=True, text=True).stdout
        for l in nm.splitlines():
            c = l.split()
            if len(c) == 4 and any(c[3].startswith(h) for h in symbols_outside_model):
                outside += int(c[1], 16)
    flash = sum(v for k, v in s.items() if k in (".vectors", ".text", ".rodata", ".ARM.extab", ".ARM.exidx",
                                                  ".preinit_array", ".init_array", ".fini_array", ".data")) - outside
    ram = s.get(".data", 0) + s.get(".bss", 0)
    return dict(sections=s, flash=flash, ram=ram, outside_model=outside + s.get(".test_inputs", 0))


def run_qemu(elf):
    r = subprocess.run([str(QEMU), "-M", "mps2-an500", "-nographic", "-monitor", "none", "-serial", "none",
                        "-semihosting-config", "enable=on,target=native", "-icount", "shift=0",
                        "-kernel", str(elf)], capture_output=True, text=True, timeout=1800)
    if r.returncode:
        raise RuntimeError(r.stdout[-800:] + r.stderr[-800:])
    return r.stdout


def to_c(name, a, c_type, section=None):
    attr = f' __attribute__((section("{section}")))' if section else ""
    fmt = (lambda v: str(int(v))) if "int" in c_type else (lambda v: f"{v:.9g}f")
    return f"const {c_type} {name}[{a.size}]{attr} = {{" + ",".join(fmt(v) for v in a.ravel()) + "};\n"


def generate_pca():
    """int8 weights of the Step 4 PCA and 4 test measurements, with the reference indicator."""
    meta, X = load_meta(), load_signals()
    s1 = (meta.damaged == 0) & (meta.cycle == 1)
    det = PCADetector(5).fit(X, np.flatnonzero(s1 & (meta.ramp == 0)), np.flatnonzero(s1 & (meta.ramp == 1)))
    mod = PCAModule(det, int8_weights=True).eval()
    s2 = np.flatnonzero((meta.damaged == 0) & (meta.cycle == 2))
    tests = np.array([s2[10], 296, np.flatnonzero(meta.position == "D04")[40],   # 296: path 8-9 empty
                      np.flatnonzero(meta.position == "D16")[40]])
    with torch.no_grad():
        expected, _ = mod(torch.from_numpy(X[tests]))
    nt, L = X.shape[1:]
    b = {k: v.numpy() for k, v in mod.state_dict().items()}
    h = ["#ifndef PCA_DATA_H", "#define PCA_DATA_H", "#include <stdint.h>",
         f"#define PCA_NT {nt}", f"#define PCA_L {L}", f"#define PCA_D {nt * L}", f"#define PCA_K {det.k}",
         f"#define PCA_NE {len(tests)}",
         *[f"extern const {t} {n}[];" for t, n in [("float", "pca_norm"), ("int8_t", "pca_q_mu"), ("float", "pca_s_mu"),
                                                   ("int8_t", "pca_q_V"), ("float", "pca_s_V"), ("float", "pca_res_scale"),
                                                   ("float", "pca_energy"), ("float", "pca_inputs")]], "#endif"]
    (BUILD / "pca_data.h").write_text("\n".join(h) + "\n")
    c = ['#include "pca_data.h"\n', to_c("pca_norm", b["norm"], "float"), to_c("pca_q_mu", b["q_mu"], "int8_t"),
         to_c("pca_s_mu", b["s_mu"], "float"), to_c("pca_q_V", b["q_V"], "int8_t"), to_c("pca_s_V", b["s_V"], "float"),
         to_c("pca_res_scale", b["res_scale"], "float"), to_c("pca_energy", b["energy"], "float"),
         to_c("pca_inputs", X[tests].astype(np.float32), "float", ".test_inputs")]
    (BUILD / "pca_data.c").write_text("".join(c))
    return expected.numpy(), tests, meta


AIDGE_HARNESS = r"""#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "forward.hpp"
#include "@HEADER@"
extern "C" {
#include "timer.h"
}
int main()
{
    @DECL@
    timer_start();
    const uint32_t t0 = timer_read();
    model_forward(@INPUT@, @ARGS@);
    const uint32_t t1 = timer_read();
    printf("AIDGE output=%.7g ticks=%lu\n", (double)s0[0], (unsigned long)(t1 - t0));
    exit(0);
}
"""


def firmware_aidge(name):
    d = S5 / name
    fwd = (d / "dnn" / "include" / "forward.hpp").read_text()
    params = [p.strip() for p in re.search(r"model_forward\s*\((.*?)\);", fwd, re.S).group(1).split(",")]
    input_name = params[0].split("*")[-1].strip()
    output_types = [p.split("*")[0].strip() for p in params[1:]]
    header = next((d / "data").glob("*.h")).name
    src = (AIDGE_HARNESS.replace("@HEADER@", f"data/{header}").replace("@INPUT@", input_name)
           .replace("@DECL@", " ".join(f"{t}* s{k} = nullptr;" for k, t in enumerate(output_types)))
           .replace("@ARGS@", ", ".join(f"&s{k}" for k in range(len(output_types)))))
    main_src = BUILD / f"main_{name}.cpp"
    main_src.write_text(src)
    sources = [main_src, *(d / "data").glob("*.cpp"), *(d / "dnn" / "src").rglob("*.cpp")]
    elf = compile_elf(name, sources, [d, d / "dnn", d / "dnn" / "include"], cxx=True)
    host = subprocess.run(["./build/run_export"], cwd=d, capture_output=True, text=True).stdout
    ref = float(host.split("\n")[1].split()[0])      # 1st value of the 1st output (host program)
    return elf, input_name, ref


if __name__ == "__main__":
    BUILD.mkdir(parents=True, exist_ok=True)
    res = {}
    # 0. Counter calibration
    out = run_qemu(compile_elf("calibration", [TARGET / "calibration.c"]))
    n_ins, ticks = map(int, re.search(r"instructions=(\d+) ticks=(\d+)", out).groups())
    ipt = n_ins / ticks
    res["calibration"] = dict(instructions=n_ins, ticks=ticks, instructions_per_tick=ipt)
    print(f"calibration: {ipt:.3f} instructions per tick", flush=True)

    # 1. PCA with int8 weights, written in C
    expected, tests, meta = generate_pca()
    elf = compile_elf("pca_int8", [TARGET / "pca.c", TARGET / "main_pca.c", BUILD / "pca_data.c"], [BUILD])
    out = run_qemu(elf)
    lines = re.findall(r"PCA (\d+) indicator=(\S+) ticks=(\d+) empty_paths=(\d+)", out)
    obtained = np.array([float(l[1]) for l in lines])
    t = sizes(elf)                     # test measurements already excluded from the count (.test_inputs section)
    res["pca_int8"] = dict(**t, indicators=obtained.tolist(), expected=expected.tolist(), measurements=tests.tolist(),
                           max_rel_diff=float(np.max(np.abs(obtained / expected - 1))),
                           empty_paths=[int(l[3]) for l in lines],
                           instructions=float(np.median([int(l[2]) for l in lines]) * ipt),
                           input_ram=int(np.prod(load_signals().shape[1:])) * 4)
    print("PCA:", {k: v for k, v in res["pca_int8"].items() if k != "sections"}, flush=True)

    # 2. and 3. Networks generated by Aidge (int8 worked around, float)
    for name, outside in (("cnn2d_16_int8_nchw", ("_0_",)), ("cnn1d_f32_ncw", ("_0_",))):
        elf, input_name, ref = firmware_aidge(name)
        out = run_qemu(elf)
        s, tc = re.search(r"output=(\S+) ticks=(\d+)", out).groups()
        t = sizes(elf, (input_name,))
        res[name] = dict(**t, output=float(s), host_output=ref, instructions=int(tc) * ipt)
        print(name, {k: v for k, v in res[name].items() if k != "sections"}, flush=True)

    (OUT / "measurements.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    ms = lambda n: n / (F_MHZ * 1e3)
    L = ["# Step 8: the detectors on an emulated Cortex-M7 (QEMU mps2-an500)", "",
         "Generated by `step08_cortex_m7/build.py`. Compiled with arm-none-eabi-gcc 15.2.1, `-mcpu=cortex-m7 -O3 "
         "-ffast-math`. **Sizes: exact** (binary). **Instructions: exact for the emulator** (counter "
         f"calibrated, {ipt:.1f} instructions per tick on a loop of " + f"{n_ins:,}"
         + " instructions)." + f" **Durations: estimate** at {F_MHZ} MHz assuming one instruction per cycle "
         "(a real Cortex-M7 can do better or worse depending on memory and pipeline).", "",
         "| detector | Flash (code + weights) | static RAM | input in RAM | instructions per inference | ≈ ms at 480 MHz | check |",
         "|---|---|---|---|---|---|---|"]
    a = res["pca_int8"]
    L.append(f"| PCA k = 5, int8 weights (hand-written C) | {a['flash'] / 1024:.0f} KB | {a['ram'] / 1024:.0f} KB "
             f"| {a['input_ram'] / 1024:.0f} KB (float32) | {a['instructions'] / 1e6:.2f} M | {ms(a['instructions']):.2f} "
             f"| max relative deviation {a['max_rel_diff']:.1e} from Python on 4 measurements |")
    for name, lib in (("cnn2d_16_int8_nchw", "int8 CNN (Aidge, padded channels)"), ("cnn1d_f32_ncw", "float32 CNN (Aidge)")):
        b = res[name]
        e = "int8, 80 × 820 bytes" if "int8" in name else "float32"
        L.append(f"| {lib} | {b['flash'] / 1024:.0f} KB | {b['ram'] / 1024:.0f} KB | {b['outside_model'] / 1024:.0f} KB ({e}) "
                 f"| {b['instructions'] / 1e6:.2f} M | {ms(b['instructions']):.2f} | output {b['output']:.6g} "
                 f"(host program: {b['host_output']:.6g}) |")
    L += ["", "PCA test measurements: " + ", ".join(f"{m} ({meta.label[m]}, indicator {v:.3g}, "
                                                   f"{n} empty path(s))" for m, v, n in
                                                   zip(a["measurements"], a["indicators"], a["empty_paths"])) + ".",
          "", "Static RAM: data and .bss (including the Aidge compute arena), excluding the stack. For reference: "
          "STM32H743, 2 MB of Flash and 1 MB of RAM; STM32F746, 1 MB and 320 KB; STM32L476, 1 MB and 128 KB."]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
