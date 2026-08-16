# `Intrinsics` — SIMD dispatch and CPU capability

`CpuFeatures` reports what the running CPU supports; `Simd` holds the vector helpers the kernels
dispatch through. Two files, and one of them has caused the single largest performance defect in this
project's history.

## Native-AOT compiles for the *baseline* instruction set

A JIT-compiled process inspects the CPU at startup and uses AVX2 if it is there. **An ahead-of-time
compiled binary does not** — without an explicit instruction-set property it targets the baseline x86-64
ISA, so every `Vector256` path silently degrades to the software fallback. Measured effect: SIMD decode
ran about **6× slower** under Native AOT than under the JIT, with no warning, no error, and identical
output.

The fix is `IlcInstructionSet=avx2` in the publishing project. `x86-x64-v3` and `fma` were rejected by
the toolchain; `avx2` is the value that works.

**This is not detectable by running the code** — it produces correct results at every step. If you add
a project that publishes AOT, set the property, and if a number is inexplicably several times worse
under AOT than under the JIT, look here first.

`CpuFeatures` is the runtime-side counterpart: it reports what is actually available so a kernel can
choose, and a hardware check that never sees AVX2 on a machine that has it is the symptom of the above.
