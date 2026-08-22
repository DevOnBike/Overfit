// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

// ILGPU.Half and System.Half are different types and the cuBLAS overload takes ILGPU's. The alias is
// explicit rather than relying on using-order, because the two are silently interchangeable at every
// call site here and picking the wrong one is a compile error at best.
using Half = ILGPU.Half;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// The cuBLAS arms. <b>X1 is FP16 (<c>cublasHgemm</c>) and is the PRIMARY arm</b> per Amendment 1 of
    /// the plan; X2 is the FP32 <c>cublasSgemm</c> kept beside it so the FP16-against-FP32 gap on the
    /// same card, through the same library, is a measured number rather than an assumption.
    /// <para>
    /// <b>What ILGPU's wrapper can and cannot reach, verified against the package metadata on
    /// 2026-08-21 rather than assumed.</b> <c>CuBlas&lt;T&gt;.Gemm</c> has overloads for
    /// <c>ILGPU.Half</c>, <c>float</c>, <c>double</c>, <c>Float2</c> and <c>Double2</c>. There is
    /// <b>no <c>GemmEx</c></b>. That matters more than it looks: the FP16 path reachable from here is
    /// <c>cublasHgemm</c>, which accumulates in FP16, whereas a TENSOR CORE accumulates in FP32 and is
    /// reached through <c>cublasGemmEx</c> with <c>CUBLAS_COMPUTE_32F</c>. So this arm is FP16 storage
    /// AND FP16 accumulate, and it is neither the fastest nor the most accurate FP16 path the card
    /// offers. The gap is measured, not guessed: <c>--fp16-bound</c> puts FP32-accumulate at a relative
    /// L2 of 2.9e-4 and FP16-accumulate at 6.5e-3 (k = 2048) to 1.6e-2 (k = 11008).
    /// </para>
    /// <para>
    /// <c>CuBlasMathMode.TensorOpMath</c> exists on the wrapper and is deliberately NOT set. Since CUDA
    /// 11 that flag is deprecated and cuBLAS selects tensor cores itself; setting it would let the
    /// report imply the probe had switched something on when it had not.
    /// </para>
    /// <para>
    /// OFF by default and it must SKIP with a stated reason rather than fail. cuBLAS lives in
    /// <c>cublas64_*.dll</c>, which comes with the CUDA <b>redistributable</b>; the core ILGPU path
    /// needs only <c>nvcuda</c> from the display driver.
    /// </para>
    /// <para>
    /// <b>NOT ONE LINE OF THE FP16 PATH HAS EVER EXECUTED.</b> The development machine has no NVIDIA
    /// device, so <see cref="TryCreate"/> returns null here and every method below is compile-checked
    /// only. Every call is guarded so that a failure on the friend's machine degrades to a stated
    /// reason instead of taking the run down.
    /// </para>
    /// </summary>
    internal sealed class CuBlasArm : IDisposable
    {
        /// <summary>
        /// The only cuBLAS library names ILGPU 1.5.3 contains, read out of the assembly's own bytes on
        /// 2026-08-22: <c>cublas64_10</c>, <c>cublas64_11</c> and <c>cublas64_12</c>, two occurrences
        /// each. <c>cublas64_13</c> and <c>V13</c> appear zero times.
        /// <para>
        /// <b>This list is the handover risk of the whole probe.</b> A machine carrying only a CUDA 13
        /// redistributable is expected to satisfy none of these, which skips X1 AND X2 - the primary arm
        /// among them - and returns a report that looks entirely normal apart from one NOT MEASURED
        /// line. Naming the three in the failure text turns that silence into a statement the reader can
        /// act on. It is used to REPORT, never to decide: ILGPU picks its own name and this array only
        /// says what was tried.
        /// </para>
        /// </summary>
        private static readonly string[] KnownLibraries = ["cublas64_12", "cublas64_11", "cublas64_10"];

        private readonly CuBlas _blas;
        private readonly Accelerator _accelerator;

        private MemoryBuffer1D<Half, Stride1D.Dense>? _inputHalf;
        private MemoryBuffer1D<Half, Stride1D.Dense>? _weightHalf;
        private MemoryBuffer1D<Half, Stride1D.Dense>? _outputHalf;

        private CuBlasArm(CuBlas blas, Accelerator accelerator)
        {
            _blas = blas;
            _accelerator = accelerator;
        }

        /// <summary>Why the arm is not measured, or null when it is.</summary>
        public static string? Unavailable { get; private set; }

        /// <summary>
        /// Which <c>cublas64_*.dll</c> this process actually loaded, once the arm is running - observed
        /// in the module list rather than assumed from a name in the source. Null until
        /// <see cref="TryCreate"/> succeeds.
        /// <para>
        /// It is in the report because the version is the thing most likely to be wrong on a machine
        /// nobody here can see, and because "cuBLAS worked" and "cuBLAS worked, through 12" are different
        /// facts to somebody reading a result a month later.
        /// </para>
        /// </summary>
        public static string? LoadedLibrary { get; private set; }

        /// <summary>Why the FP16 arm specifically is not measured, when the FP32 one is.</summary>
        public string? Fp16Unavailable { get; private set; }

        /// <summary>True once the FP16 buffers exist and the FP16 arm may be called.</summary>
        public bool Fp16Ready => _inputHalf is not null && _weightHalf is not null && _outputHalf is not null;

        /// <summary>
        /// The raw device pointer ILGPU allocated for the FP16 input, so arm X3 can pass it straight to
        /// <c>cublasGemmEx</c> through its own P/Invoke instead of allocating and uploading a second
        /// copy. A device pointer is scoped to the CUDA CONTEXT, which both arms share, and not to the
        /// cuBLAS module - which is why sharing this is safe while sharing a cuBLAS HANDLE is not.
        /// </summary>
        public nint Fp16InputPointer => _inputHalf!.NativePtr;

        /// <summary>The device pointer for the FP16 weight. See <see cref="Fp16InputPointer"/>.</summary>
        public nint Fp16WeightPointer => _weightHalf!.NativePtr;

        /// <summary>
        /// The device pointer for the FP16 output. Arms X1 and X3 write to the SAME buffer; every read
        /// back happens immediately after the write that produced it, and nothing in the timed phase
        /// depends on its contents.
        /// </summary>
        public nint Fp16OutputPointer => _outputHalf!.NativePtr;

        /// <summary>
        /// Returns null and sets <see cref="Unavailable"/> when cuBLAS cannot be used here. Every failure
        /// mode of a missing redistributable surfaces as a load or type-initialization failure at the
        /// first call, so the whole construction is guarded rather than a probe of the file system.
        /// </summary>
        public static CuBlasArm? TryCreate(Accelerator accelerator)
        {
            if (accelerator is not CudaAccelerator cuda)
            {
                Unavailable = $"the selected accelerator is {accelerator.AcceleratorType}, not CUDA";
                return null;
            }

            try
            {
                var arm = new CuBlasArm(new CuBlas(cuda), accelerator);
                LoadedLibrary = DescribeLoadedLibrary();
                return arm;
            }
            catch (DllNotFoundException ex)
            {
                Unavailable =
                    $"cublas64_*.dll was not found ({ex.Message.Trim()}); install the CUDA 12 " +
                    "redistributable - CUDA 13 does not satisfy this. " + DescribeCandidates();
                return null;
            }
            catch (TypeInitializationException ex)
            {
                Unavailable =
                    $"cuBLAS failed to initialize ({ex.InnerException?.Message.Trim() ?? ex.Message.Trim()}). " +
                    DescribeCandidates();
                return null;
            }
            catch (EntryPointNotFoundException ex)
            {
                Unavailable =
                    $"the installed cuBLAS is a different version ({ex.Message.Trim()}). " + DescribeCandidates();
                return null;
            }
            catch (NotSupportedException ex)
            {
                Unavailable = $"cuBLAS is not supported here ({ex.Message.Trim()}). " + DescribeCandidates();
                return null;
            }
        }

        /// <summary>
        /// Reports which <c>cublas64_*.dll</c> is loaded into this process, by reading the module list.
        /// <para>
        /// An OBSERVATION and not a guess: the name ILGPU used is not exposed by its API, so the only
        /// honest way to state it is to look at what the loader actually mapped. Returns a stated reason
        /// rather than a name when the module list cannot be read, because a missing name and a wrong
        /// name must not look the same in the report.
        /// </para>
        /// </summary>
        private static string DescribeLoadedLibrary()
        {
            try
            {
                using var process = Process.GetCurrentProcess();
                foreach (ProcessModule module in process.Modules)
                {
                    if (module.ModuleName.StartsWith("cublas64_", StringComparison.OrdinalIgnoreCase))
                    {
                        return module.ModuleName;
                    }
                }

                return "loaded, but no cublas64_*.dll is in this process's module list - which should be " +
                       "impossible once cuBLAS has initialized, so read the name as UNKNOWN rather than absent";
            }
            catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException
                                          or PlatformNotSupportedException)
            {
                return $"loaded, but its file name could not be read here ({ex.Message.Trim()})";
            }
        }

        /// <summary>
        /// Names the libraries ILGPU could have used and says which of them this machine can load, so a
        /// skip line says what is missing instead of only that something is.
        /// <para>
        /// <b>What this does and does not establish.</b> It loads by the same names ILGPU contains,
        /// through the ordinary OS search, so a name that fails here is one ILGPU is not going to find
        /// either. It cannot distinguish a library that is absent from one whose own dependencies are
        /// missing, and it says nothing about what a CUDA 13 install is called - this project has never
        /// seen one. The wording below is bounded to what was actually attempted.
        /// </para>
        /// </summary>
        private static string DescribeCandidates()
        {
            var found = new List<string>();
            foreach (var name in KnownLibraries)
            {
                if (NativeLibrary.TryLoad(name, out var handle))
                {
                    found.Add(name);
                    NativeLibrary.Free(handle);
                }
            }

            var tried = string.Join(", ", KnownLibraries);

            if (found.Count == 0)
            {
                return $"None of the cuBLAS names ILGPU 1.5.3 contains ({tried}) could be loaded on this " +
                       "machine, so there is no cuBLAS here that ILGPU knows how to ask for. A CUDA 13 " +
                       "redistributable alone produces exactly this: ILGPU 1.5.3 carries no name for a " +
                       "major 13. Install the CUDA 12 redistributable.";
            }

            return $"Of the cuBLAS names ILGPU 1.5.3 contains ({tried}), this machine can load " +
                   string.Join(", ", found) +
                   " - so the library is present and the failure above is something other than a missing file.";
        }

        /// <summary>
        /// Allocates the FP16 buffers and uploads the operands, converting on the host. Returns false and
        /// sets <see cref="Fp16Unavailable"/> rather than throwing, so the FP32 arms survive a failure.
        /// </summary>
        public bool TryPrepareFp16(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, int n, int k, int m)
        {
            try
            {
                DisposeFp16();

                _inputHalf = _accelerator.Allocate1D<Half>((long)n * k);
                _weightHalf = _accelerator.Allocate1D<Half>((long)m * k);
                _outputHalf = _accelerator.Allocate1D<Half>((long)n * m);

                _inputHalf.CopyFromCPU(ToHalf(input));
                _weightHalf.CopyFromCPU(ToHalf(weight));
                return true;
            }
            catch (Exception ex) when (ex is OutOfMemoryException or CuBlasException or CudaException
                                          or NotSupportedException or InvalidOperationException)
            {
                Fp16Unavailable = $"the FP16 buffers could not be prepared ({ex.Message.Trim()})";
                DisposeFp16();
                return false;
            }
        }

        /// <summary>
        /// Arm X1. <c>output[n,m] = input[n,k] * weight[m,k]^T</c> in FP16, all three row-major. The
        /// operand order is identical to <see cref="Forward"/> and the reasoning is there.
        /// </summary>
        public void ForwardFp16(int n, int k, int m)
        {
            _blas.Gemm(
                CuBlasOperation.Transpose,
                CuBlasOperation.NonTranspose,
                m,
                n,
                k,
                (Half)1.0f,
                _weightHalf!.View,
                k,
                _inputHalf!.View,
                k,
                (Half)0.0f,
                _outputHalf!.View,
                m);
        }

        /// <summary>Reads the FP16 result back, widened to F32 so the parity gate can judge it.</summary>
        public void ReadFp16Output(Span<float> destination)
        {
            var raw = _outputHalf!.GetAsArray1D();
            for (var i = 0; i < destination.Length && i < raw.Length; i++)
            {
                destination[i] = raw[i];
            }
        }

        /// <summary>
        /// Arm X2. <c>output[n,m] = input[n,k] * weight[m,k]^T</c>, all three row-major.
        /// <para>
        /// cuBLAS is column-major, and a row-major buffer of shape (r, c) IS a column-major buffer of
        /// shape (c, r) holding the transpose. So the row-major product above is, read column-major,
        /// <c>C(m,n) = W(m,k) * in(k,n)</c> where the column-major views are <c>W^T(k,m)</c> and
        /// <c>in^T(k,n)</c> — hence transpose on the first operand and none on the second.
        /// </para>
        /// If that reasoning is wrong the parity check fails and no timing is printed, which is exactly
        /// what the oracle is for.
        /// </summary>
        public void Forward(
            ArrayView<float> input,
            ArrayView<float> weight,
            ArrayView<float> output,
            int n,
            int k,
            int m)
        {
            _blas.Gemm(
                CuBlasOperation.Transpose,
                CuBlasOperation.NonTranspose,
                m,
                n,
                k,
                1.0f,
                weight,
                k,
                input,
                k,
                0.0f,
                output,
                m);
        }

        private static Half[] ToHalf(ReadOnlySpan<float> source)
        {
            var result = new Half[source.Length];
            for (var i = 0; i < source.Length; i++)
            {
                result[i] = (Half)source[i];
            }

            return result;
        }

        private void DisposeFp16()
        {
            _inputHalf?.Dispose();
            _weightHalf?.Dispose();
            _outputHalf?.Dispose();
            _inputHalf = null;
            _weightHalf = null;
            _outputHalf = null;
        }

        public void Dispose()
        {
            DisposeFp16();
            _blas.Dispose();
        }
    }
}
