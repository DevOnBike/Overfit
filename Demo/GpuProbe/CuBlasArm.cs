// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

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

        /// <summary>Why the FP16 arm specifically is not measured, when the FP32 one is.</summary>
        public string? Fp16Unavailable { get; private set; }

        /// <summary>True once the FP16 buffers exist and the FP16 arm may be called.</summary>
        public bool Fp16Ready => _inputHalf is not null && _weightHalf is not null && _outputHalf is not null;

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
                return new CuBlasArm(new CuBlas(cuda), accelerator);
            }
            catch (DllNotFoundException ex)
            {
                Unavailable = $"cublas64_*.dll was not found ({ex.Message.Trim()}); install the CUDA redistributable";
                return null;
            }
            catch (TypeInitializationException ex)
            {
                Unavailable = $"cuBLAS failed to initialize ({ex.InnerException?.Message.Trim() ?? ex.Message.Trim()})";
                return null;
            }
            catch (EntryPointNotFoundException ex)
            {
                Unavailable = $"the installed cuBLAS is a different version ({ex.Message.Trim()})";
                return null;
            }
            catch (NotSupportedException ex)
            {
                Unavailable = $"cuBLAS is not supported here ({ex.Message.Trim()})";
                return null;
            }
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
