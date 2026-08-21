// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Arm X1 — the same forward through cuBLAS, as an upper bound on what the card can do in FP32.
    /// Finding F2 of the plan: ILGPU has no tensor-core path, so G1 and G2 measure OUR kernels and the
    /// gap between them and a vendor library is otherwise an assumption. This arm turns it into a number.
    /// <para>
    /// OFF by default and it must SKIP with a stated reason rather than fail. cuBLAS lives in
    /// <c>cublas64_*.dll</c>, which is part of the CUDA <b>toolkit</b>; the core ILGPU path needs only
    /// <c>nvcuda</c> from the display driver. Requiring the toolkit would break the probe's
    /// one-command constraint, so the toolkit is optional and its absence is reported, not fatal.
    /// </para>
    /// </summary>
    internal sealed class CuBlasArm : IDisposable
    {
        private readonly CuBlas _blas;

        private CuBlasArm(CuBlas blas)
        {
            _blas = blas;
        }

        /// <summary>Why the arm is not measured, or null when it is.</summary>
        public static string? Unavailable { get; private set; }

        /// <summary>
        /// Returns null and sets <see cref="Unavailable"/> when cuBLAS cannot be used here. Every failure
        /// mode of a missing toolkit surfaces as a load or type-initialization failure at the first call,
        /// so the whole construction is guarded rather than a probe of the file system.
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
                return new CuBlasArm(new CuBlas(cuda));
            }
            catch (DllNotFoundException ex)
            {
                Unavailable = $"the CUDA toolkit is not installed ({ex.Message.Trim()})";
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
        /// <c>output[n,m] = input[n,k] * weight[m,k]^T</c>, all three row-major.
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

        public void Dispose() => _blas.Dispose();
    }
}
