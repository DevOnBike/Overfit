// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Arm X3. <c>cublasGemmEx</c> with <c>CUBLAS_COMPUTE_32F</c> - FP16 storage, FP32 accumulate,
    /// which is the path a TENSOR CORE actually takes.
    /// <para>
    /// <b>Why this arm exists.</b> <c>ILGPU.Algorithms 1.5.3</c> exports no <c>GemmEx</c>, so the only
    /// FP16 route arm X1 can reach is <c>cublasHgemm</c>, which accumulates in FP16. That is neither the
    /// fastest nor the most accurate FP16 path the card offers. Without X3 the probe measures a floor
    /// and cannot reach the ceiling the whole purchase decision rests on.
    /// </para>
    /// <para>
    /// <b>X3 OWNS ITS CUBLAS HANDLE AND BORROWS EVERYTHING ELSE.</b> <c>CuBlas&lt;T&gt;.Handle</c> is a
    /// public <c>IntPtr</c> and is deliberately not used: passing a handle created inside one loaded copy
    /// of cuBLAS to a function resolved in another copy is undefined, and nothing in the type system
    /// stops it. Owning the handle costs three declarations and removes the whole class of defect rather
    /// than making it unlikely. There is a second reason - cuBLAS defaults to
    /// <c>CUBLAS_POINTER_MODE_HOST</c>, and ILGPU's wrapper manipulates pointer mode, so a borrowed
    /// handle would make this arm's correctness depend on ILGPU's internal state at the moment of the
    /// call. Device pointers, the stream and the CUDA context ARE borrowed, because those are scoped to
    /// the context, which is shared, and not to the library module.
    /// </para>
    /// <para>
    /// <b>What catches the one mistake that would matter.</b> Passing <c>CUBLAS_COMPUTE_16F</c> (64)
    /// instead of <c>CUBLAS_COMPUTE_32F</c> (68) would give FP16 accumulate and a fast, plausible, wrong
    /// number. It cannot pass the oracle: FP32 accumulate measures a relative L2 of 2.9e-4 on the host
    /// and FP16 accumulate measures 6.5e-3 at k=2048 - 6.5x over the
    /// <see cref="ParityResult.Fp16Fp32AccumulateCeiling"/> of 1e-3 - and the probe prints no timing for
    /// an arm that fails parity. The constants themselves were verified against NVIDIA's own
    /// <c>cublas_api.h</c>; see <see cref="CublasNative"/>.
    /// </para>
    /// <para>
    /// <b>NOT ONE LINE OF THIS ARM HAS EVER EXECUTED AGAINST CUBLAS.</b> The development machine has no
    /// NVIDIA device and no CUDA installation, so <see cref="TryCreate"/> returns null here. Everything
    /// below is compile-checked; the branches that were exercised at all were forced by temporary source
    /// mutation and are named in the task report, not claimed here.
    /// </para>
    /// </summary>
    internal sealed class CublasGemmExArm : IDisposable
    {
        // alpha and beta, as fields rather than literals so their addresses are stable and the marshaller
        // has nothing to construct. Nothing in the timed path allocates.
        private static readonly float One = 1.0f;
        private static readonly float Zero = 0.0f;

        private readonly nint _handle;
        private readonly nint _input;
        private readonly nint _weight;
        private readonly nint _output;
        private readonly Fp16Operands? _owned;
        private readonly CuBlasArm? _borrowed;

        private CublasGemmExArm(
            nint handle,
            nint input,
            nint weight,
            nint output,
            Fp16Operands? owned,
            CuBlasArm? borrowed)
        {
            _handle = handle;
            _input = input;
            _weight = weight;
            _output = output;
            _owned = owned;
            _borrowed = borrowed;
        }

        /// <summary>Why arm X3 is not measured, or null when it is.</summary>
        public static string? Unavailable { get; private set; }

        /// <summary>
        /// Which cuBLAS library X3's own resolver loaded, or null. <b>This can differ from the one ILGPU
        /// loaded, and the difference is the point:</b> ILGPU 1.5.3 contains no name for a major 13, so
        /// on a CUDA-13-only machine this is the only cuBLAS in the report.
        /// </summary>
        public static string? ResolvedLibrary { get; private set; }

        /// <summary>
        /// The version <c>cublasGetVersion_v2</c> reported, or null when the call failed. A diagnostic:
        /// X3 owns its handle, so a disagreement with ILGPU's reported version does not invalidate this
        /// arm - it means X1 and X3 measured two different libraries and must not be compared.
        /// </summary>
        public static string? Version { get; private set; }

        /// <summary>
        /// True when the FP16 operands came from <see cref="CuBlasArm"/> rather than being allocated
        /// here. Recorded because "no second upload happened" is a claim about VRAM and about what the
        /// two FP16 arms actually shared.
        /// </summary>
        public bool BorrowedOperands => _borrowed is not null;

        /// <summary>The status of the most recent <c>cublasGemmEx</c> call.</summary>
        public int LastStatus { get; private set; }

        /// <summary>
        /// True when every dimension is a multiple of 8, which is a PRECONDITION for tensor-core use and
        /// not evidence of it. The default batches are 16, 128 and 256, but <c>--batches</c> can be given
        /// anything, so this is checked rather than asserted in prose.
        /// </summary>
        public static bool ShapeAllowsTensorCores(int n, int k, int m) =>
            n % 8 == 0 && k % 8 == 0 && m % 8 == 0;

        /// <summary>
        /// Builds the arm, or returns null and sets <see cref="Unavailable"/> with a stated reason. It
        /// never throws.
        /// <para>
        /// <b>The load is forced here rather than left to the first call.</b> A missing cuBLAS surfaces
        /// lazily, as a <see cref="DllNotFoundException"/> at the first P/Invoke - measured under both
        /// Native-AOT and the JIT - so constructing the arm would otherwise prove nothing and the failure
        /// would land in the middle of a measurement.
        /// </para>
        /// </summary>
        /// <param name="sharedFp16">
        /// The FP16 buffers arm X1 already allocated and uploaded, when it exists. X3 borrows them so
        /// there is no second allocation and no second upload. When it is null - the CUDA 13 case, where
        /// ILGPU's cuBLAS wrapper found nothing to load - X3 allocates its own, because otherwise the one
        /// advantage this route has over X1 would exist only on paper.
        /// </param>
        public static CublasGemmExArm? TryCreate(
            Accelerator accelerator,
            CuBlasArm? sharedFp16,
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weight,
            int n,
            int k,
            int m)
        {
            if (accelerator is not CudaAccelerator cuda)
            {
                Unavailable = $"X3: the selected accelerator is {accelerator.AcceleratorType}, not CUDA.";
                return null;
            }

            if (!CublasNative.TryLoad(out var loadReason))
            {
                Unavailable = loadReason;
                return null;
            }

            ResolvedLibrary = CublasNative.ResolvedName;

            var handle = nint.Zero;
            try
            {
                var status = CublasNative.Create(out handle);
                if (status != CublasNative.StatusSuccess)
                {
                    Unavailable = $"X3: cublasCreate_v2 returned {CublasNative.StatusName(status)} " +
                                  $"({status}) from {ResolvedLibrary}.";
                    return null;
                }

                // The CUDA context ILGPU created must be current on this thread before cuBLAS is asked to
                // do anything with pointers from it. The probe is single-threaded - ArmRunner interleaves
                // on the calling thread and starts no task - so binding once here holds for every later
                // call, and nothing has to happen inside a timed region.
                cuda.Bind();

                // ILGPU's default stream, so the accelerator.Synchronize() barrier that Arm.Gpu puts
                // inside the clock really does wait for this work. Without it cuBLAS would run on the
                // legacy null stream and the timing would still be correct only by accident.
                if (cuda.DefaultStream is CudaStream stream)
                {
                    var streamStatus = CublasNative.SetStream(handle, stream.StreamPtr);
                    if (streamStatus != CublasNative.StatusSuccess)
                    {
                        Unavailable = $"X3: cublasSetStream_v2 returned " +
                                      $"{CublasNative.StatusName(streamStatus)} ({streamStatus}). Without a " +
                                      "known stream the synchronise that ends the timed region cannot be " +
                                      "shown to cover this work, so no timing is taken.";
                        CublasNative.Destroy(handle);
                        return null;
                    }
                }

                Version = CublasNative.GetVersion(handle, out var version) == CublasNative.StatusSuccess
                    ? CublasNative.DescribeVersion(version)
                    : null;
            }
            catch (DllNotFoundException ex)
            {
                Unavailable = $"X3: {ResolvedLibrary} loaded and then could not be resolved for a call, " +
                              "which should not happen. The operating system's text, in the machine's own " +
                              $"language, which may not be English: \"{ex.Message.Trim()}\".";
                return null;
            }
            catch (EntryPointNotFoundException ex)
            {
                Unavailable = $"X3: {ResolvedLibrary} is missing an entry point this arm needs, so it is a " +
                              "cuBLAS older than the one the ABI was transcribed from. The operating " +
                              "system's text, in the machine's own language, which may not be English: " +
                              $"\"{ex.Message.Trim()}\".";
                return null;
            }
            catch (BadImageFormatException ex)
            {
                Unavailable = $"X3: {ResolvedLibrary} was found but cannot be called from this process " +
                              $"(\"{ex.Message.Trim()}\").";
                return null;
            }

            if (sharedFp16 is not null && sharedFp16.Fp16Ready)
            {
                return new CublasGemmExArm(
                    handle,
                    sharedFp16.Fp16InputPointer,
                    sharedFp16.Fp16WeightPointer,
                    sharedFp16.Fp16OutputPointer,
                    owned: null,
                    borrowed: sharedFp16);
            }

            var operands = Fp16Operands.TryPrepare(accelerator, input, weight, n, k, m, out var failure);
            if (operands is null)
            {
                Unavailable = "X3: " + failure;
                CublasNative.Destroy(handle);
                return null;
            }

            return new CublasGemmExArm(
                handle,
                operands.InputPointer,
                operands.WeightPointer,
                operands.OutputPointer,
                operands,
                borrowed: null);
        }

        /// <summary>
        /// Runs the GEMM once and reports whether cuBLAS accepted it. <b>Call this before the arm is
        /// timed.</b> It is where an absent <c>cublasGemmEx</c> entry point surfaces, so
        /// <see cref="Forward"/> can stay a bare call with no exception handling inside the clock.
        /// </summary>
        public bool TryForward(int n, int k, int m, out string failure)
        {
            try
            {
                Forward(n, k, m);
            }
            catch (EntryPointNotFoundException ex)
            {
                failure = $"cublasGemmEx is not exported by {ResolvedLibrary}" +
                          (Version is null ? string.Empty : $" (version {Version})") +
                          $". The operating system's text, in the machine's own language, which may not be " +
                          $"English: \"{ex.Message.Trim()}\".";
                return false;
            }

            if (LastStatus != CublasNative.StatusSuccess)
            {
                failure = $"cublasGemmEx returned {CublasNative.StatusName(LastStatus)} ({LastStatus}) " +
                          $"from {ResolvedLibrary} at n={n} k={k} m={m}.";
                return false;
            }

            failure = string.Empty;
            return true;
        }

        /// <summary>
        /// The timed body. <c>output[n,m] = input[n,k] * weight[m,k]^T</c>, all three row-major and all
        /// three FP16, accumulated in FP32.
        /// <para>
        /// cuBLAS is column-major, and a row-major buffer of shape (r, c) IS a column-major buffer of
        /// shape (c, r) holding the transpose. So the row-major product above is, read column-major,
        /// <c>C(m,n) = W(m,k) * in(k,n)</c> where the column-major views are <c>W^T(k,m)</c> and
        /// <c>in^T(k,n)</c> - hence transpose on the first operand and none on the second. This is the
        /// same mapping <see cref="CuBlasArm.ForwardFp16"/> uses, deliberately, so that X1 and X3 differ
        /// in exactly one thing: the accumulate precision.
        /// </para>
        /// <para>
        /// Allocates nothing. <c>alpha</c> and <c>beta</c> are static readonly fields whose addresses the
        /// marshaller takes directly, and the status write is one field store.
        /// </para>
        /// </summary>
        public void Forward(int n, int k, int m)
        {
            LastStatus = CublasNative.GemmEx(
                _handle,
                CublasNative.OpTranspose,
                CublasNative.OpNonTranspose,
                m,
                n,
                k,
                in One,
                _weight,
                CublasNative.DataTypeR16F,
                k,
                _input,
                CublasNative.DataTypeR16F,
                k,
                in Zero,
                _output,
                CublasNative.DataTypeR16F,
                m,
                CublasNative.Compute32F,
                CublasNative.GemmDefault);
        }

        /// <summary>
        /// Reads the FP16 result back, widened to F32 so the parity gate can judge it. The read goes
        /// through whichever object owns the buffer, so a borrowed run reads exactly the buffer arm X1
        /// wrote to.
        /// </summary>
        public void ReadOutput(Span<float> destination)
        {
            if (_owned is not null)
            {
                _owned.ReadOutput(destination);
                return;
            }

            _borrowed!.ReadFp16Output(destination);
        }

        public void Dispose()
        {
            if (_handle != nint.Zero)
            {
                CublasNative.Destroy(_handle);
            }

            _owned?.Dispose();
        }
    }
}
