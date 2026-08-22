// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

// ILGPU.Half and System.Half are different types and every device buffer here holds ILGPU's. The alias
// is explicit rather than relying on using-order, exactly as CuBlasArm does it.
using Half = ILGPU.Half;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// The three FP16 device buffers a cuBLAS FP16 arm needs, owned independently of any cuBLAS wrapper.
    /// <para>
    /// <b>Why this is a separate type rather than three more fields on <see cref="CuBlasArm"/>.</b>
    /// Allocating and uploading FP16 operands needs ILGPU and a CUDA device; it does NOT need
    /// <c>cublas64_*.dll</c>. <see cref="CuBlasArm"/> owns its FP16 buffers, but it only exists when
    /// ILGPU's cuBLAS wrapper loaded - and ILGPU 1.5.3 knows no name for a CUDA 13 library. On a machine
    /// carrying only a CUDA 13 redistributable, arms X1 and X2 both skip while arm X3, which resolves
    /// its own library, can still run. Without this type X3 would skip with them, and the CUDA 13
    /// advantage the design claims for the P/Invoke route would exist only on paper.
    /// </para>
    /// <para>
    /// So X3 BORROWS <see cref="CuBlasArm"/>'s buffers whenever that arm exists - no second allocation
    /// and no second upload, which is what the signed design asks for - and owns a set of its own only
    /// in the case where nothing else has any.
    /// </para>
    /// <para>
    /// <b>Nothing here has ever executed.</b> There is no NVIDIA device on the development machine, so
    /// <see cref="TryPrepare"/> is compile-checked only.
    /// </para>
    /// </summary>
    internal sealed class Fp16Operands : IDisposable
    {
        private MemoryBuffer1D<Half, Stride1D.Dense>? _input;
        private MemoryBuffer1D<Half, Stride1D.Dense>? _weight;
        private MemoryBuffer1D<Half, Stride1D.Dense>? _output;

        private Fp16Operands(Accelerator accelerator)
        {
            Accelerator = accelerator;
        }

        private Accelerator Accelerator { get; }

        /// <summary>Why the operands could not be prepared, or null.</summary>
        public string? Unavailable { get; private set; }

        /// <summary>True once all three buffers exist and their device pointers may be taken.</summary>
        public bool Ready => _input is not null && _weight is not null && _output is not null;

        /// <summary>The device pointer ILGPU allocated for the FP16 input, borrowed by X3.</summary>
        public nint InputPointer => _input!.NativePtr;

        /// <summary>The device pointer ILGPU allocated for the FP16 weight, borrowed by X3.</summary>
        public nint WeightPointer => _weight!.NativePtr;

        /// <summary>The device pointer ILGPU allocated for the FP16 output, borrowed by X3.</summary>
        public nint OutputPointer => _output!.NativePtr;

        /// <summary>
        /// Allocates the three buffers and uploads the operands, converting to FP16 on the host.
        /// Returns null with the reason on <paramref name="failure"/> rather than throwing, so a failure
        /// here costs one arm instead of the run.
        /// </summary>
        public static Fp16Operands? TryPrepare(
            Accelerator accelerator,
            ReadOnlySpan<float> input,
            ReadOnlySpan<float> weight,
            int n,
            int k,
            int m,
            out string failure)
        {
            var operands = new Fp16Operands(accelerator);
            if (operands.TryPrepare(input, weight, n, k, m))
            {
                failure = string.Empty;
                return operands;
            }

            failure = operands.Unavailable ?? "the FP16 operands could not be prepared, with no reason recorded";
            operands.Dispose();
            return null;
        }

        private bool TryPrepare(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, int n, int k, int m)
        {
            try
            {
                Release();

                _input = Accelerator.Allocate1D<Half>((long)n * k);
                _weight = Accelerator.Allocate1D<Half>((long)m * k);
                _output = Accelerator.Allocate1D<Half>((long)n * m);

                _input.CopyFromCPU(ToHalf(input));
                _weight.CopyFromCPU(ToHalf(weight));
                return true;
            }
            catch (Exception ex) when (ex is OutOfMemoryException or CudaException
                                          or NotSupportedException or InvalidOperationException)
            {
                Unavailable = $"the FP16 operands could not be prepared ({ex.Message.Trim()})";
                Release();
                return false;
            }
        }

        /// <summary>Reads the FP16 output back, widened to F32 so the parity gate can judge it.</summary>
        public void ReadOutput(Span<float> destination)
        {
            var raw = _output!.GetAsArray1D();
            for (var i = 0; i < destination.Length && i < raw.Length; i++)
            {
                destination[i] = raw[i];
            }
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

        private void Release()
        {
            _input?.Dispose();
            _weight?.Dispose();
            _output?.Dispose();
            _input = null;
            _weight = null;
            _output = null;
        }

        public void Dispose() => Release();
    }
}
