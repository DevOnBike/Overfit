// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// Device buffers and kernel launchers for one cell. The host-to-device weight upload is a separate,
    /// separately reported step (<see cref="UploadWeight"/>) and never sits inside a timed region: a real
    /// fine-tune uploads the frozen base once for the whole run (plan, section 3.5 rule 7).
    /// </summary>
    internal sealed class GpuArms : IDisposable
    {
        private readonly Accelerator _accelerator;
        private readonly int _n;
        private readonly int _k;
        private readonly int _m;

        private readonly MemoryBuffer1D<float, Stride1D.Dense> _input;
        private readonly MemoryBuffer1D<float, Stride1D.Dense> _weight;
        private readonly MemoryBuffer1D<float, Stride1D.Dense> _output;
        private readonly MemoryBuffer1D<float, Stride1D.Dense> _outputGrad;
        private readonly MemoryBuffer1D<float, Stride1D.Dense> _inputGrad;

        private readonly Action<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> _naive;
        private readonly Action<KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> _tiled;
        private readonly Action<KernelConfig, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> _backward;

        public GpuArms(Accelerator accelerator, int n, int k, int m)
        {
            _accelerator = accelerator;
            _n = n;
            _k = k;
            _m = m;

            _input = accelerator.Allocate1D<float>((long)n * k);
            _weight = accelerator.Allocate1D<float>((long)m * k);
            _output = accelerator.Allocate1D<float>((long)n * m);
            _outputGrad = accelerator.Allocate1D<float>((long)n * m);
            _inputGrad = accelerator.Allocate1D<float>((long)n * k);

            _naive = accelerator
                .LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
                    GpuKernels.NaiveForward);
            _tiled = accelerator
                .LoadStreamKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
                    GpuKernels.TiledForward);
            _backward = accelerator
                .LoadStreamKernel<ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
                    GpuKernels.TiledBackward);
        }

        /// <summary>Bytes of F32 weight this cell holds on the device.</summary>
        public long WeightBytes => (long)_m * _k * sizeof(float);

        /// <summary>
        /// Uploads the dequantized F32 weight and returns how long it took, barrier included. Reported
        /// beside the timings, never inside them.
        /// </summary>
        public double UploadWeight(ReadOnlySpan<float> weight)
        {
            var start = Stopwatch.GetTimestamp();
            _weight.View.CopyFromCPU(_accelerator.DefaultStream, weight);
            _accelerator.Synchronize();
            return Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }

        public void UploadInput(ReadOnlySpan<float> input)
        {
            _input.View.CopyFromCPU(_accelerator.DefaultStream, input);
            _accelerator.Synchronize();
        }

        public void UploadOutputGrad(ReadOnlySpan<float> outputGrad)
        {
            _outputGrad.View.CopyFromCPU(_accelerator.DefaultStream, outputGrad);
            _accelerator.Synchronize();
        }

        public ArrayView<float> InputView => _input.View;

        public ArrayView<float> WeightView => _weight.View;

        public ArrayView<float> OutputView => _output.View;

        public void RunNaiveForward() => _naive(new Index2D(_m, _n), _input.View, _weight.View, _output.View, _n, _k, _m);

        public void RunTiledForward() =>
            _tiled(ForwardConfig(), _input.View, _weight.View, _output.View, _n, _k, _m);

        public void RunBackward() =>
            _backward(BackwardConfig(), _outputGrad.View, _weight.View, _inputGrad.View, _n, _k, _m);

        public float[] ReadOutput()
        {
            var host = new float[(long)_n * _m];
            ReadOutput(host);
            return host;
        }

        /// <summary>Reads the device output into a caller-owned buffer. Never inside a timed region.</summary>
        public void ReadOutput(float[] destination)
        {
            _output.View.CopyToCPU(_accelerator.DefaultStream, destination);
            _accelerator.Synchronize();
        }

        public float[] ReadInputGrad()
        {
            var host = new float[(long)_n * _k];
            ReadInputGrad(host);
            return host;
        }

        public void ReadInputGrad(float[] destination)
        {
            _inputGrad.View.CopyToCPU(_accelerator.DefaultStream, destination);
            _accelerator.Synchronize();
        }

        public void Dispose()
        {
            _input.Dispose();
            _weight.Dispose();
            _output.Dispose();
            _outputGrad.Dispose();
            _inputGrad.Dispose();
        }

        private KernelConfig ForwardConfig() => new(
            new Index2D(Blocks(_m), Blocks(_n)),
            new Index2D(GpuKernels.TileSize, GpuKernels.TileSize));

        private KernelConfig BackwardConfig() => new(
            new Index2D(Blocks(_k), Blocks(_n)),
            new Index2D(GpuKernels.TileSize, GpuKernels.TileSize));

        private static int Blocks(int extent) => (extent + GpuKernels.TileSize - 1) / GpuKernels.TileSize;
    }
}
