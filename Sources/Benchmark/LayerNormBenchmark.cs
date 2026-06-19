// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace Benchmarks
{
    /// <summary>
    /// Perf tracker for <see cref="TensorMath.LayerNorm"/> forward + backward.
    ///
    /// <para>
    /// Two reasons this benchmark exists:
    /// </para>
    /// <list type="number">
    ///   <item>LayerNorm forward/backward is on every transformer layer's hot
    ///         path. Regressions here propagate everywhere (GPT-1/2 training
    ///         + inference). Tracker catches drift over time.</item>
    ///   <item>Backward stackalloc's <c>workerCount × C × 2</c> floats of
    ///         partial accumulators per call. <c>[module: SkipLocalsInit]</c>
    ///         in <c>Sources/Main/Properties/AssemblyInfo.cs</c> elides the
    ///         per-frame zero-init; the explicit <c>.Clear()</c> on each
    ///         partial is the only memset that runs. Track that the win
    ///         doesn't silently regress (e.g. if someone re-adds the
    ///         <c>.locals init</c> flag or doubles the partial-buffer size).</item>
    /// </list>
    ///
    /// <para>
    /// Sizes chosen to match a transformer workload — small (TinyShakespeare-
    /// scale), GPT-1 (dModel=128, dFF=512), and a larger config that crosses
    /// into significant stackalloc territory.
    /// </para>
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class LayerNormBenchmark : IDisposable
    {
        // numRows = batchSize * seqLen (we flatten to [numRows, C] in LN).
        [Params(64, 1024, 4096)]
        public int NumRows
        {
            get; set;
        }

        // Hidden size (last dim normalized).
        [Params(128, 512)]
        public int C
        {
            get; set;
        }

        private TensorStorage<float> _inputStorage = null!;
        private TensorStorage<float> _gammaStorage = null!;
        private TensorStorage<float> _betaStorage = null!;
        private AutogradNode _input = null!;
        private AutogradNode _gamma = null!;
        private AutogradNode _beta = null!;
        private ComputationGraph _graph = null!;
        private bool _disposed;

        [GlobalSetup]
        public void Setup()
        {
            _inputStorage = new TensorStorage<float>(NumRows * C);
            _gammaStorage = new TensorStorage<float>(C);
            _betaStorage = new TensorStorage<float>(C);

            var rng = new Random(42);
            var inputSpan = _inputStorage.AsSpan();
            for (var i = 0; i < inputSpan.Length; i++)
            {
                inputSpan[i] = (float)(rng.NextDouble() * 2 - 1);
            }
            var gammaSpan = _gammaStorage.AsSpan();
            for (var i = 0; i < gammaSpan.Length; i++)
            {
                gammaSpan[i] = 1f + (float)(rng.NextDouble() * 0.1);
            }
            var betaSpan = _betaStorage.AsSpan();
            for (var i = 0; i < betaSpan.Length; i++)
            {
                betaSpan[i] = (float)(rng.NextDouble() * 0.01);
            }

            _input = new AutogradNode(_inputStorage, new TensorShape(NumRows, C), requiresGrad: true);
            _gamma = new AutogradNode(_gammaStorage, new TensorShape(C), requiresGrad: true);
            _beta = new AutogradNode(_betaStorage, new TensorShape(C), requiresGrad: true);

            _graph = new ComputationGraph();
        }

        [Benchmark(Description = "Forward")]
        public float Forward()
        {
            _graph.Reset();
            var output = TensorMath.LayerNorm(_graph, _input, _gamma, _beta);
            var value = output.DataView.AsReadOnlySpan()[0];
            output.Dispose();
            return value;
        }

        /// <summary>
        /// Forward + Backward — the path that stackalloc's the per-worker
        /// dGamma / dBeta partial buffers (size <c>workerCount × C × 2</c> floats).
        /// </summary>
        [Benchmark(Description = "Forward+Backward")]
        public float ForwardBackward()
        {
            _graph.Reset();
            _gamma.GradView.AsSpan().Clear();
            _beta.GradView.AsSpan().Clear();
            _input.GradView.AsSpan().Clear();

            var output = TensorMath.LayerNorm(_graph, _input, _gamma, _beta);
            output.GradView.AsSpan().Fill(0.01f);

            _graph.BackwardFromGrad(output);
            var value = _input.GradView.AsReadOnlySpan()[0];
            output.Dispose();
            return value;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;

            _input.Dispose();
            _gamma.Dispose();
            _beta.Dispose();
            _graph.Dispose();
            _inputStorage.Dispose();
            _gammaStorage.Dispose();
            _betaStorage.Dispose();
        }

        public void Dispose() => Cleanup();
    }
}
