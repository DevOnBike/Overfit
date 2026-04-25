// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core; // Dodano namespace DOD
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Benchmarks
{
    /// <summary>
    ///     Analyzes the latency distribution (P50 to Max) and jitter performance.
    ///     This is the most critical benchmark for production Service Level Agreements (SLA).
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    public class TailLatencyBenchmark
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;
        private const int TotalCalls = 100_000;

        private float[] _inputData;
        private AutogradNode _inputNode;
        private NamedOnnxValue[] _onnxInputs;
        private DenseTensor<float> _onnxInputTensor;

        private long[] _onnxLatencies;

        private InferenceSession _onnxSession;

        // Zmiana na TensorStorage
        private TensorStorage<float> _overfitInputTensor;
        private long[] _overfitLatencies;

        private Sequential _overfitModel;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            _inputData = Enumerable.Range(0, InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();

            _onnxSession = new InferenceSession("benchmark_model.onnx");
            _onnxInputTensor = new DenseTensor<float>(_inputData, [1, InputSize]);
            _onnxInputs = [NamedOnnxValue.CreateFromTensor("input", _onnxInputTensor)];

            _overfitModel = new Sequential(new LinearLayer(InputSize, OutputSize));
            _overfitModel.Load("benchmark_model.bin");
            _overfitModel.Eval();

            // POPRAWKA: Przejście na TensorStorage
            _overfitInputTensor = new TensorStorage<float>(InputSize, clearMemory: false);
            _inputData.AsSpan().CopyTo(_overfitInputTensor.AsSpan());
            _inputNode = new AutogradNode(_overfitInputTensor, new TensorShape(1, InputSize), false);

            for (var i = 0; i < 1000; i++)
            {
                using var r = _onnxSession.Run(_onnxInputs);
                _overfitModel.Forward(null, _inputNode);
            }

            GC.Collect(2, GCCollectionMode.Forced, true);
            GC.WaitForPendingFinalizers();

            _onnxLatencies = new long[TotalCalls];
            _overfitLatencies = new long[TotalCalls];
        }

        [Benchmark(Baseline = true)]
        public void OnnxRuntime_LatencyProfile()
        {
            var gc0Before = GC.CollectionCount(0);
            var gc1Before = GC.CollectionCount(1);
            var gc2Before = GC.CollectionCount(2);

            for (var i = 0; i < TotalCalls; i++)
            {
                var sw = ValueStopwatch.StartNew();
                using var results = _onnxSession.Run(_onnxInputs);
                _ = results.First().AsTensor<float>()[0];
                var elapsed = sw.GetElapsedTime();
                _onnxLatencies[i] = (long)elapsed.TotalMilliseconds;
            }

            PrintLatencyReport(
            "ONNX Runtime",
            _onnxLatencies,
            GC.CollectionCount(0) - gc0Before,
            GC.CollectionCount(1) - gc1Before,
            GC.CollectionCount(2) - gc2Before);
        }

        [Benchmark]
        public void Overfit_LatencyProfile()
        {
            var gc0Before = GC.CollectionCount(0);
            var gc1Before = GC.CollectionCount(1);
            var gc2Before = GC.CollectionCount(2);

            for (var i = 0; i < TotalCalls; i++)
            {
                var sw = ValueStopwatch.StartNew();
                _ = _overfitModel.Forward(null, _inputNode).DataView.AsReadOnlySpan()[0];
                var elapsed = sw.GetElapsedTime();
                _overfitLatencies[i] = (long)elapsed.TotalMilliseconds;
            }

            PrintLatencyReport(
            "Overfit",
            _overfitLatencies,
            GC.CollectionCount(0) - gc0Before,
            GC.CollectionCount(1) - gc1Before,
            GC.CollectionCount(2) - gc2Before);
        }

        private static void PrintLatencyReport(string name, long[] latencies, int gc0, int gc1, int gc2)
        {
            var ticksPerUs = Stopwatch.Frequency / 1_000_000.0;
            Array.Sort(latencies);

            var p50 = latencies[(int)(TotalCalls * 0.50)] / ticksPerUs;
            var p90 = latencies[(int)(TotalCalls * 0.90)] / ticksPerUs;
            var p95 = latencies[(int)(TotalCalls * 0.95)] / ticksPerUs;
            var p99 = latencies[(int)(TotalCalls * 0.99)] / ticksPerUs;
            var p999 = latencies[(int)(TotalCalls * 0.999)] / ticksPerUs;
            var max = latencies[TotalCalls - 1] / ticksPerUs;
            var jitter = p999 / Math.Max(p50, 0.01);

            Console.WriteLine();
            Console.WriteLine("  +------------------------------------------------+");
            Console.WriteLine($"  |  {name,-44}  |");
            Console.WriteLine("  +------------------------------------------------+");
            Console.WriteLine($"  |  P50:       {p50,10:F2} us                       |");
            Console.WriteLine($"  |  P90:       {p90,10:F2} us                       |");
            Console.WriteLine($"  |  P95:       {p95,10:F2} us                       |");
            Console.WriteLine($"  |  P99:       {p99,10:F2} us                       |");
            Console.WriteLine($"  |  P99.9:     {p999,10:F2} us                       |");
            Console.WriteLine($"  |  Max:       {max,10:F2} us                       |");
            Console.WriteLine($"  |  Jitter:    {jitter,10:F2}x (P99.9/P50)           |");
            Console.WriteLine($"  |  GC Gen-0:  {gc0,10}                         |");
            Console.WriteLine($"  |  GC Gen-1:  {gc1,10}                         |");
            Console.WriteLine($"  |  GC Gen-2:  {gc2,10}                         |");
            Console.WriteLine("  +------------------------------------------------+");
        }

        public void Cleanup()
        {
            _onnxSession?.Dispose();
            _overfitInputTensor?.Dispose();
            _inputNode?.Dispose();
            _overfitModel?.Dispose();
        }
    }
}