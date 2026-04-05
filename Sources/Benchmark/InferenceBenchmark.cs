using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Benchmarks
{
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    [DisassemblyDiagnoser(maxDepth: 2)]
    public class InferenceBenchmark
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        private float[] _inputData;

        // ONNX
        private InferenceSession _onnxSession;
        private NamedOnnxValue[] _onnxInputs;         // Pre-alokowane
        private DenseTensor<float> _onnxInputTensor;   // Pre-alokowany

        // Overfit
        private Sequential _overfitModel;
        private FastTensor<float> _overfitInputTensor;
        private AutogradNode _inputNode;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            _inputData = Enumerable.Range(0, InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();

            // --- ONNX Setup ---
            _onnxSession = new InferenceSession("benchmark_model.onnx");
            _onnxInputTensor = new DenseTensor<float>(_inputData, [1, InputSize]);
            _onnxInputs = [NamedOnnxValue.CreateFromTensor("input", _onnxInputTensor)];

            // --- Overfit Setup ---
            _overfitModel = new Sequential(new LinearLayer(InputSize, OutputSize));
            _overfitModel.Load("benchmark_model.bin");
            _overfitModel.Eval(); // Transpozycja wag + tryb inferencji

            _overfitInputTensor = new FastTensor<float>(false, 1, InputSize);
            _inputData.AsSpan().CopyTo(_overfitInputTensor.AsSpan());
            _inputNode = new AutogradNode(_overfitInputTensor, requiresGrad: false);

            // Warmup — JIT Tier-1 + PGO
            for (var i = 0; i < 100; i++)
            {
                _overfitModel.Forward(null, _inputNode);
            }
        }

        /// <summary>
        /// ONNX Runtime — realistyczne użycie z pre-alokowanym tensorem wejściowym.
        /// Alokacja wynikowa (Run + GetTensor) jest wewnątrz ONNX i niemożliwa do uniknięcia.
        /// </summary>
        [Benchmark(Baseline = true)]
        public float OnnxRuntime_PreAllocated()
        {
            using var results = _onnxSession.Run(_onnxInputs);
            return results.First().AsTensor<float>()[0];
        }

        /// <summary>
        /// ONNX Runtime — "uczciwe" porównanie z pełnym kosztem tworzenia tensora.
        /// Typowe w produkcji: nowe dane wejściowe per request.
        /// </summary>
        [Benchmark]
        public float OnnxRuntime_FullAllocation()
        {
            var tensor = new DenseTensor<float>(_inputData, [1, InputSize]);
            var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor("input", tensor) };
            using var results = _onnxSession.Run(inputs);
            return results.First().AsTensor<float>()[0];
        }

        /// <summary>
        /// Overfit — zero alokacji, SIMD na pre-transponowanych wagach.
        /// Pełna ścieżka inferencji: dane → wynik, bez pośredników.
        /// </summary>
        [Benchmark]
        public float Overfit_ZeroAlloc()
        {
            var outputNode = _overfitModel.Forward(null, _inputNode);
            return outputNode.Data.AsSpan()[0];
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _onnxSession?.Dispose();
            _overfitInputTensor?.Dispose();
            _inputNode?.Dispose();
            _overfitModel?.Dispose();
        }
    }
}