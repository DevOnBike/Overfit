// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Benchmarks
{
    /// <summary>
    ///     Baseline performance comparison for a single inference pass.
    ///     Includes a "True Zero-Allocation" ONNX test to ensure the fairest possible
    ///     comparison against Overfit's SIMD-optimized zero-allocation engine.
    /// </summary>
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    [DisassemblyDiagnoser(maxDepth: 2)]
    // Dodajemy analizę cache'u procesora – to pokaże, dlaczego Overfit dominuje
    [HardwareCounters(HardwareCounter.InstructionRetired, HardwareCounter.CacheMisses)]
    public class SingleInferenceBenchmark
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        // Wspólne dane wejściowe
        private float[] _inputData;
        private float[] _onnxRawOutputData;

        // ================== ONNX ==================
        private InferenceSession _onnxSession;

        // Pola dla standardowego ONNX (alokującego)
        private NamedOnnxValue[] _onnxInputs;

        // Pola dla ONNX "True Zero Alloc"
        private RunOptions _onnxRunOptions;
        private OrtValue _onnxPreAllocatedInput;
        private OrtValue _onnxPreAllocatedOutput;
        private string[] _inputNames;
        private string[] _outputNames;
        private OrtValue[] _ortInputValues;
        private OrtValue[] _ortOutputValues;

        // ================== OVERFIT ==================
        private Sequential _overfitModel;
        private FastTensor<float> _overfitInputTensor;
        private AutogradNode _inputNode;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            _inputData = Enumerable.Range(0, InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();
            _onnxRawOutputData = new float[OutputSize];

            // --------------------------------------------------------
            // 1. Setup ONNX Runtime session
            // --------------------------------------------------------
            _onnxSession = new InferenceSession("benchmark_model.onnx");

            // Standard ONNX setup
            var tensor = new DenseTensor<float>(_inputData, [1, InputSize]);
            _onnxInputs = [NamedOnnxValue.CreateFromTensor("input", tensor)];

            // True Zero-Alloc ONNX setup (Najbardziej zoptymalizowana ścieżka C++ interop)
            _onnxRunOptions = new RunOptions();
            var memoryInfo = OrtMemoryInfo.DefaultInstance;

            // Mapowanie pamięci C# bezpośrednio do C++ bez kopiowania
            _onnxPreAllocatedInput = OrtValue.CreateTensorValueFromMemory<float>(memoryInfo, _inputData.AsMemory(), new long[] { 1, InputSize });
            _onnxPreAllocatedOutput = OrtValue.CreateTensorValueFromMemory<float>(memoryInfo, _onnxRawOutputData.AsMemory(), new long[] { 1, OutputSize });

            _inputNames = new[] { "input" };
            // UWAGA: Upewnij się, że "output" to prawidłowa nazwa węzła wyjściowego w Twoim modelu ONNX
            _outputNames = new[] { "output" };

            _ortInputValues = new[] { _onnxPreAllocatedInput };
            _ortOutputValues = new[] { _onnxPreAllocatedOutput };

            // --------------------------------------------------------
            // 2. Setup Overfit model
            // --------------------------------------------------------
            _overfitModel = new Sequential(new LinearLayer(InputSize, OutputSize));
            _overfitModel.Load("benchmark_model.bin");
            _overfitModel.Eval();

            _overfitInputTensor = new FastTensor<float>(1, InputSize, clearMemory: false);
            _inputData.AsSpan().CopyTo(_overfitInputTensor.GetView().AsSpan());
            _inputNode = new AutogradNode(_overfitInputTensor, false);

            // WARMUP
            for (var i = 0; i < 100; i++)
            {
                _overfitModel.Forward(null, _inputNode);
                using var r = _onnxSession.Run(_onnxInputs);
                _onnxSession.Run(_onnxRunOptions, _inputNames, _ortInputValues, _outputNames, _ortOutputValues);
            }
        }

        /// <summary>
        ///     Klasyczne wywołanie ONNX z alokacją tensora wejściowego w locie (najczęstszy antywzorzec).
        /// </summary>
        [Benchmark]
        public float OnnxRuntime_FullAllocation()
        {
            var tensor = new DenseTensor<float>(_inputData, [1, InputSize]);
            var inputs = new[] { NamedOnnxValue.CreateFromTensor("input", tensor) };
            using var results = _onnxSession.Run(inputs);
            return results.First().AsTensor<float>()[0];
        }

        /// <summary>
        ///     Standardowe wywołanie ONNX (Prealokowane wejście, ale wyjście nadal tworzy IDisposable).
        /// </summary>
        [Benchmark(Baseline = true)]
        public float OnnxRuntime_PreAllocated()
        {
            using var results = _onnxSession.Run(_onnxInputs);
            return results.First().AsTensor<float>()[0];
        }

        /// <summary>
        ///     Wywołanie "HPC" dla ONNX. Brak alokacji pamięci w C#. 
        ///     Mierzymy wyłącznie "podatek P/Invoke" i obliczenia C++.
        /// </summary>
        [Benchmark]
        public float OnnxRuntime_TrueZeroAlloc()
        {
            _onnxSession.Run(_onnxRunOptions, _inputNames, _ortInputValues, _outputNames, _ortOutputValues);

            // Wynik jest zapisywany bezpośrednio przez C++ do naszej tablicy C#
            return _onnxRawOutputData[0];
        }

        /// <summary>
        ///     Twoja architektura. Rejestry CPU (SIMD) + Zero Alloc w czystym C#.
        /// </summary>
        [Benchmark]
        public float Overfit_ZeroAlloc()
        {
            return _overfitModel.Forward(null, _inputNode).DataView.AsReadOnlySpan()[0];
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _onnxPreAllocatedInput?.Dispose();
            _onnxPreAllocatedOutput?.Dispose();
            _onnxRunOptions?.Dispose();
            _onnxSession?.Dispose();
            _overfitInputTensor?.Dispose();
            _inputNode?.Dispose();
            _overfitModel?.Dispose();
        }
    }
}