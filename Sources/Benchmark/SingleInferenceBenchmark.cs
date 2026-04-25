// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core; // Zmieniono na Tensors.Core
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
    public class SingleInferenceBenchmark
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;
        private float[] _inputData;
        private AutogradNode _inputNode;
        private NamedOnnxValue[] _onnxInputs;

        private InferenceSession _onnxSession;
        private Sequential _overfitModel;

        // Zmiana na TensorStorage
        private TensorStorage<float> _overfitInputTensor;

        // TRUE ZERO-ALLOC ONNX FIELDS
        private float[] _onnxRawOutputData;
        private OrtValue _onnxPreAllocatedInput;
        private OrtValue _onnxPreAllocatedOutput;
        private RunOptions _onnxRunOptions;
        private string[] _inputNames;
        private string[] _outputNames;
        private OrtValue[] _ortInputValues;
        private OrtValue[] _ortOutputValues;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);
            _inputData = Enumerable.Range(0, InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();

            // 1. STANDARD ONNX SETUP
            _onnxSession = new InferenceSession("benchmark_model.onnx");
            var tensor = new DenseTensor<float>(_inputData, [1, InputSize]);
            _onnxInputs = [NamedOnnxValue.CreateFromTensor("input", tensor)];

            // 2. OVERFIT SETUP (DOD)
            _overfitModel = new Sequential(new LinearLayer(InputSize, OutputSize));
            _overfitModel.Load("benchmark_model.bin");
            _overfitModel.Eval();

            _overfitInputTensor = new TensorStorage<float>(InputSize, clearMemory: false);
            _inputData.AsSpan().CopyTo(_overfitInputTensor.AsSpan());
            _inputNode = new AutogradNode(_overfitInputTensor, new TensorShape(1, InputSize), false);

            // 3. TRUE ZERO-ALLOC ONNX SETUP
            _onnxRawOutputData = new float[OutputSize];
            var allocator = OrtAllocator.DefaultInstance;

            var inputShape = new long[] { 1, InputSize };
            _onnxPreAllocatedInput = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance,
                _inputData.AsMemory(), inputShape);

            var outputShape = new long[] { 1, OutputSize };
            _onnxPreAllocatedOutput = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance,
                _onnxRawOutputData.AsMemory(), outputShape);

            _onnxRunOptions = new RunOptions();
            _inputNames = ["input"];
            _outputNames = ["output"];
            _ortInputValues = [_onnxPreAllocatedInput];
            _ortOutputValues = [_onnxPreAllocatedOutput];

            // WARMUP
            for (var i = 0; i < 100; i++)
            {
                _overfitModel.Forward(null, _inputNode);
                _onnxSession.Run(_onnxInputs).Dispose();
                _onnxSession.Run(_onnxRunOptions, _inputNames, _ortInputValues, _outputNames, _ortOutputValues);
            }
        }

        /// <summary>
        ///     Standardowe wywołanie ONNX używane w 99% tutoriali .NET 
        ///     (Alokuje tablicę wyników pod spodem i nadal tworzy IDisposable).
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
            _overfitModel?.Dispose();
            _overfitInputTensor?.Dispose();
            _inputNode?.Dispose();
        }
    }
}