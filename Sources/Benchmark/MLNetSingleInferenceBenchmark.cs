// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;

namespace Benchmarks
{
    /// <summary>
    ///     Single-inference latency comparison across the three engines a typical .NET
    ///     developer would actually consider for ML inference: Overfit (pure C# zero-alloc),
    ///     ONNX Runtime (best ONNX can do from C#), and ML.NET (the default Microsoft answer
    ///     for "ML in .NET"). Run on a 3-layer MLP (784 → 256 → 128 → 10) so the comparison
    ///     reflects something more realistic than a single-Linear toy model.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         <b>Why ML.NET appears at all.</b> ML.NET is what most .NET developers reach
    ///         for first when they hear "machine learning in .NET", and it's what most
    ///         Microsoft documentation steers you toward. Under the hood, when fed an ONNX
    ///         file via <see cref="Microsoft.ML.Transforms.Onnx.OnnxScoringEstimator"/>,
    ///         ML.NET delegates to the same ONNX Runtime that we benchmark directly — but
    ///         with substantial wrapper overhead in the
    ///         <see cref="PredictionEngine{TSrc, TDst}"/> hot path: per-call
    ///         <c>IDataView</c> synthesis, schema validation, output materialisation.
    ///     </para>
    ///     <para>
    ///         <b>Two ML.NET configurations are measured</b>:
    ///         <list type="bullet">
    ///             <item>
    ///                 <c>MLNet_PredictionEngine_ReusedInput</c> — the production path,
    ///                 reusing a single <c>OnnxInput</c> instance. Best you can do without
    ///                 leaving the high-level ML.NET API.
    ///             </item>
    ///             <item>
    ///                 <c>MLNet_PredictionEngine_FreshInput</c> — the tutorial path. Most
    ///                 ML.NET sample code allocates a fresh input object every call, and
    ///                 production code accidentally inherits that pattern because it reads
    ///                 naturally. Worth measuring because that's what the field actually
    ///                 deploys.
    ///             </item>
    ///         </list>
    ///     </para>
    ///     <para>
    ///         All three engines score the same model weights, so output values are
    ///         identical modulo floating-point ordering. The benchmark methods return
    ///         <c>output[0]</c> only to defeat dead-code elimination — the full output
    ///         vector is computed in every case.
    ///     </para>
    /// </remarks>
    [Config(typeof(BenchmarkConfig))]
    public class MLNetSingleInferenceBenchmark
    {
        private const int InputSize = 784;
        private const int OutputSize = 10;

        private const string OnnxPath = "benchmark_mlp3.onnx";
        private const string BinPath = "benchmark_mlp3.bin";

        private float[] _inputData;

        // Overfit
        private Sequential _overfitModel;
        private TensorStorage<float> _overfitInputTensor;
        private AutogradNode _inputNode;

        // ONNX Runtime — true zero-alloc path (best ONNX can do)
        private InferenceSession _onnxSession;
        private float[] _onnxOutputData;
        private OrtValue _onnxInputValue;
        private OrtValue _onnxOutputValue;
        private RunOptions _onnxRunOptions;
        private string[] _inputNames;
        private string[] _outputNames;
        private OrtValue[] _ortInputValues;
        private OrtValue[] _ortOutputValues;

        // ML.NET — naive PredictionEngine + reusable-input variant
        private PredictionEngine<OnnxInput, OnnxOutput> _mlNetEngine;
        private OnnxInput _mlNetReusableInput;

        [GlobalSetup]
        public void Setup()
        {
            if (!File.Exists(OnnxPath) || !File.Exists(BinPath))
            {
                throw new InvalidOperationException(
                    $"Missing {OnnxPath} or {BinPath}. Generate the model files by running:\n" +
                    "  pip install torch onnx onnxruntime\n" +
                    "  python prepare-onnx.py");
            }

            var rnd = new Random(42);
            _inputData = Enumerable.Range(0, InputSize).Select(_ => (float)rnd.NextDouble()).ToArray();

            SetupOverfit();
            SetupOnnxRuntime();
            SetupMLNet();

            // Warm up each path so JIT, ONNX session caches, and ML.NET's transformer
            // chain have all stabilised before measurement begins. 200 iterations is
            // overkill for the JIT but cheap; the cost of underwarming would be a
            // misleading first batch of measurements.
            for (var i = 0; i < 200; i++)
            {
                _overfitModel.Forward(null, _inputNode);

                _onnxSession.Run(_onnxRunOptions, _inputNames, _ortInputValues, _outputNames, _ortOutputValues);

                _ = _mlNetEngine.Predict(_mlNetReusableInput);
            }
        }

        // =====================================================================
        // Overfit — pure C# SIMD path. Same Sequential shape as the ONNX export
        // produced by prepare-onnx.py, loading the same .bin weights. Uses graph=null
        // forward (inference path, no autograd tape).
        // =====================================================================

        private void SetupOverfit()
        {
            _overfitModel = new Sequential(
                new LinearLayer(InputSize, 256),
                new ReluActivation(),
                new LinearLayer(256, 128),
                new ReluActivation(),
                new LinearLayer(128, OutputSize));
            _overfitModel.Load(BinPath);
            _overfitModel.Eval();

            _overfitInputTensor = new TensorStorage<float>(InputSize, clearMemory: false);
            _inputData.AsSpan().CopyTo(_overfitInputTensor.AsSpan());
            _inputNode = new AutogradNode(_overfitInputTensor, new TensorShape(1, InputSize), false);
        }

        // =====================================================================
        // ONNX Runtime — true zero-alloc path. Pre-allocated input/output OrtValues
        // wrap pinned managed memory so .Run(...) does no managed-side allocation.
        // This is the fastest ONNX can be from C#.
        // =====================================================================

        private void SetupOnnxRuntime()
        {
            _onnxSession = new InferenceSession(OnnxPath);
            _onnxOutputData = new float[OutputSize];

            var inputShape = new long[] { 1, InputSize };
            _onnxInputValue = OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance, _inputData.AsMemory(), inputShape);

            var outputShape = new long[] { 1, OutputSize };
            _onnxOutputValue = OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance, _onnxOutputData.AsMemory(), outputShape);

            _onnxRunOptions = new RunOptions();
            _inputNames = ["input"];
            _outputNames = ["output"];
            _ortInputValues = [_onnxInputValue];
            _ortOutputValues = [_onnxOutputValue];
        }

        // =====================================================================
        // ML.NET — load the same ONNX file via OnnxScoringEstimator. ML.NET wraps
        // ONNX Runtime under the hood; what we measure is the wrapper cost (DataView
        // synthesis, schema validation, output column materialisation) layered on
        // top of ONNX Runtime's own dispatch.
        //
        // OnnxScoringEstimator.Fit needs an example IDataView even though it doesn't
        // actually train anything — supply an empty one with the right schema.
        // =====================================================================

        private void SetupMLNet()
        {
            var ml = new MLContext(seed: 42);
            var emptyData = ml.Data.LoadFromEnumerable(new List<OnnxInput>());

            var pipeline = ml.Transforms.ApplyOnnxModel(
                modelFile: OnnxPath,
                outputColumnNames: ["output"],
                inputColumnNames: ["input"]);

            var transformer = pipeline.Fit(emptyData);
            _mlNetEngine = ml.Model.CreatePredictionEngine<OnnxInput, OnnxOutput>(transformer);

            // The "reused" variant pins the same OnnxInput across all calls. The "fresh"
            // variant allocates a new one each call — that path doesn't reuse this field.
            _mlNetReusableInput = new OnnxInput { Input = _inputData };
        }

        // =====================================================================
        // Benchmark methods
        // =====================================================================

        /// <summary>
        ///     The fastest ONNX path achievable from C#: pre-allocated input/output OrtValues,
        ///     no managed-side allocation per call. Baseline for the comparison.
        /// </summary>
        [Benchmark(Baseline = true)]
        public float OnnxRuntime_TrueZeroAlloc()
        {
            _onnxSession.Run(_onnxRunOptions, _inputNames, _ortInputValues, _outputNames, _ortOutputValues);
            return _onnxOutputData[0];
        }

        /// <summary>
        ///     Overfit: pure C# SIMD path, no P/Invoke, no managed allocations.
        /// </summary>
        [Benchmark]
        public float Overfit_ZeroAlloc()
        {
            return _overfitModel.Forward(null, _inputNode).DataView.AsReadOnlySpan()[0];
        }

        /// <summary>
        ///     ML.NET PredictionEngine with a stable input instance reused across calls.
        ///     Production-grade ML.NET usage. Output object is allocated on every call
        ///     by design — there's no public API hook to pre-allocate it.
        /// </summary>
        [Benchmark]
        public float MLNet_PredictionEngine_ReusedInput()
        {
            var output = _mlNetEngine.Predict(_mlNetReusableInput);
            return output.Output[0];
        }

        /// <summary>
        ///     ML.NET PredictionEngine in the "as written in tutorials" pattern: fresh
        ///     <see cref="OnnxInput"/> on every call. This is what the majority of
        ///     ML.NET sample code shows and what most production systems accidentally
        ///     replicate because it reads more naturally than the reused variant.
        /// </summary>
        [Benchmark]
        public float MLNet_PredictionEngine_FreshInput()
        {
            var input = new OnnxInput { Input = _inputData };
            var output = _mlNetEngine.Predict(input);
            return output.Output[0];
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _onnxInputValue?.Dispose();
            _onnxOutputValue?.Dispose();
            _onnxRunOptions?.Dispose();
            _onnxSession?.Dispose();
            _overfitModel?.Dispose();
            _overfitInputTensor?.Dispose();
            _inputNode?.Dispose();
            _mlNetEngine?.Dispose();
        }

        // =====================================================================
        // ML.NET schema classes
        //
        // ML.NET binds input/output columns to ONNX tensor names by reflection over
        // public properties. [VectorType] carries the tensor shape; [ColumnName] maps
        // the property to the ONNX column name. Must be top-level (or at least non-
        // nested-private) for the binder to discover them.
        // =====================================================================

        public sealed class OnnxInput
        {
            [VectorType(InputSize)]
            [ColumnName("input")]
            public float[] Input { get; set; }
        }

        public sealed class OnnxOutput
        {
            [VectorType(OutputSize)]
            [ColumnName("output")]
            public float[] Output { get; set; }
        }
    }
}
