// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Onnx;

namespace Benchmarks
{
    /// <summary>
    /// Benchmarks for AveragePool and DAG (ResNetBlock) inference.
    ///
    /// Fixtures:
    ///   Sources/Benchmark/Helpers/tiny_avgpool.onnx   — Conv→AvgPool→Linear
    ///   Sources/Benchmark/Helpers/resnet_block.onnx   — Conv+BN(folded)+ReLU+skip
    ///   Sources/Benchmark/Helpers/tiny_resnet.onnx    — Linear+skip+Linear (from fixture_resnet.py)
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark --filter "*AvgPool*|*ResNet*|*OnnxGraph*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    [MemoryDiagnoser]
    public class OnnxPoolingAndDagBenchmark
    {
        private const string FixtureDir = "Helpers";

        // AveragePool model: Conv(4→8,k=3,p=1) → AvgPool(k=2,s=2) → FC(8*4*4→10)
        private InferenceEngine _avgPoolEngine = null!;
        private float[]         _avgPoolInput  = null!;
        private float[]         _avgPoolOutput = null!;

        // ResNetBlock DAG: Conv(4,padding=1) + BN + ReLU + skip
        private OnnxGraphModel  _resnetModel   = null!;
        private float[]         _resnetInput   = null!;
        private float[]         _resnetOutput  = null!;

        // TinyResNet DAG: Linear(8→8)+skip+Linear(8→4)
        private OnnxGraphModel  _tinyResnet    = null!;
        private float[]         _tinyInput     = null!;
        private float[]         _tinyOutput    = null!;

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(42);

            // AveragePool
            var avgPoolPath = Path.Combine(FixtureDir, "tiny_avgpool.onnx");
            if (File.Exists(avgPoolPath))
            {
                var avgModel = OnnxImporter.Load(avgPoolPath);
                avgModel.Eval();
                _avgPoolEngine = InferenceEngine.FromSequential(avgModel, 256, 10,
                    new DevOnBike.Overfit.Inference.Contracts.InferenceEngineOptions
                    { WarmupIterations = 256 });
                _avgPoolInput  = new float[256];
                _avgPoolOutput = new float[10];
                Fill(rng, _avgPoolInput);
            }

            // ResNetBlock DAG
            var resnetPath = Path.Combine(FixtureDir, "resnet_block.onnx");
            if (File.Exists(resnetPath))
            {
                _resnetModel  = OnnxGraphImporter.Load(resnetPath, 256, 256);
                _resnetModel.Eval();
                _resnetInput  = new float[256];
                _resnetOutput = new float[256];
                Fill(rng, _resnetInput);

                for (var i = 0; i < 256; i++)
                {
                    _resnetModel.RunInference(_resnetInput, _resnetOutput);
                }
            }

            // TinyResNet
            var tinyPath = Path.Combine(FixtureDir, "tiny_resnet.onnx");
            if (File.Exists(tinyPath))
            {
                _tinyResnet  = OnnxGraphImporter.Load(tinyPath, 8, 4);
                _tinyResnet.Eval();
                _tinyInput   = new float[8];
                _tinyOutput  = new float[4];
                Fill(rng, _tinyInput);

                for (var i = 0; i < 256; i++)
                {
                    _tinyResnet.RunInference(_tinyInput, _tinyOutput);
                }
            }
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _avgPoolEngine?.Dispose();
            _resnetModel?.Dispose();
            _tinyResnet?.Dispose();
        }

        /// <summary>Conv+AveragePool+Linear — 256→10, sequential topology.</summary>
        [Benchmark]
        public void AvgPool_ConvPoolFc_Sequential()
        {
            if (_avgPoolEngine == null)
            {
                return;
            }
            _avgPoolEngine.Run(_avgPoolInput, _avgPoolOutput);
        }

        /// <summary>ResNetBlock — Conv+BN(folded)+ReLU+skip. DAG with Add node.</summary>
        [Benchmark]
        public void ResNetBlock_ConvSkip_DAG()
        {
            if (_resnetModel == null)
            {
                return;
            }
            _resnetModel.RunInference(_resnetInput, _resnetOutput);
        }

        /// <summary>TinyResNet — Linear+skip+Linear. Smallest possible DAG baseline.</summary>
        [Benchmark(Baseline = true)]
        public void TinyResNet_LinearSkip_DAG()
        {
            if (_tinyResnet == null)
            {
                return;
            }
            _tinyResnet.RunInference(_tinyInput, _tinyOutput);
        }

        private static void Fill(Random rng, float[] arr)
        {
            for (var i = 0; i < arr.Length; i++)
            {
                arr[i] = (float)(rng.NextDouble() * 2 - 1);
            }
        }
    }
}
