using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Inference.Contracts;
using DevOnBike.Overfit.Licensing;
using TorchSharp;
using static TorchSharp.torch;
using OverfitSequential = DevOnBike.Overfit.DeepLearning.Sequential;

namespace Benchmarks
{
    /// <summary>
    /// Overfit vs TorchSharp (LibTorch backend) inference benchmark.
    ///
    /// Model: 3-layer MLP 784→256→128→10.
    ///
    /// TorchSharp benchmark:
    /// - native LibTorch forward
    /// - no tensor-to-managed ToArray() copy
    ///
    /// Overfit benchmark:
    /// - InferenceEngine.Run(...)
    /// - managed output buffer
    /// - expected 0 B hot-path allocation
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class TorchSharpInferenceBenchmark : IDisposable
    {
        private const int InputSize = 784;
        private const int Hidden1 = 256;
        private const int Hidden2 = 128;
        private const int OutputSize = 10;

        private const int OperationsPerInvoke = 16_384;

        private float[] _inputData = null!;
        private float[] _overfitOutput = null!;

        private OverfitSequential _overfitModel = null!;
        private InferenceEngine _overfitEngine = null!;

        private TorchMlp _torchModel = null!;
        private Tensor _torchInput = null!;

        [GlobalSetup]
        public void Setup()
        {
            OverfitLicense.SuppressNotice = true;

            _inputData = new float[InputSize];
            _overfitOutput = new float[OutputSize];

            FillDeterministic(_inputData);

            SetupOverfit();
            SetupTorchSharp();

            // Warmup.
            using var noGrad = no_grad();

            for (var i = 0; i < 512; i++)
            {
                _overfitEngine.Run(_inputData, _overfitOutput);

                using var y = _torchModel.forward(_torchInput);
                GC.KeepAlive(y);
            }
        }

        private void SetupOverfit()
        {
            _overfitModel = new OverfitSequential(
                new LinearLayer(InputSize, Hidden1),
                new ReluActivation(),
                new LinearLayer(Hidden1, Hidden2),
                new ReluActivation(),
                new LinearLayer(Hidden2, OutputSize));

            _overfitModel.Eval();

            _overfitEngine = InferenceEngine.FromSequential(
                _overfitModel,
                inputSize: InputSize,
                outputSize: OutputSize,
                new InferenceEngineOptions
                {
                    WarmupIterations = 32,
                    MaxIntermediateElements = 64 * 1024,
                    ValidateFiniteInput = false,
                    DisposeModelWithEngine = false,
                });

            _overfitEngine.Run(_inputData, _overfitOutput);
        }

        private void SetupTorchSharp()
        {
            _torchInput = tensor(_inputData, dtype: ScalarType.Float32)
                .reshape(1, InputSize);

            _torchModel = new TorchMlp();
            _torchModel.eval();
        }

        [Benchmark(Baseline = true, OperationsPerInvoke = OperationsPerInvoke)]
        public float TorchSharp_3Layer_Mlp_Forward()
        {
            var checksum = 0f;

            using var noGrad = no_grad();

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                using var y = _torchModel.forward(_torchInput);

                // Do not call ToArray(); that would benchmark native -> managed copy.
                GC.KeepAlive(y);

                checksum += 1f;
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Overfit_3Layer_Mlp_InferenceEngine_ZeroAlloc()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _overfitEngine.Run(_inputData, _overfitOutput);
                checksum += _overfitOutput[0];
            }

            return checksum;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _torchInput?.Dispose();
            _torchModel?.Dispose();

            _overfitEngine?.Dispose();
            _overfitModel?.Dispose();
        }

        public void Dispose()
        {
            Cleanup();
        }

        private static void FillDeterministic(float[] data)
        {
            var seed = 0x12345678u;

            for (var i = 0; i < data.Length; i++)
            {
                seed = seed * 1664525u + 1013904223u;
                var normalized = (seed & 0x00FFFFFF) / 16777216f;
                data[i] = normalized * 2f - 1f;
            }
        }

        private sealed class TorchMlp : nn.Module<Tensor, Tensor>
        {
            private readonly nn.Module<Tensor, Tensor> _lin1;
            private readonly nn.Module<Tensor, Tensor> _relu1;
            private readonly nn.Module<Tensor, Tensor> _lin2;
            private readonly nn.Module<Tensor, Tensor> _relu2;
            private readonly nn.Module<Tensor, Tensor> _lin3;

            public TorchMlp()
                : base(nameof(TorchMlp))
            {
                _lin1 = nn.Linear(InputSize, Hidden1);
                _relu1 = nn.ReLU();

                _lin2 = nn.Linear(Hidden1, Hidden2);
                _relu2 = nn.ReLU();

                _lin3 = nn.Linear(Hidden2, OutputSize);

                RegisterComponents();
            }

            public override Tensor forward(Tensor input)
            {
                using var x1 = _lin1.forward(input);
                using var r1 = _relu1.forward(x1);

                using var x2 = _lin2.forward(r1);
                using var r2 = _relu2.forward(x2);

                return _lin3.forward(r2);
            }
        }
    }
}