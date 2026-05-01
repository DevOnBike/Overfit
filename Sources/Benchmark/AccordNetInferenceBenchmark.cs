using Accord.Neuro;
using Accord.Neuro.Learning;
using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Inference;
using DevOnBike.Overfit.Inference.Contracts;
using DevOnBike.Overfit.Licensing;
using OverfitSequential = DevOnBike.Overfit.DeepLearning.Sequential;

namespace Benchmarks
{
    [Config(typeof(BenchmarkConfig))]
    public class AccordNetInferenceBenchmark : IDisposable
    {
        private const int InputSize = 784;
        private const int HiddenSize = 128;
        private const int OutputSize = 10;

        private const int OperationsPerInvoke = 16_384;

        private float[] _inputFloat = null!;
        private double[] _inputDouble = null!;

        private OverfitSequential _overfitModel = null!;
        private InferenceEngine _overfitEngine = null!;
        private float[] _overfitOutput = null!;

        private ActivationNetwork _accordNetwork = null!;

        [GlobalSetup]
        public void Setup()
        {
            OverfitLicense.SuppressNotice = true;

            _inputFloat = new float[InputSize];
            _inputDouble = new double[InputSize];
            _overfitOutput = new float[OutputSize];

            FillDeterministic(_inputFloat);

            for (var i = 0; i < InputSize; i++)
            {
                _inputDouble[i] = _inputFloat[i];
            }

            SetupOverfit();
            SetupAccord();

            for (var i = 0; i < 512; i++)
            {
                _overfitEngine.Run(_inputFloat, _overfitOutput);
                var y = _accordNetwork.Compute(_inputDouble);
                GC.KeepAlive(y);
            }
        }

        private void SetupOverfit()
        {
            _overfitModel = new OverfitSequential(
            new LinearLayer(InputSize, HiddenSize),
            new ReluActivation(),
            new LinearLayer(HiddenSize, OutputSize));

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
                DisposeModelWithEngine = false
            });

            _overfitEngine.Run(_inputFloat, _overfitOutput);
        }

        private void SetupAccord()
        {
            // Accord uses double[] and returns a new double[] from Compute(...).
            // This is intentional: benchmark the real public inference API.
            _accordNetwork = new ActivationNetwork(
            new SigmoidFunction(alpha: 2.0),
            InputSize,
            HiddenSize,
            OutputSize);

            new NguyenWidrow(_accordNetwork).Randomize();
        }

        [Benchmark(Baseline = true, OperationsPerInvoke = OperationsPerInvoke)]
        public double AccordNet_ActivationNetwork_Compute()
        {
            var checksum = 0.0;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                var y = _accordNetwork.Compute(_inputDouble);
                checksum += y[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float Overfit_2Layer_Mlp_InferenceEngine_ZeroAlloc()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                _overfitEngine.Run(_inputFloat, _overfitOutput);
                checksum += _overfitOutput[0];
            }

            return checksum;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
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
    }
}