using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;

namespace Benchmarks
{
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    public class OverfitKernelBenchmarks
    {
        [Params(64, 256)]
        public int Batch { get; set; }

        [Params(128, 512)]
        public int Hidden { get; set; }

        private ComputationGraph _inferGraph = null!;
        private ComputationGraph _trainGraph = null!;

        private FastTensor<float> _aTensor = null!;
        private FastTensor<float> _bTensor = null!;
        private FastTensor<float> _biasTensor = null!;
        private FastTensor<float> _targetTensor = null!;

        private AutogradNode _aNode = null!;
        private AutogradNode _bNode = null!;
        private AutogradNode _biasNode = null!;
        private AutogradNode _targetNode = null!;

        private LinearLayer _linear = null!;
        private ResidualBlock _residual = null!;
        private BatchNorm1D _batchNorm = null!;
        private LSTMLayer _lstm = null!;

        private FastTensor<float> _lstmInputTensor = null!;
        private FastTensor<float> _lstmTargetTensor = null!;
        private AutogradNode _lstmInputNode = null!;
        private AutogradNode _lstmTargetNode = null!;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);

            _inferGraph = new ComputationGraph { IsRecording = false };
            _trainGraph = new ComputationGraph { IsRecording = true };

            _aTensor = new FastTensor<float>(Batch, Hidden, clearMemory: true);
            _bTensor = new FastTensor<float>(Hidden, Hidden, clearMemory: true);
            _biasTensor = new FastTensor<float>(1, Hidden, clearMemory: true);
            _targetTensor = new FastTensor<float>(Batch, Hidden, clearMemory: true);

            Fill(_aTensor.GetView().AsSpan(), rnd);
            Fill(_bTensor.GetView().AsSpan(), rnd);
            Fill(_biasTensor.GetView().AsSpan(), rnd);
            Fill(_targetTensor.GetView().AsSpan(), rnd);

            _aNode = new AutogradNode(_aTensor, requiresGrad: true);
            _bNode = new AutogradNode(_bTensor, requiresGrad: true);
            _biasNode = new AutogradNode(_biasTensor, requiresGrad: true);
            _targetNode = new AutogradNode(_targetTensor, requiresGrad: false);

            _linear = new LinearLayer(Hidden, Hidden);
            _residual = new ResidualBlock(Hidden);
            _batchNorm = new BatchNorm1D(Hidden);
            _lstm = new LSTMLayer(inputSize: 32, hiddenSize: 32, returnSequences: false);

            _lstmInputTensor = new FastTensor<float>(Batch, 8, 32, clearMemory: true);
            _lstmTargetTensor = new FastTensor<float>(Batch, 32, clearMemory: true);

            Fill(_lstmInputTensor.GetView().AsSpan(), rnd);
            Fill(_lstmTargetTensor.GetView().AsSpan(), rnd);

            _lstmInputNode = new AutogradNode(_lstmInputTensor, requiresGrad: true);
            _lstmTargetNode = new AutogradNode(_lstmTargetTensor, requiresGrad: false);

            WarmUp();
        }

        private void WarmUp()
        {
            for (var i = 0; i < 4; i++)
            {
                _inferGraph.Reset();
                _inferGraph.IsRecording = false;

                using var mm = _inferGraph.MatMul(_aNode, _bNode);
                using var ab = _inferGraph.AddBias(mm, _biasNode);
                using var relu = _inferGraph.ReLU(ab);

                _linear.Eval();
                _residual.Eval();
                _batchNorm.Eval();
                _lstm.Eval();

                using var lin = _linear.Forward(_inferGraph, _aNode);
                using var bn = _batchNorm.Forward(_inferGraph, _aNode);
                using var res = _residual.Forward(_inferGraph, _aNode);
                using var lstm = _lstm.Forward(_inferGraph, _lstmInputNode);
            }
        }

        [IterationSetup]
        public void IterationSetup()
        {
            _inferGraph.Reset();
            _trainGraph.Reset();

            _aNode.ZeroGrad();
            _bNode.ZeroGrad();
            _biasNode.ZeroGrad();
            _lstmInputNode.ZeroGrad();

            ZeroModuleGradients(_linear);
            ZeroModuleGradients(_residual);
            ZeroModuleGradients(_batchNorm);
            ZeroModuleGradients(_lstm);
        }

        [Benchmark(Baseline = true)]
        public float MatMul_Forward()
        {
            _inferGraph.IsRecording = false;

            using var y = _inferGraph.MatMul(_aNode, _bNode);
            return y.DataView.AsReadOnlySpan()[0];
        }

        [Benchmark]
        public float AddBias_Forward()
        {
            _inferGraph.IsRecording = false;

            using var y = _inferGraph.AddBias(_aNode, _biasNode);
            return y.DataView.AsReadOnlySpan()[0];
        }

        [Benchmark]
        public float ReLU_Forward()
        {
            _inferGraph.IsRecording = false;

            using var y = _inferGraph.ReLU(_aNode);
            return y.DataView.AsReadOnlySpan()[0];
        }

        [Benchmark]
        public float MatMul_MSE_Backward()
        {
            _trainGraph.IsRecording = true;

            using var y = _trainGraph.MatMul(_aNode, _bNode);
            using var loss = TensorMath.MSELoss(_trainGraph, y, _targetNode);

            var v = loss.DataView.AsReadOnlySpan()[0];
            _trainGraph.Backward(loss);
            return v;
        }

        [Benchmark]
        public float Linear_Forward_InferenceStyle()
        {
            _inferGraph.IsRecording = false;
            _linear.Eval();

            using var y = _linear.Forward(_inferGraph, _aNode);
            return y.DataView.AsReadOnlySpan()[0];
        }

        [Benchmark]
        public float Linear_MSE_Backward()
        {
            _trainGraph.IsRecording = true;
            _linear.Train();

            using var y = _linear.Forward(_trainGraph, _aNode);
            using var loss = TensorMath.MSELoss(_trainGraph, y, _targetNode);

            var v = loss.DataView.AsReadOnlySpan()[0];
            _trainGraph.Backward(loss);
            return v;
        }

        [Benchmark]
        public float BatchNorm1D_Forward_InferenceStyle()
        {
            _inferGraph.IsRecording = false;
            _batchNorm.Eval();

            using var y = _batchNorm.Forward(_inferGraph, _aNode);
            return y.DataView.AsReadOnlySpan()[0];
        }

        [Benchmark]
        public float ResidualBlock_Forward_InferenceStyle()
        {
            _inferGraph.IsRecording = false;
            _residual.Eval();

            using var y = _residual.Forward(_inferGraph, _aNode);
            return y.DataView.AsReadOnlySpan()[0];
        }

        [Benchmark]
        public float ResidualBlock_MSE_Backward()
        {
            _trainGraph.IsRecording = true;
            _residual.Train();

            using var y = _residual.Forward(_trainGraph, _aNode);
            using var loss = TensorMath.MSELoss(_trainGraph, y, _targetNode);

            var v = loss.DataView.AsReadOnlySpan()[0];
            _trainGraph.Backward(loss);
            return v;
        }

        [Benchmark]
        public float LSTMLayer_Forward_InferenceStyle()
        {
            _inferGraph.IsRecording = false;
            _lstm.Eval();

            using var y = _lstm.Forward(_inferGraph, _lstmInputNode);
            return y.DataView.AsReadOnlySpan()[0];
        }

        [Benchmark]
        public float LSTMLayer_MSE_Backward()
        {
            _trainGraph.IsRecording = true;
            _lstm.Train();

            using var y = _lstm.Forward(_trainGraph, _lstmInputNode);
            using var loss = TensorMath.MSELoss(_trainGraph, y, _lstmTargetNode);

            var v = loss.DataView.AsReadOnlySpan()[0];
            _trainGraph.Backward(loss);
            return v;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _aNode.Dispose();
            _bNode.Dispose();
            _biasNode.Dispose();
            _targetNode.Dispose();

            _lstmInputNode.Dispose();
            _lstmTargetNode.Dispose();

            _linear.Dispose();
            _residual.Dispose();
            _batchNorm.Dispose();
            _lstm.Dispose();
        }

        private static void Fill(Span<float> span, Random rnd)
        {
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = (float)(rnd.NextDouble() * 2.0 - 1.0);
            }
        }

        private static void ZeroModuleGradients(IModule module)
        {
            foreach (var p in module.Parameters())
            {
                if (p.RequiresGrad)
                {
                    p.ZeroGrad();
                }
            }
        }
    }
}