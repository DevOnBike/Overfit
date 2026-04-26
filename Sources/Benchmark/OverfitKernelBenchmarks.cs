// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using Benchmarks.Helpers;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core; // Zmieniono na Tensors.Core

namespace Benchmarks
{
    [Config(typeof(BenchmarkConfig))]
    public class OverfitKernelBenchmarks
    {
        [Params(64, 256)]
        public int Batch { get; set; }

        [Params(128, 512)]
        public int Hidden { get; set; }

        private ComputationGraph _inferGraph = null!;
        private ComputationGraph _trainGraph = null!;

        // Zmiana z FastTensor na TensorStorage
        private TensorStorage<float> _aTensor = null!;
        private TensorStorage<float> _bTensor = null!;
        private TensorStorage<float> _biasTensor = null!;
        private TensorStorage<float> _targetTensor = null!;
        private TensorStorage<float> _lstmInputTensor = null!;
        private TensorStorage<float> _lstmTargetTensor = null!;

        private AutogradNode _aNode = null!;
        private AutogradNode _bNode = null!;
        private AutogradNode _biasNode = null!;
        private AutogradNode _targetNode = null!;
        private AutogradNode _lstmInputNode = null!;
        private AutogradNode _lstmTargetNode = null!;

        private LinearLayer _linear = null!;
        private ResidualBlock _residual = null!;
        private BatchNorm1D _batchNorm = null!;
        private LstmLayer _lstm = null!;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(42);

            _inferGraph = new ComputationGraph { IsRecording = false };
            _trainGraph = new ComputationGraph { IsRecording = true };

            // Alokacja płaska dla DOD
            _aTensor = new TensorStorage<float>(Batch * Hidden, clearMemory: false);
            _bTensor = new TensorStorage<float>(Hidden * Hidden, clearMemory: false);
            _biasTensor = new TensorStorage<float>(Hidden, clearMemory: false);
            _targetTensor = new TensorStorage<float>(Batch * Hidden, clearMemory: false);

            var seqLen = 10;
            var inputSize = Hidden;
            _lstmInputTensor = new TensorStorage<float>(Batch * seqLen * inputSize, clearMemory: false);
            _lstmTargetTensor = new TensorStorage<float>(Batch * seqLen * Hidden, clearMemory: false);

            Fill(_aTensor.AsSpan(), rnd);
            Fill(_bTensor.AsSpan(), rnd);
            Fill(_biasTensor.AsSpan(), rnd);
            Fill(_targetTensor.AsSpan(), rnd);
            Fill(_lstmInputTensor.AsSpan(), rnd);
            Fill(_lstmTargetTensor.AsSpan(), rnd);

            // Węzły dostają parametry TensorShape
            _aNode = new AutogradNode(_aTensor, new TensorShape(Batch, Hidden), true);
            _bNode = new AutogradNode(_bTensor, new TensorShape(Hidden, Hidden), true);
            _biasNode = new AutogradNode(_biasTensor, new TensorShape(Hidden), true);
            _targetNode = new AutogradNode(_targetTensor, new TensorShape(Batch, Hidden), false);

            _lstmInputNode = new AutogradNode(_lstmInputTensor, new TensorShape(Batch, seqLen, inputSize), true);
            _lstmTargetNode = new AutogradNode(_lstmTargetTensor, new TensorShape(Batch, seqLen, Hidden), false);

            _linear = new LinearLayer(Hidden, Hidden);
            _residual = new ResidualBlock(Hidden);
            _batchNorm = new BatchNorm1D(Hidden);
            _lstm = new LstmLayer(inputSize, Hidden, returnSequences: true);
        }

        [Benchmark]
        public float MatMul_Forward()
        {
            using var y = TensorMath.MatMul(_inferGraph, _aNode, _bNode);
            return y.DataView.AsReadOnlySpan()[0];
        }

        [Benchmark]
        public float MatMul_Backward()
        {
            using var y = TensorMath.MatMul(_trainGraph, _aNode, _bNode);
            using var loss = TensorMath.MSELoss(_trainGraph, y, _targetNode);
            var v = loss.DataView.AsReadOnlySpan()[0];
            _trainGraph.Backward(loss);
            _trainGraph.Reset();
            return v;
        }

        [Benchmark]
        public float LinearLayer_Forward_Infer()
        {
            _inferGraph.IsRecording = false;
            _linear.Eval();

            using var y = _linear.Forward(_inferGraph, _aNode);
            return y.DataView.AsReadOnlySpan()[0];
        }

        [Benchmark]
        public float LinearLayer_MSE_Backward()
        {
            _trainGraph.IsRecording = true;
            _linear.Train();

            using var y = _linear.Forward(_trainGraph, _aNode);
            using var loss = TensorMath.MSELoss(_trainGraph, y, _targetNode);

            var v = loss.DataView.AsReadOnlySpan()[0];
            _trainGraph.Backward(loss);
            _trainGraph.Reset();
            ZeroModuleGradients(_linear);
            return v;
        }

        [Benchmark]
        public float ResidualBlock_Forward_Infer()
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
            _trainGraph.Reset();
            ZeroModuleGradients(_residual);
            return v;
        }

        [Benchmark]
        public float LSTMLayer_Forward_Infer()
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
            _trainGraph.Reset();
            ZeroModuleGradients(_lstm);
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

            _aTensor.Dispose();
            _bTensor.Dispose();
            _biasTensor.Dispose();
            _targetTensor.Dispose();
            _lstmInputTensor.Dispose();
            _lstmTargetTensor.Dispose();
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
                p.ZeroGrad();
            }
        }
    }
}