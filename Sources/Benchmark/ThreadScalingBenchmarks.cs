// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Order;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core; // Dodano namespace DOD

namespace Benchmarks
{
    [SimpleJob(RuntimeMoniker.Net10_0)]
    [Orderer(SummaryOrderPolicy.FastestToSlowest)]
    [MemoryDiagnoser]
    public class ThreadScalingBenchmarks
    {
        [Params(1, 2, 4, 8, 16)]
        public int Threads { get; set; }

        [Params(256)]
        public int Batch { get; set; }

        [Params(512)]
        public int Hidden { get; set; }

        private ComputationGraph _inferGraph = null!;
        private ComputationGraph _trainGraph = null!;

        // Zmiana na TensorStorage
        private TensorStorage<float> _aTensor = null!;
        private TensorStorage<float> _bTensor = null!;
        private TensorStorage<float> _targetTensor = null!;

        private AutogradNode _aNode = null!;
        private AutogradNode _bNode = null!;
        private AutogradNode _targetNode = null!;

        private ResidualBlock _residual = null!;
        private LstmLayer _lstm = null!;

        private TensorStorage<float> _lstmInputTensor = null!;
        private TensorStorage<float> _lstmTargetTensor = null!;
        private AutogradNode _lstmInputNode = null!;
        private AutogradNode _lstmTargetNode = null!;

        private int _oldMinWorker;
        private int _oldMinIo;
        private int _oldMaxWorker;
        private int _oldMaxIo;

        private float _sink;

        [GlobalSetup]
        public void Setup()
        {
            var rnd = new Random(123);

            _inferGraph = new ComputationGraph { IsRecording = false };
            _trainGraph = new ComputationGraph { IsRecording = true };

            // POPRAWKA: Przejście na płaskie alokacje dla DOD
            _aTensor = new TensorStorage<float>(Batch * Hidden, clearMemory: true);
            _bTensor = new TensorStorage<float>(Hidden * Hidden, clearMemory: true);
            _targetTensor = new TensorStorage<float>(Batch * Hidden, clearMemory: true);

            Fill(_aTensor.AsSpan(), rnd);
            Fill(_bTensor.AsSpan(), rnd);
            Fill(_targetTensor.AsSpan(), rnd);

            // Wstrzyknięcie TensorShape
            _aNode = new AutogradNode(_aTensor, new TensorShape(Batch, Hidden), requiresGrad: true);
            _bNode = new AutogradNode(_bTensor, new TensorShape(Hidden, Hidden), requiresGrad: true);
            _targetNode = new AutogradNode(_targetTensor, new TensorShape(Batch, Hidden), requiresGrad: false);

            _residual = new ResidualBlock(Hidden);
            _lstm = new LstmLayer(inputSize: 32, hiddenSize: 32, returnSequences: false);

            _lstmInputTensor = new TensorStorage<float>(Batch * 8 * 32, clearMemory: true);
            _lstmTargetTensor = new TensorStorage<float>(Batch * 32, clearMemory: true);

            Fill(_lstmInputTensor.AsSpan(), rnd);
            Fill(_lstmTargetTensor.AsSpan(), rnd);

            _lstmInputNode = new AutogradNode(_lstmInputTensor, new TensorShape(Batch, 8, 32), requiresGrad: true);
            _lstmTargetNode = new AutogradNode(_lstmTargetTensor, new TensorShape(Batch, 32), requiresGrad: false);

            ThreadPool.GetMinThreads(out _oldMinWorker, out _oldMinIo);
            ThreadPool.GetMaxThreads(out _oldMaxWorker, out _oldMaxIo);

            WarmUp();
        }

        private void WarmUp()
        {
            for (var i = 0; i < 10; i++)
            {
                _inferGraph.Reset();
                _inferGraph.IsRecording = false;

                using (var y = _inferGraph.MatMul(_aNode, _bNode))
                {
                    _sink += y.DataView.AsReadOnlySpan()[0];
                }

                _residual.Eval();
                using (var y = _residual.Forward(_inferGraph, _aNode))
                {
                    _sink += y.DataView.AsReadOnlySpan()[0];
                }

                _lstm.Eval();
                using (var y = _lstm.Forward(_inferGraph, _lstmInputNode))
                {
                    _sink += y.DataView.AsReadOnlySpan()[0];
                }

                _trainGraph.Reset();
                _trainGraph.IsRecording = true;

                using (var y = _trainGraph.MatMul(_aNode, _bNode))
                using (var loss = TensorMath.MSELoss(_trainGraph, y, _targetNode))
                {
                    _sink += loss.DataView.AsReadOnlySpan()[0];
                    _trainGraph.Backward(loss);
                }
            }
        }

        [IterationSetup]
        public void IterationSetup()
        {
            _inferGraph.Reset();
            _trainGraph.Reset();

            _aNode.ZeroGrad();
            _bNode.ZeroGrad();
            _lstmInputNode.ZeroGrad();

            ZeroModuleGradients(_residual);
            ZeroModuleGradients(_lstm);

            _residual.Train();
            _lstm.Train();

            ThreadPool.SetMinThreads(Threads, _oldMinIo);
            ThreadPool.SetMaxThreads(Math.Max(Threads, 1), _oldMaxIo);
        }

        [IterationCleanup]
        public void IterationCleanup()
        {
            ThreadPool.SetMinThreads(_oldMinWorker, _oldMinIo);
            ThreadPool.SetMaxThreads(_oldMaxWorker, _oldMaxIo);
        }

        [Benchmark(Baseline = true)]
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
            ThreadPool.SetMinThreads(_oldMinWorker, _oldMinIo);
            ThreadPool.SetMaxThreads(_oldMaxWorker, _oldMaxIo);

            _aNode.Dispose();
            _bNode.Dispose();
            _targetNode.Dispose();

            _lstmInputNode.Dispose();
            _lstmTargetNode.Dispose();

            _residual.Dispose();
            _lstm.Dispose();

            _aTensor.Dispose();
            _bTensor.Dispose();
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
                if (p.RequiresGrad)
                {
                    p.ZeroGrad();
                }
            }
        }
    }
}