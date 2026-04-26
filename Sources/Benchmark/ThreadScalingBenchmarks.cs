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
using DevOnBike.Overfit.Tensors.Core;

namespace Benchmarks
{
    [Config(typeof(BenchmarkConfig))]
    public class ThreadScalingBenchmarks
    {
        private const int OperationsPerInvoke = 32;

        [Params(1, 2, 4, 8, 16)]
        public int Threads { get; set; }

        [Params(256)]
        public int Batch { get; set; }

        [Params(512)]
        public int Hidden { get; set; }

        private ComputationGraph _inferGraph = null!;
        private ComputationGraph _trainGraph = null!;

        private AutogradNode _aNode = null!;
        private AutogradNode _bNode = null!;
        private AutogradNode _targetNode = null!;

        private ResidualBlock _residual = null!;
        private LstmLayer _lstm = null!;

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

            _inferGraph = new ComputationGraph
            {
                IsRecording = false
            };

            _trainGraph = new ComputationGraph
            {
                IsRecording = true
            };

            var aStorage = new TensorStorage<float>(
                Batch * Hidden,
                clearMemory: true);

            var bStorage = new TensorStorage<float>(
                Hidden * Hidden,
                clearMemory: true);

            var targetStorage = new TensorStorage<float>(
                Batch * Hidden,
                clearMemory: true);

            Fill(aStorage.AsSpan(), rnd);
            Fill(bStorage.AsSpan(), rnd);
            Fill(targetStorage.AsSpan(), rnd);

            _aNode = new AutogradNode(
                aStorage,
                new TensorShape(Batch, Hidden),
                requiresGrad: true);

            _bNode = new AutogradNode(
                bStorage,
                new TensorShape(Hidden, Hidden),
                requiresGrad: true);

            _targetNode = new AutogradNode(
                targetStorage,
                new TensorShape(Batch, Hidden),
                requiresGrad: false);

            _residual = new ResidualBlock(Hidden);

            _lstm = new LstmLayer(
                inputSize: 32,
                hiddenSize: 32,
                returnSequences: false);

            var lstmInputStorage = new TensorStorage<float>(
                Batch * 8 * 32,
                clearMemory: true);

            var lstmTargetStorage = new TensorStorage<float>(
                Batch * 32,
                clearMemory: true);

            Fill(lstmInputStorage.AsSpan(), rnd);
            Fill(lstmTargetStorage.AsSpan(), rnd);

            _lstmInputNode = new AutogradNode(
                lstmInputStorage,
                new TensorShape(Batch, 8, 32),
                requiresGrad: true);

            _lstmTargetNode = new AutogradNode(
                lstmTargetStorage,
                new TensorShape(Batch, 32),
                requiresGrad: false);

            ThreadPool.GetMinThreads(
                out _oldMinWorker,
                out _oldMinIo);

            ThreadPool.GetMaxThreads(
                out _oldMaxWorker,
                out _oldMaxIo);

            WarmUp();
        }

        [IterationSetup]
        public void IterationSetup()
        {
            ResetGraphsAndGradients();

            _residual.Train();
            _lstm.Train();

            var workerThreads = Math.Max(1, Threads);

            ThreadPool.SetMinThreads(
                workerThreads,
                _oldMinIo);

            ThreadPool.SetMaxThreads(
                workerThreads,
                _oldMaxIo);
        }

        [IterationCleanup]
        public void IterationCleanup()
        {
            ThreadPool.SetMinThreads(
                _oldMinWorker,
                _oldMinIo);

            ThreadPool.SetMaxThreads(
                _oldMaxWorker,
                _oldMaxIo);
        }

        [Benchmark(
            Baseline = true,
            OperationsPerInvoke = OperationsPerInvoke)]
        public float MatMul_MSE_Backward()
        {
            var checksum = 0f;

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                PrepareTrainingOperation();

                _trainGraph.IsRecording = true;

                using var y = _trainGraph.MatMul(
                    _aNode,
                    _bNode);

                using var loss = TensorMath.MSELoss(
                    _trainGraph,
                    y,
                    _targetNode);

                var v = loss.DataView.AsReadOnlySpan()[0];

                _trainGraph.Backward(loss);

                checksum += v;
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float ResidualBlock_Forward_InferenceStyle()
        {
            var checksum = 0f;

            _residual.Eval();

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                PrepareInferenceOperation();

                using var y = _residual.Forward(
                    _inferGraph,
                    _aNode);

                checksum += y.DataView.AsReadOnlySpan()[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float ResidualBlock_MSE_Backward()
        {
            var checksum = 0f;

            _residual.Train();

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                PrepareTrainingOperation();

                _trainGraph.IsRecording = true;

                using var y = _residual.Forward(
                    _trainGraph,
                    _aNode);

                using var loss = TensorMath.MSELoss(
                    _trainGraph,
                    y,
                    _targetNode);

                var v = loss.DataView.AsReadOnlySpan()[0];

                _trainGraph.Backward(loss);

                checksum += v;
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float LSTMLayer_Forward_InferenceStyle()
        {
            var checksum = 0f;

            _lstm.Eval();

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                PrepareInferenceOperation();

                using var y = _lstm.Forward(
                    _inferGraph,
                    _lstmInputNode);

                checksum += y.DataView.AsReadOnlySpan()[0];
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = OperationsPerInvoke)]
        public float LSTMLayer_MSE_Backward()
        {
            var checksum = 0f;

            _lstm.Train();

            for (var i = 0; i < OperationsPerInvoke; i++)
            {
                PrepareTrainingOperation();

                _trainGraph.IsRecording = true;

                using var y = _lstm.Forward(
                    _trainGraph,
                    _lstmInputNode);

                using var loss = TensorMath.MSELoss(
                    _trainGraph,
                    y,
                    _lstmTargetNode);

                var v = loss.DataView.AsReadOnlySpan()[0];

                _trainGraph.Backward(loss);

                checksum += v;
            }

            return checksum;
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            ThreadPool.SetMinThreads(
                _oldMinWorker,
                _oldMinIo);

            ThreadPool.SetMaxThreads(
                _oldMaxWorker,
                _oldMaxIo);

            _aNode.Dispose();
            _bNode.Dispose();
            _targetNode.Dispose();

            _lstmInputNode.Dispose();
            _lstmTargetNode.Dispose();

            _residual.Dispose();
            _lstm.Dispose();
        }

        private void WarmUp()
        {
            for (var i = 0; i < 8; i++)
            {
                PrepareInferenceOperation();

                using (var y = _inferGraph.MatMul(
                    _aNode,
                    _bNode))
                {
                    _sink += y.DataView.AsReadOnlySpan()[0];
                }

                _residual.Eval();

                PrepareInferenceOperation();

                using (var y = _residual.Forward(
                    _inferGraph,
                    _aNode))
                {
                    _sink += y.DataView.AsReadOnlySpan()[0];
                }

                _lstm.Eval();

                PrepareInferenceOperation();

                using (var y = _lstm.Forward(
                    _inferGraph,
                    _lstmInputNode))
                {
                    _sink += y.DataView.AsReadOnlySpan()[0];
                }

                PrepareTrainingOperation();

                using (var y = _trainGraph.MatMul(
                           _aNode,
                           _bNode))
                using (var loss = TensorMath.MSELoss(
                           _trainGraph,
                           y,
                           _targetNode))
                {
                    _sink += loss.DataView.AsReadOnlySpan()[0];
                    _trainGraph.Backward(loss);
                }
            }
        }

        private void PrepareInferenceOperation()
        {
            _inferGraph.Reset();
            _inferGraph.IsRecording = false;
        }

        private void PrepareTrainingOperation()
        {
            _trainGraph.Reset();
            _trainGraph.IsRecording = true;

            ZeroGradients();
        }

        private void ResetGraphsAndGradients()
        {
            _inferGraph.Reset();
            _inferGraph.IsRecording = false;

            _trainGraph.Reset();
            _trainGraph.IsRecording = true;

            ZeroGradients();
        }

        private void ZeroGradients()
        {
            _aNode.ZeroGrad();
            _bNode.ZeroGrad();
            _lstmInputNode.ZeroGrad();

            ZeroModuleGradients(_residual);
            ZeroModuleGradients(_lstm);
        }

        private static void Fill(
            Span<float> span,
            Random rnd)
        {
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = (float)(rnd.NextDouble() * 2.0 - 1.0);
            }
        }

        private static void ZeroModuleGradients(
            IModule module)
        {
            foreach (var parameter in module.Parameters())
            {
                if (parameter.RequiresGrad)
                {
                    parameter.ZeroGrad();
                }
            }
        }
    }
}