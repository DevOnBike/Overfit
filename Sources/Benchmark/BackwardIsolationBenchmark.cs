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
    /// <summary>
    /// Diagnostic benchmark for training/backward cost.
    ///
    /// Motivation:
    /// MnistTrainingTests shows that the Small CNN benchmark is now in the target range
    /// (~28-29s for 5 epochs), but the dominant per-epoch cost is still graph.Backward.
    ///
    /// This benchmark isolates backward cost for:
    /// - Conv2D + MSE
    /// - ReLU + MSE
    /// - MaxPool2D + MSE
    /// - Linear hidden + MSE
    /// - Linear out + MSE
    /// - Full Small CNN + SoftmaxCrossEntropy
    ///
    /// It intentionally measures training/autograd paths, not the zero-allocation inference engine.
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class BackwardIsolationBenchmark
    {
        private const int BatchSize = 64;

        private const int ImageH = 28;
        private const int ImageW = 28;

        private const int ConvOutChannels = 8;
        private const int ConvOutH = 26;
        private const int ConvOutW = 26;

        private const int PoolOutH = 13;
        private const int PoolOutW = 13;

        private const int FlattenedSize = ConvOutChannels * PoolOutH * PoolOutW;
        private const int HiddenSize = 64;
        private const int ClassCount = 10;

        private const int ImageInputSize = BatchSize * ImageH * ImageW;
        private const int ConvOutputSize = BatchSize * ConvOutChannels * ConvOutH * ConvOutW;
        private const int PoolOutputSize = BatchSize * FlattenedSize;
        private const int HiddenInputSize = BatchSize * FlattenedSize;
        private const int HiddenOutputSize = BatchSize * HiddenSize;
        private const int ClassOutputSize = BatchSize * ClassCount;

        // Keep enough work per invoke to avoid tiny-iteration noise, but not so much that
        // the full CNN case becomes painful.
        private const int LightOperationsPerInvoke = 1024;
        private const int HeavyOperationsPerInvoke = 128;

        private ComputationGraph _graph = null!;

        private TensorStorage<float> _imageStorage = null!;
        private TensorStorage<float> _convFeatureStorage = null!;
        private TensorStorage<float> _poolFeatureStorage = null!;
        private TensorStorage<float> _hiddenInputStorage = null!;
        private TensorStorage<float> _hiddenStorage = null!;
        private TensorStorage<float> _classTargetStorage = null!;
        private TensorStorage<float> _convTargetStorage = null!;
        private TensorStorage<float> _poolTargetStorage = null!;
        private TensorStorage<float> _hiddenTargetStorage = null!;

        private AutogradNode _imageNode = null!;
        private AutogradNode _convFeatureNode = null!;
        private AutogradNode _poolFeatureNode = null!;
        private AutogradNode _hiddenInputNode = null!;
        private AutogradNode _hiddenNode = null!;
        private AutogradNode _classTargetNode = null!;
        private AutogradNode _convTargetNode = null!;
        private AutogradNode _poolTargetNode = null!;
        private AutogradNode _hiddenTargetNode = null!;

        private ConvLayer _conv1 = null!;
        private LinearLayer _fcHidden = null!;
        private LinearLayer _fcOut = null!;

        private float _sink;

        [GlobalSetup]
        public void Setup()
        {
            var random = new Random(123);

            _graph = new ComputationGraph();

            _imageStorage = new TensorStorage<float>(
                ImageInputSize,
                clearMemory: false);

            _convFeatureStorage = new TensorStorage<float>(
                ConvOutputSize,
                clearMemory: false);

            _poolFeatureStorage = new TensorStorage<float>(
                PoolOutputSize,
                clearMemory: false);

            _hiddenInputStorage = new TensorStorage<float>(
                HiddenInputSize,
                clearMemory: false);

            _hiddenStorage = new TensorStorage<float>(
                HiddenOutputSize,
                clearMemory: false);

            _classTargetStorage = new TensorStorage<float>(
                ClassOutputSize,
                clearMemory: false);

            _convTargetStorage = new TensorStorage<float>(
                ConvOutputSize,
                clearMemory: false);

            _poolTargetStorage = new TensorStorage<float>(
                PoolOutputSize,
                clearMemory: false);

            _hiddenTargetStorage = new TensorStorage<float>(
                HiddenOutputSize,
                clearMemory: false);

            Fill(
                _imageStorage.AsSpan(),
                random);

            Fill(
                _convFeatureStorage.AsSpan(),
                random);

            Fill(
                _poolFeatureStorage.AsSpan(),
                random);

            Fill(
                _hiddenInputStorage.AsSpan(),
                random);

            Fill(
                _hiddenStorage.AsSpan(),
                random);

            FillOneHot(
                _classTargetStorage.AsSpan(),
                BatchSize,
                ClassCount);

            Fill(
                _convTargetStorage.AsSpan(),
                random);

            Fill(
                _poolTargetStorage.AsSpan(),
                random);

            Fill(
                _hiddenTargetStorage.AsSpan(),
                random);

            _imageNode = new AutogradNode(
                _imageStorage,
                new TensorShape(
                    BatchSize,
                    1,
                    ImageH,
                    ImageW),
                requiresGrad: true);

            _convFeatureNode = new AutogradNode(
                _convFeatureStorage,
                new TensorShape(
                    BatchSize,
                    ConvOutChannels,
                    ConvOutH,
                    ConvOutW),
                requiresGrad: true);

            _poolFeatureNode = new AutogradNode(
                _poolFeatureStorage,
                new TensorShape(
                    BatchSize,
                    ConvOutChannels,
                    PoolOutH,
                    PoolOutW),
                requiresGrad: true);

            _hiddenInputNode = new AutogradNode(
                _hiddenInputStorage,
                new TensorShape(
                    BatchSize,
                    FlattenedSize),
                requiresGrad: true);

            _hiddenNode = new AutogradNode(
                _hiddenStorage,
                new TensorShape(
                    BatchSize,
                    HiddenSize),
                requiresGrad: true);

            _classTargetNode = new AutogradNode(
                _classTargetStorage,
                new TensorShape(
                    BatchSize,
                    ClassCount),
                requiresGrad: false);

            _convTargetNode = new AutogradNode(
                _convTargetStorage,
                new TensorShape(
                    BatchSize,
                    ConvOutChannels,
                    ConvOutH,
                    ConvOutW),
                requiresGrad: false);

            _poolTargetNode = new AutogradNode(
                _poolTargetStorage,
                new TensorShape(
                    BatchSize,
                    ConvOutChannels,
                    PoolOutH,
                    PoolOutW),
                requiresGrad: false);

            _hiddenTargetNode = new AutogradNode(
                _hiddenTargetStorage,
                new TensorShape(
                    BatchSize,
                    HiddenSize),
                requiresGrad: false);

            _conv1 = new ConvLayer(
                1,
                ConvOutChannels,
                ImageH,
                ImageW,
                3);

            _fcHidden = new LinearLayer(
                FlattenedSize,
                HiddenSize);

            _fcOut = new LinearLayer(
                HiddenSize,
                ClassCount);

            _conv1.Train();
            _fcHidden.Train();
            _fcOut.Train();

            Warmup();
        }

        [IterationSetup]
        public void IterationSetup()
        {
            _graph.Reset();

            _imageNode.ZeroGrad();
            _convFeatureNode.ZeroGrad();
            _poolFeatureNode.ZeroGrad();
            _hiddenInputNode.ZeroGrad();
            _hiddenNode.ZeroGrad();

            ZeroModuleGradients(_conv1);
            ZeroModuleGradients(_fcHidden);
            ZeroModuleGradients(_fcOut);

            _conv1.Train();
            _fcHidden.Train();
            _fcOut.Train();
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            _conv1.Dispose();
            _fcHidden.Dispose();
            _fcOut.Dispose();

            _imageNode.Dispose();
            _convFeatureNode.Dispose();
            _poolFeatureNode.Dispose();
            _hiddenInputNode.Dispose();
            _hiddenNode.Dispose();
            _classTargetNode.Dispose();
            _convTargetNode.Dispose();
            _poolTargetNode.Dispose();
            _hiddenTargetNode.Dispose();

            _imageStorage.Dispose();
            _convFeatureStorage.Dispose();
            _poolFeatureStorage.Dispose();
            _hiddenInputStorage.Dispose();
            _hiddenStorage.Dispose();
            _classTargetStorage.Dispose();
            _convTargetStorage.Dispose();
            _poolTargetStorage.Dispose();
            _hiddenTargetStorage.Dispose();

            _graph.Dispose();
        }

        [Benchmark(OperationsPerInvoke = HeavyOperationsPerInvoke)]
        public float Conv2D_MSE_Backward()
        {
            var checksum = 0f;

            for (var i = 0; i < HeavyOperationsPerInvoke; i++)
            {
                _graph.Reset();
                _imageNode.ZeroGrad();
                ZeroModuleGradients(_conv1);

                using var y = _conv1.Forward(
                    _graph,
                    _imageNode);

                using var loss = TensorMath.MSELoss(
                    _graph,
                    y,
                    _convTargetNode);

                checksum += loss.DataView.AsReadOnlySpan()[0];

                _graph.Backward(loss);
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = LightOperationsPerInvoke)]
        public float ReLU_MSE_Backward()
        {
            var checksum = 0f;

            for (var i = 0; i < LightOperationsPerInvoke; i++)
            {
                _graph.Reset();
                _convFeatureNode.ZeroGrad();

                using var y = TensorMath.ReLU(
                    _graph,
                    _convFeatureNode);

                using var loss = TensorMath.MSELoss(
                    _graph,
                    y,
                    _convTargetNode);

                checksum += loss.DataView.AsReadOnlySpan()[0];

                _graph.Backward(loss);
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = HeavyOperationsPerInvoke)]
        public float MaxPool2D_MSE_Backward()
        {
            var checksum = 0f;

            for (var i = 0; i < HeavyOperationsPerInvoke; i++)
            {
                _graph.Reset();
                _convFeatureNode.ZeroGrad();

                using var y = TensorMath.MaxPool2D(
                    _graph,
                    _convFeatureNode,
                    ConvOutChannels,
                    ConvOutH,
                    ConvOutW,
                    2);

                using var loss = TensorMath.MSELoss(
                    _graph,
                    y,
                    _poolTargetNode);

                checksum += loss.DataView.AsReadOnlySpan()[0];

                _graph.Backward(loss);
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = LightOperationsPerInvoke)]
        public float LinearHidden_MSE_Backward()
        {
            var checksum = 0f;

            for (var i = 0; i < LightOperationsPerInvoke; i++)
            {
                _graph.Reset();
                _hiddenInputNode.ZeroGrad();
                ZeroModuleGradients(_fcHidden);

                using var y = _fcHidden.Forward(
                    _graph,
                    _hiddenInputNode);

                using var loss = TensorMath.MSELoss(
                    _graph,
                    y,
                    _hiddenTargetNode);

                checksum += loss.DataView.AsReadOnlySpan()[0];

                _graph.Backward(loss);
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = LightOperationsPerInvoke)]
        public float LinearOut_MSE_Backward()
        {
            var checksum = 0f;

            for (var i = 0; i < LightOperationsPerInvoke; i++)
            {
                _graph.Reset();
                _hiddenNode.ZeroGrad();
                ZeroModuleGradients(_fcOut);

                using var y = _fcOut.Forward(
                    _graph,
                    _hiddenNode);

                using var loss = TensorMath.MSELoss(
                    _graph,
                    y,
                    _classTargetNode);

                checksum += loss.DataView.AsReadOnlySpan()[0];

                _graph.Backward(loss);
            }

            return checksum;
        }

        [Benchmark(OperationsPerInvoke = HeavyOperationsPerInvoke)]
        public float SmallCnn_SoftmaxCrossEntropy_Backward()
        {
            var checksum = 0f;

            for (var i = 0; i < HeavyOperationsPerInvoke; i++)
            {
                _graph.Reset();

                _imageNode.ZeroGrad();
                ZeroModuleGradients(_conv1);
                ZeroModuleGradients(_fcHidden);
                ZeroModuleGradients(_fcOut);

                using var h1 = _conv1.Forward(
                    _graph,
                    _imageNode);

                using var a1 = TensorMath.ReLU(
                    _graph,
                    h1);

                using var p1 = TensorMath.MaxPool2D(
                    _graph,
                    a1,
                    ConvOutChannels,
                    ConvOutH,
                    ConvOutW,
                    2);

                using var p1F = TensorMath.Reshape(
                    _graph,
                    p1,
                    BatchSize,
                    FlattenedSize);

                using var hidden = _fcHidden.Forward(
                    _graph,
                    p1F);

                using var hiddenAct = TensorMath.ReLU(
                    _graph,
                    hidden);

                using var logits = _fcOut.Forward(
                    _graph,
                    hiddenAct);

                using var loss = TensorMath.SoftmaxCrossEntropy(
                    _graph,
                    logits,
                    _classTargetNode);

                checksum += loss.DataView.AsReadOnlySpan()[0];

                _graph.Backward(loss);
            }

            return checksum;
        }

        private void Warmup()
        {
            for (var i = 0; i < 3; i++)
            {
                _sink += Conv2D_MSE_Backward();
                _sink += ReLU_MSE_Backward();
                _sink += MaxPool2D_MSE_Backward();
                _sink += LinearHidden_MSE_Backward();
                _sink += LinearOut_MSE_Backward();
                _sink += SmallCnn_SoftmaxCrossEntropy_Backward();
            }
        }

        private static void Fill(
            Span<float> span,
            Random random)
        {
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = (float)(random.NextDouble() * 2.0 - 1.0);
            }
        }

        private static void FillOneHot(
            Span<float> span,
            int batchSize,
            int classCount)
        {
            span.Clear();

            for (var b = 0; b < batchSize; b++)
            {
                var label = b % classCount;
                span[(b * classCount) + label] = 1f;
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