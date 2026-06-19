// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;
using DevOnBike.Overfit.Maths;

// Added Core

namespace DevOnBike.Overfit.Inference
{
    public sealed class MnistPredictor : IDisposable
    {
        private readonly BatchNorm1D _bn1;
        private readonly ConvLayer _conv1;
        private readonly LinearLayer _fcOut;
        private readonly ResidualBlock _res1;

        public MnistPredictor(string modelPath)
        {
            _conv1 = new ConvLayer(1, 8, 28, 28, 3);
            _bn1 = new BatchNorm1D(8 * 26 * 26);
            _res1 = new ResidualBlock(1352);
            _fcOut = new LinearLayer(8, 10);

            using (var fs = new FileStream(modelPath, FileMode.Open))
            using (var br = new BinaryReader(fs))
            {
                _conv1.Load(br);
                _bn1.Load(br);
                _res1.Load(br);
                _fcOut.Load(br);
            }

            _conv1.Eval();
            _bn1.Eval();
            _res1.Eval();
            _fcOut.Eval();
        }

        public void Dispose()
        {
            _conv1?.Dispose();
            _bn1?.Dispose();
            _res1?.Dispose();
            _fcOut?.Dispose();
        }

        public int Predict(float[] pixelData)
        {
            if (pixelData == null || pixelData.Length != 784)
            {
                throw new ArgumentException("Invalid input data. Expected a 784-element pixel array.");
            }

            // DOD: Allocate memory without knowledge of shape...
            using var inputMat = new TensorStorage<float>(784, clearMemory: false);
            pixelData.CopyTo(inputMat.AsSpan());

            // ...then overlay the appropriate Shape on top of it in the node!
            using var input = new AutogradNode(inputMat, new TensorShape(1, 1, 28, 28));

            using var h1 = _conv1.Forward(null, input);
            using var bn1Out = _bn1.Forward(null, h1);
            using var a1 = TensorMath.ReLU(null, bn1Out);
            using var p1 = TensorMath.MaxPool2D(null, a1, 8, 26, 26, 2);

            using var p1Flat = TensorMath.Reshape(null, p1, 1, 1352);
            using var resOut = _res1.Forward(null, p1Flat);

            using var res4D = TensorMath.Reshape(null, resOut, 1, 8, 13, 13);
            using var gapOut = TensorMath.GlobalAveragePool2D(null, res4D, 8, 13, 13);

            using var output = _fcOut.Forward(null, gapOut);

            return MathUtils.ArgMax(output.DataView.AsReadOnlySpan());
        }
    }
}