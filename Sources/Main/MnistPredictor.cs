// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit
{
    /// <summary>
    ///     High-level predictor for MNIST handwritten digit recognition.
    ///     Implements a Convolutional Neural Network (CNN) architecture with Residual Blocks.
    /// </summary>
    public sealed class MnistPredictor : IDisposable
    {
        private readonly BatchNorm1D _bn1;
        private readonly ConvLayer _conv1;
        private readonly LinearLayer _fcOut;
        private readonly ResidualBlock _res1;

        /// <summary>
        ///     Initializes the predictor and loads the model parameters from the specified path.
        /// </summary>
        /// <param name="modelPath">Path to the binary file containing serialized weights.</param>
        public MnistPredictor(string modelPath)
        {
            // Architecture definition: Conv(3x3) -> BN -> ReLU -> MaxPool -> ResNet Block -> GAP -> Linear
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

        /// <summary>
        ///     Predicts the digit (0-9) for the given pixel data.
        /// </summary>
        /// <param name="pixelData">Flattened array of 784 pixels (28x28 normalized 0.0-1.0).</param>
        /// <returns>The predicted digit.</returns>
        public int Predict(float[] pixelData)
        {
            if (pixelData == null || pixelData.Length != 784)
            {
                throw new ArgumentException("Invalid input data. Expected a 784-element pixel array.");
            }

            using var inputMat = new FastTensor<float>(1, 1, 28, 28);
            pixelData.CopyTo(inputMat.AsSpan());

            using var input = new AutogradNode(inputMat, false);

            // --- FORWARD PASS (Inference Mode: null graph) ---
            using var h1 = _conv1.Forward(null, input);
            using var bn1Out = _bn1.Forward(null, h1);
            using var a1 = TensorMath.ReLU(null, bn1Out);
            using var p1 = TensorMath.MaxPool2D(null, a1, 8, 26, 26, 2);

            using var p1Flat = TensorMath.Reshape(null, p1, 1, 1352);
            using var resOut = _res1.Forward(null, p1Flat);

            using var res4D = TensorMath.Reshape(null, resOut, 1, 8, 13, 13);
            using var gapOut = TensorMath.GlobalAveragePool2D(null, res4D, 8, 13, 13);

            using var output = _fcOut.Forward(null, gapOut);

            return GetArgMax(output.Data.AsSpan());
        }

        /// <summary>
        ///     Finds the index of the highest probability value in the output tensor.
        /// </summary>
        private int GetArgMax(ReadOnlySpan<float> span)
        {
            var maxIdx = 0;

            for (var j = 1; j < span.Length; j++)
            {
                if (span[j] > span[maxIdx])
                {
                    maxIdx = j;
                }
            }

            return maxIdx;
        }
    }
}