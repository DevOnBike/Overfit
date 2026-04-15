// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.IO;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.UI
{
    public sealed class MnistPredictor : IDisposable
    {
        private readonly BatchNorm1D _bn1;
        private readonly ConvLayer _conv1;
        private readonly LinearLayer _fc1;
        private readonly Sequential _weightsContainer;

        public MnistPredictor(string modelPath)
        {
            // 1. Definicja architektury (Musi być identyczna jak w treningu)
            _conv1 = new ConvLayer(1, 8, 28, 28, 3);
            _bn1 = new BatchNorm1D(1352);
            _fc1 = new LinearLayer(1352, 10);

            _weightsContainer = new Sequential(_conv1, _bn1, _fc1);

            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Brak pliku modelu: {modelPath}");
            }

            using (var fs = new FileStream(modelPath, FileMode.Open))
            using (var br = new BinaryReader(fs))
            {
                _weightsContainer.Load(br);
            }

            // Tryb inferencji - wyłącza Dropout i BatchNorm Momentum
            _weightsContainer.Eval();
        }

        public void Dispose()
        {
            _weightsContainer?.Dispose();
        }

        public int Predict(float[] pixelData)
        {
            if (pixelData == null || pixelData.Length != 784)
            {
                throw new ArgumentException("Niepoprawne dane wejściowe. Oczekiwano 784 pikseli.");
            }

            // POPRAWKA: inputMat.GetView().AsSpan() zamiast inputMat.AsSpan()
            using var inputMat = new FastTensor<float>(1, 1, 28, 28);
            pixelData.CopyTo(inputMat.GetView().AsSpan());

            using var input = new AutogradNode(inputMat, false);

            // --- INFERENCJA (FORWARD PASS) ---
            // WAŻNE: Usuwamy 'using' przy wynikach Forward, bo zwracają one wewnętrzne bufory warstw.
            // Ich zdisposowanie uniemożliwiłoby kolejne wywołania Predict!

            var h1 = _conv1.Forward(null, input);
            using var a1 = TensorMath.ReLU(null, h1);

            using var p1 = TensorMath.MaxPool2D(null, a1, 8, 26, 26, 2);
            using var p1Flat = TensorMath.Reshape(null, p1, 1, 1352);

            var bnOut = _bn1.Forward(null, p1Flat);
            var output = _fc1.Forward(null, bnOut);

            // POPRAWKA: output.DataView.AsReadOnlySpan() zamiast output.Data.AsSpan()
            return GetArgMax(output.DataView.AsReadOnlySpan());
        }

        private int GetArgMax(ReadOnlySpan<float> span)
        {
            var maxIdx = 0;
            for (var i = 1; i < span.Length; i++)
            {
                if (span[i] > span[maxIdx])
                {
                    maxIdx = i;
                }
            }
            return maxIdx;
        }
    }
}