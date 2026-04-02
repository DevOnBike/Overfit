using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit
{
    public sealed class MnistPredictor : IDisposable
    {
        private readonly ConvLayer _conv1;
        private readonly BatchNorm1D _bn1;
        private readonly ResidualBlock _res1;
        private readonly LinearLayer _fcOut;

        public MnistPredictor(string modelPath)
        {
            // 1. Definicja architektury (Zgodna z MnistTrainingTests.Mnist_FullTrain60k_ResNetBeastMode)
            _conv1 = new ConvLayer(1, 8, 28, 28, 3);
            _bn1 = new BatchNorm1D(8 * 26 * 26);
            _res1 = new ResidualBlock(1352); // Po MaxPool: 8 * 13 * 13 = 1352
            _fcOut = new LinearLayer(8, 10); // Wynik z GAP to tylko 8 wartości

            using (var fs = new FileStream(modelPath, FileMode.Open))
            using (var br = new BinaryReader(fs))
            {
                _conv1.Load(br);
                _bn1.Load(br);
                _res1.Load(br);
                _fcOut.Load(br);
            }

            // Ustawienie trybu inferencji
            _conv1.Eval();
            _bn1.Eval();
            _res1.Eval();
            _fcOut.Eval();
        }

        public int Predict(float[] pixelData)
        {
            if (pixelData == null || pixelData.Length != 784)
                throw new ArgumentException("Niepoprawne dane wejściowe.");

            ComputationGraph.Active.IsRecording = false;
            try
            {
                ComputationGraph.Active.Reset();

                using var inputMat = new FastTensor<float>(1, 1, 28, 28);
                pixelData.CopyTo(inputMat.AsSpan());

                using var input = new AutogradNode(inputMat, requiresGrad: false);

                // --- FORWARD PASS (Pełna architektura ResNet) ---
                using var h1 = _conv1.Forward(input);
                using var bn1Out = _bn1.Forward(h1); // BN musi być przed ReLU!
                using var a1 = TensorMath.ReLU(bn1Out);
                using var p1 = TensorMath.MaxPool2D(a1, 8, 26, 26, 2);

                // Spłaszczamy i wchodzimy w Residual Block
                using var p1Flat = TensorMath.Reshape(p1, 1, 1352);
                using var resOut = _res1.Forward(p1Flat);

                // Powrót do 4D dla Global Average Pooling
                using var res4D = TensorMath.Reshape(resOut, 1, 8, 13, 13);
                using var gapOut = TensorMath.GlobalAveragePool2D(res4D, 8, 13, 13);

                // Końcowa warstwa liniowa
                using var output = _fcOut.Forward(gapOut);

                return GetArgMax(output.Data.AsSpan());
            }
            finally { ComputationGraph.Active.IsRecording = true; }
        }

        private int GetArgMax(ReadOnlySpan<float> span)
        {
            var maxIdx = 0;
            for (var j = 1; j < span.Length; j++) if (span[j] > span[maxIdx]) maxIdx = j;
            return maxIdx;
        }

        public void Dispose()
        {
            _conv1?.Dispose();
            _bn1?.Dispose();
            _res1?.Dispose();
            _fcOut?.Dispose();
        }
    }
}