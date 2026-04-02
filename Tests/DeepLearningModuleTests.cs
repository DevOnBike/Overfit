using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Tests
{
    public class DeepLearningModuleTests : IDisposable
    {
        public DeepLearningModuleTests()
        {
            // Inicjalizacja grafu dla każdego testu
            ComputationGraph.Active = new ComputationGraph();
        }

        public void Dispose() => ComputationGraph.Active = null;

        [Fact]
        public void LinearLayer_ForwardAndSerialization_Correct()
        {
            // Test wymiarów i parametrów - LinearLayer(input, output)
            using var layer = new LinearLayer(10, 5);
            Assert.Equal(2, layer.Parameters().Count());

            // Używamy FastTensor zamiast FastMatrix
            using var inputTensor = new FastTensor<float>(2, 10);
            using var input = new AutogradNode(inputTensor, false);
            using var output = layer.Forward(input);

            // Sprawdzamy wymiary poprzez Shape
            Assert.Equal(2, output.Data.Shape[0]);
            Assert.Equal(5, output.Data.Shape[1]);

            // Test zapisu i odczytu (Serializacja N-wymiarowa)
            using var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, true)) layer.Save(bw);

            ms.Position = 0;
            using var layer2 = new LinearLayer(10, 5);
            using (var br = new BinaryReader(ms)) layer2.Load(br);

            // Weryfikacja wag po odczycie
            Assert.Equal(layer.Weights.Data[0, 0], layer2.Weights.Data[0, 0]);
        }

        [Fact]
        public void Sequential_ChainsModulesAndHandlesModes()
        {
            // Sequential powinien poprawnie propagować tryb Train/Eval
            var l1 = new LinearLayer(5, 5);
            var act = new ReluActivation();
            using var seq = new Sequential(l1, act);

            seq.Eval();
            Assert.False(seq.IsTraining);
            Assert.False(l1.IsTraining);

            seq.Train();
            Assert.True(seq.IsTraining);

            // Test przepływu przez Sequential
            using var inputTensor = new FastTensor<float>(1, 5);
            using var input = new AutogradNode(inputTensor, false);
            using var output = seq.Forward(input);

            Assert.Equal(5, output.Data.Shape[1]);
        }

        [Fact]
        public void ResidualBlock_Forward_CalculatesCorrectShape()
        {
            // ResidualBlock musi zachować wymiary wejściowe (Skip Connection)
            using var res = new ResidualBlock(8);
            using var inputTensor = new FastTensor<float>(1, 8);
            using var input = new AutogradNode(inputTensor, false);
            using var output = res.Forward(input);

            Assert.Equal(input.Data.Shape[1], output.Data.Shape[1]);
        }
    }
}