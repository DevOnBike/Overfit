using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Tests
{
    public class DeepLearningModuleTests : IDisposable
    {
        public DeepLearningModuleTests()
        {
            ComputationGraph.Active = new ComputationGraph();
        }

        public void Dispose() => ComputationGraph.Active = null;

        [Fact]
        public void LinearLayer_ForwardAndSerialization_Correct()
        {
            // Test wymiarów i parametrów
            using var layer = new LinearLayer(10, 5);
            Assert.Equal(2, layer.Parameters().Count());

            using var input = new AutogradNode(new FastMatrix<float>(2, 10), false);
            using var output = layer.Forward(input);

            Assert.Equal(2, output.Data.Rows);
            Assert.Equal(5, output.Data.Cols);

            // Test zapisu i odczytu
            using var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, true)) layer.Save(bw);

            ms.Position = 0;
            using var layer2 = new LinearLayer(10, 5);
            using (var br = new BinaryReader(ms)) layer2.Load(br);

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

            using var input = new AutogradNode(new FastMatrix<float>(1, 5), false);
            using var output = seq.Forward(input);
            Assert.Equal(5, output.Data.Cols);
        }

        [Fact]
        public void ResidualBlock_Forward_CalculatesCorrectShape()
        {
            // ResidualBlock musi zachować wymiary wejściowe
            using var res = new ResidualBlock(8);
            using var input = new AutogradNode(new FastMatrix<float>(1, 8), false);
            using var output = res.Forward(input);

            Assert.Equal(input.Data.Cols, output.Data.Cols);
        }
    }
}