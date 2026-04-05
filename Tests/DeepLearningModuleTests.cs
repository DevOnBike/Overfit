// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Tests
{
    public class DeepLearningModuleTests
    {
        [Fact]
        public void LinearLayer_ForwardAndSerialization_Correct()
        {
            using var layer = new LinearLayer(10, 5);
            Assert.Equal(2, layer.Parameters().Count());

            using var inputTensor = new FastTensor<float>(2, 10);
            using var input = new AutogradNode(inputTensor, false);

            // Przekazujemy null, bo nie potrzebujemy nagrywać operacji dla testów wymiarów
            using var output = layer.Forward(null, input);

            Assert.Equal(2, output.Data.Shape[0]);
            Assert.Equal(5, output.Data.Shape[1]);

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
            var l1 = new LinearLayer(5, 5);
            var act = new ReluActivation();
            using var seq = new Sequential(l1, act);

            seq.Eval();
            Assert.False(seq.IsTraining);
            Assert.False(l1.IsTraining);

            seq.Train();
            Assert.True(seq.IsTraining);

            using var inputTensor = new FastTensor<float>(1, 5);
            using var input = new AutogradNode(inputTensor, false);

            // Null jako ComputationGraph w trybie testowym
            using var output = seq.Forward(null, input);

            Assert.Equal(5, output.Data.Shape[1]);
        }

        [Fact]
        public void ResidualBlock_Forward_CalculatesCorrectShape()
        {
            using var res = new ResidualBlock(8);
            using var inputTensor = new FastTensor<float>(1, 8);
            using var input = new AutogradNode(inputTensor, false);

            using var output = res.Forward(null, input);

            Assert.Equal(input.Data.Shape[1], output.Data.Shape[1]);
        }
    }
}