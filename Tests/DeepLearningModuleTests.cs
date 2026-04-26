// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Tests
{
    public class DeepLearningModuleTests
    {
        [Fact]
        public void LinearLayer_ForwardAndSerialization_Correct()
        {
            using var layer = new LinearLayer(10, 5);
            Assert.Equal(2, layer.Parameters().Count());

            using var inputTensor = new TensorStorage<float>(20, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(2, 10), false);

            using var output = layer.Forward(null, input);

            Assert.Equal(2, output.DataView.GetDim(0));
            Assert.Equal(5, output.DataView.GetDim(1));

            using var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, Encoding.UTF8, true))
            {
                layer.Save(bw);
            }

            ms.Position = 0;
            using var layer2 = new LinearLayer(10, 5);
            using (var br = new BinaryReader(ms))
            {
                layer2.Load(br);
            }

            Assert.Equal(layer.Weights.DataView.AsReadOnlySpan()[0], layer2.Weights.DataView.AsReadOnlySpan()[0]);
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

            using var inputTensor = new TensorStorage<float>(5, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(1, 5), false);

            using var output = seq.Forward(null, input);

            Assert.Equal(5, output.DataView.GetDim(1));
        }

        [Fact]
        public void ResidualBlock_Forward_CalculatesCorrectShape()
        {
            using var res = new ResidualBlock(8);

            // POPRAWKA: Blok rezydualny na bazie LinearLayer oczekuje tensora 2D [Batch, HiddenSize]
            using var inputTensor = new TensorStorage<float>(16, clearMemory: true);
            using var input = new AutogradNode(inputTensor, new TensorShape(2, 8), false);

            using var output = res.Forward(null, input);

            Assert.Equal(2, output.DataView.GetDim(0));
            Assert.Equal(8, output.DataView.GetDim(1));
        }
    }
}