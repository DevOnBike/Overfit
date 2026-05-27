// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning;

namespace DevOnBike.Overfit.Tests.DeepLearning
{
    /// <summary>
    /// Fast structural checks for the <see cref="Crnn"/> facade (no training): geometry, forward logit
    /// shape, the convenience recogniser, and config validation.
    /// </summary>
    public sealed class CrnnTests
    {
        [Fact]
        public void Geometry_And_Defaults()
        {
            using var crnn = new Crnn(imageHeight: 8, imageWidth: 24, classCount: 11);
            Assert.Equal(24 - 3 + 1, crnn.TimeSteps); // VALID conv, default kernel 3
            Assert.Equal(11, crnn.ClassCount);
            Assert.Equal(10, crnn.BlankIndex);        // default = classCount - 1
            Assert.Equal(8, crnn.ImageHeight);
            Assert.Equal(24, crnn.ImageWidth);
        }

        [Fact]
        public void Forward_ProducesTimestepByClassLogits()
        {
            using var crnn = new Crnn(imageHeight: 8, imageWidth: 24, classCount: 11);
            crnn.Eval();
            using var graph = new ComputationGraph(4_000_000);

            var image = new float[8 * 24]; // blank image is fine for a shape check
            using var input = crnn.CreateInput(image);
            var logits = crnn.Forward(graph, input);

            Assert.Equal(crnn.TimeSteps * crnn.ClassCount, logits.Shape.Size);
        }

        [Fact]
        public void Recognize_ConvenienceOverload_Runs_And_ReturnsInRangeLabels()
        {
            using var crnn = new Crnn(imageHeight: 8, imageWidth: 24, classCount: 11);
            crnn.Eval();

            var decoded = crnn.Recognize(new float[8 * 24]); // internal graph
            // Untrained ⇒ content arbitrary, but every emitted label must be a valid non-blank class.
            foreach (var label in decoded)
            {
                Assert.InRange(label, 0, crnn.ClassCount - 1);
                Assert.NotEqual(crnn.BlankIndex, label);
            }
        }

        [Fact]
        public void Constructor_Rejects_KernelLargerThanImage()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new Crnn(4, 24, 11, kernelSize: 5));
        }
    }
}
