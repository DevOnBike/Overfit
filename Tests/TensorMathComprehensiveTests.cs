using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests
{
    public class TensorMathComprehensiveTests : IDisposable
    {
        public TensorMathComprehensiveTests()
        {
            ComputationGraph.Active = new ComputationGraph();
        }

        public void Dispose() => ComputationGraph.Active = null;

        [Fact]
        public void AddBias_ForwardAndBackward_NCHW_Correct()
        {
            using var input = new AutogradNode(new FastTensor<float>(2, 3, 1, 1));
            using var bias = new AutogradNode(new FastTensor<float>(3)); // Tensor 1D

            input.Data.AsSpan().Fill(1f);
            var bSpan = bias.Data.AsSpan();
            bSpan[0] = 10f; bSpan[1] = 20f; bSpan[2] = 30f;

            using var output = TensorMath.AddBias(input, bias);

            // Forward: Output jest 4D
            Assert.Equal(11f, output.Data[0, 0, 0, 0]);
            Assert.Equal(21f, output.Data[0, 1, 0, 0]);
            Assert.Equal(31f, output.Data[1, 2, 0, 0]);

            // Backward
            output.Grad.AsSpan().Fill(1f);
            ComputationGraph.Active.Backward(output);

            // Bias jest 1D - używamy indeksu [i]
            Assert.Equal(2f, bias.Grad[0]);
            Assert.Equal(2f, bias.Grad[1]);
        }

        [Fact]
        public void BatchNorm1D_NormalizationTheory()
        {
            // Batch=3, Features=1 (Tensor 2D)
            using var input = new AutogradNode(new FastTensor<float>(3, 1));
            var inS = input.Data.AsSpan();
            inS[0] = 10f; inS[1] = 20f; inS[2] = 30f;

            using var gamma = new AutogradNode(new FastTensor<float>(1));
            using var beta = new AutogradNode(new FastTensor<float>(1));
            gamma.Data.AsSpan().Fill(1f);
            beta.Data.AsSpan().Fill(0f);

            using var rm = new FastTensor<float>(1);
            using var rv = new FastTensor<float>(1);
            rv.AsSpan().Fill(1f);

            using var output = TensorMath.BatchNorm1D(input, gamma, beta, rm, rv, 0.1f, 1e-5f, true);

            // Output jest 2D - używamy indeksu [i, j]
            // Średnia z 10,20,30 to 20. Wartość 20 (indeks [1,0]) powinna stać się 0.
            Assert.InRange(output.Data[1, 0], -0.01f, 0.01f);

            // Symetria: -1.22 i 1.22
            Assert.Equal(-output.Data[0, 0], output.Data[2, 0], 3);
        }

        [Fact]
        public void SoftmaxCrossEntropy_ForwardAndBackward_MathematicalTruth()
        {
            using var logits = new AutogradNode(new FastTensor<float>(1, 3));
            using var target = new AutogradNode(new FastTensor<float>(1, 3), false);

            var lS = logits.Data.AsSpan();
            lS[0] = 2.0f; lS[1] = 1.0f; lS[2] = 0.1f;
            target.Data[0, 0] = 1.0f;

            using var loss = TensorMath.SoftmaxCrossEntropy(logits, target);

            // Loss dla Softmaxa (1, 3)
            Assert.InRange(loss.Data[0, 0], 0.41f, 0.43f);

            ComputationGraph.Active.Backward(loss);

            // Gradient (Pred - Target)
            Assert.InRange(logits.Grad[0, 0], -0.35f, -0.33f);
            Assert.InRange(logits.Grad[0, 1], 0.23f, 0.25f);
        }

        [Fact]
        public void Reshape_ZeroCopy_Integrity()
        {
            using var tensor = new FastTensor<float>(1, 8, 13, 13);
            tensor.AsSpan().Fill(7f);

            // Reshape 4D -> 2D
            using var reshaped = tensor.Reshape(1, 1352);

            Assert.Equal(1352, reshaped.Shape[1]);
            // Dostęp do spłaszczonego tensora 2D
            Assert.Equal(7f, reshaped[0, 1351]);
        }
    }
}