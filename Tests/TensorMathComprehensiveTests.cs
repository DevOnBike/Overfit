using DevOnBike.Overfit.Core;

namespace DevOnBike.Overfit.Tests
{
    public class TensorMathComprehensiveTests : IDisposable
    {
        public TensorMathComprehensiveTests() { ComputationGraph.Active = new ComputationGraph(); }
        public void Dispose() { ComputationGraph.Active = null; }

        [Fact]
        public void Add_ForwardAndBackward_Correct()
        {
            using var a = new AutogradNode(new FastTensor<float>(1, 2));
            using var b = new AutogradNode(new FastTensor<float>(1, 2));
            ((Span<float>)[1f, 2f]).CopyTo(a.Data.AsSpan());
            ((Span<float>)[3f, 4f]).CopyTo(b.Data.AsSpan());

            using var res = TensorMath.Add(a, b);
            Assert.Equal([4f, 6f], res.Data.AsSpan().ToArray());

            ComputationGraph.Active.Backward(res);
            Assert.Equal([1f, 1f], a.Grad.AsSpan().ToArray());
        }

        [Fact]
        public void MatMul_FullCheck()
        {
            using var a = new AutogradNode(new FastTensor<float>(1, 2));
            using var b = new AutogradNode(new FastTensor<float>(2, 1));
            ((Span<float>)[2f, 3f]).CopyTo(a.Data.AsSpan());
            ((Span<float>)[4f, 5f]).CopyTo(b.Data.AsSpan());

            using var res = TensorMath.MatMul(a, b);
            Assert.Equal(23.0f, res.Data[0, 0]);

            ComputationGraph.Active.Backward(res);
            Assert.Equal(4.0f, a.Grad[0, 0]);
            Assert.Equal(2.0f, b.Grad[0, 0]);
        }

        [Fact]
        public void Conv2D_ForwardAndBackward_Check()
        {
            using var input = new AutogradNode(new FastTensor<float>(1, 1, 3, 3));
            using var weights = new AutogradNode(new FastTensor<float>(1, 4)); // outC=1, k=2, inC=1 -> 1x4
            input.Data.AsSpan().Fill(1.0f);
            ((Span<float>)[1f, 1f, 1f, 1f]).CopyTo(weights.Data.AsSpan());

            using var res = TensorMath.Conv2D(input, weights, 1, 1, 3, 3, 2);
            // Wyjście 1x1x2x2, wszystkie wartości powinny być 4 (1*1+1*1+1*1+1*1)
            Assert.Equal(4.0f, res.Data[0, 0, 0, 0]);

            ComputationGraph.Active.Backward(res);
            Assert.NotNull(weights.Grad);
            Assert.NotNull(input.Grad);
        }
    }
}