using DevOnBike.Overfit.Core;
using DevOnBike.Overfit.Optimizers;

namespace DevOnBike.Overfit.Tests
{
    public class SGDTests
    {
        private const int Precision = 5;

        [Fact]
        public void Step_UpdatesWeightsCorrectly_UsingGradientsAndLearningRate()
        {
            // Arrange
            using var mat = new FastMatrix<double>(2, 2);
            mat[0, 0] = 1.0; mat[0, 1] = 2.0;
            mat[1, 0] = 3.0; mat[1, 1] = 4.0;

            using var tensor = new AutogradNode(mat, requiresGrad: true);

            // Symulujemy policzone gradienty po wykonaniu .Backward()
            tensor.Grad[0, 0] = 0.5; tensor.Grad[0, 1] = 0.1;
            tensor.Grad[1, 0] = -1.0; tensor.Grad[1, 1] = 2.0;

            var learningRate = 0.1;
            var optimizer = new SGD([tensor], learningRate);

            // Act
            optimizer.Step();

            // Assert
            // W_new = W_old - LR * Grad
            // [0,0]: 1.0 - (0.1 * 0.5) = 1.0 - 0.05 = 0.95
            Assert.Equal(0.95, tensor.Data[0, 0], Precision);
            
            // [0,1]: 2.0 - (0.1 * 0.1) = 2.0 - 0.01 = 1.99
            Assert.Equal(1.99, tensor.Data[0, 1], Precision);
            
            // [1,0]: 3.0 - (0.1 * -1.0) = 3.0 + 0.1 = 3.1
            Assert.Equal(3.1, tensor.Data[1, 0], Precision);
            
            // [1,1]: 4.0 - (0.1 * 2.0) = 4.0 - 0.2 = 3.8
            Assert.Equal(3.8, tensor.Data[1, 1], Precision);
        }

        [Fact]
        public void ZeroGrad_ClearsAllGradientsToZero()
        {
            // Arrange
            using var mat = new FastMatrix<double>(2, 2);
            using var tensor = new AutogradNode(mat, requiresGrad: true);

            tensor.Grad.AsSpan().Fill(42.0); // Zanieczyszczamy gradient

            var optimizer = new SGD([tensor], learningRate: 0.1);

            // Act
            optimizer.ZeroGrad();

            // Assert
            Assert.Equal(0.0, tensor.Grad[0, 0]);
            Assert.Equal(0.0, tensor.Grad[0, 1]);
            Assert.Equal(0.0, tensor.Grad[1, 0]);
            Assert.Equal(0.0, tensor.Grad[1, 1]);
        }

        [Fact]
        public void Step_IgnoresTensorsWithRequiresGradFalse()
        {
            // Arrange
            using var mat = new FastMatrix<double>(1, 1);
            mat[0, 0] = 10.0;
            
            // Tensor, którego nie chcemy trenować (np. zamrożona warstwa)
            using var tensor = new AutogradNode(mat, requiresGrad: false);
            tensor.Grad[0, 0] = 5.0; // Gradient sztucznie ustawiony

            var optimizer = new SGD([tensor], learningRate: 0.1);

            // Act
            optimizer.Step();

            // Assert - Dane nie mogły ulec zmianie
            Assert.Equal(10.0, tensor.Data[0, 0]);
        }
    }
}