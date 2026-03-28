namespace DevOnBike.Overfit.Tests
{
    public class AutogradTests
    {
        private const int Precision = 5;

        // Pomocnicza metoda do wstrzykiwania sygnału błędu 
        // W prawdziwym ML robi to funkcja straty (Loss Function)
        private void FillGradient(Tensor tensor, double value)
        {
            var span = tensor.Grad.AsSpan();
            for (var i = 0; i < span.Length; i++)
            {
                span[i] = value;
            }
        }

        [Fact]
        public void TensorAdd_ForwardAndBackward_FlowsGradientsEqually()
        {
            // Arrange
            using var matA = new FastMatrix<double>(2, 2);
            using var matB = new FastMatrix<double>(2, 2);
            matA.AsSpan().Fill(1.0);
            matB.AsSpan().Fill(2.0);

            using var a = new Tensor(matA);
            using var b = new Tensor(matB);

            // Act - Forward
            using var c = TensorMath.Add(a, b);

            // Assert - Forward
            Assert.Equal(3.0, c.Data[0, 0]);

            // Act - Backward
            // W profesjonalnych silnikach '.Backward()' na węźle końcowym (Loss) 
            // inicjuje bazowy gradient jedynkami. Tu symulujemy to ręcznie,
            // ponieważ nasza architektura wstrzykuje węzeł w delegatach.
            FillGradient(c, 5.0); 
            c.Backward(); // Odpala graf topologiczny i regułę łańcuchową

            // Assert - Backward (Gradient z dodawania przepływa równo 1:1)
            Assert.Equal(5.0, a.Grad[0, 0], Precision);
            Assert.Equal(5.0, a.Grad[1, 1], Precision);
            
            Assert.Equal(5.0, b.Grad[0, 0], Precision);
            Assert.Equal(5.0, b.Grad[1, 1], Precision);
        }

        [Fact]
        public void TensorMatMul_ForwardAndBackward_CalculatesMatrixDerivatives()
        {
            // Arrange
            using var matA = new FastMatrix<double>(2, 2);
            matA[0, 0] = 1; matA[0, 1] = 2;
            matA[1, 0] = 3; matA[1, 1] = 4;

            using var matB = new FastMatrix<double>(2, 2);
            matB[0, 0] = 5; matB[0, 1] = 6;
            matB[1, 0] = 7; matB[1, 1] = 8;

            using var a = new Tensor(matA);
            using var b = new Tensor(matB);

            // Act - Forward (C = A * B)
            using var c = TensorMath.MatMul(a, b);

            // Assert - Forward
            // C = [19, 22]
            //     [43, 50]
            Assert.Equal(19.0, c.Data[0, 0]);

            // Symulujemy pochodną Lossu po C równą macierzy jedynek [1, 1; 1, 1]
            FillGradient(c, 1.0);

            // Act - Backward
            c.Backward();

            // Assert - Backward dla a.Grad (gradA = gradC * B^T)
            // [1, 1]   [5, 7]   [1*5 + 1*6, 1*7 + 1*8]   [11, 15]
            // [1, 1] * [6, 8] = [1*5 + 1*6, 1*7 + 1*8] = [11, 15]
            Assert.Equal(11.0, a.Grad[0, 0], Precision);
            Assert.Equal(15.0, a.Grad[0, 1], Precision);
            Assert.Equal(11.0, a.Grad[1, 0], Precision);
            Assert.Equal(15.0, a.Grad[1, 1], Precision);

            // Assert - Backward dla b.Grad (gradB = A^T * gradC)
            // [1, 3]   [1, 1]   [1*1 + 3*1, 1*1 + 3*1]   [4, 4]
            // [2, 4] * [1, 1] = [2*1 + 4*1, 2*1 + 4*1] = [6, 6]
            Assert.Equal(4.0, b.Grad[0, 0], Precision);
            Assert.Equal(4.0, b.Grad[0, 1], Precision);
            Assert.Equal(6.0, b.Grad[1, 0], Precision);
            Assert.Equal(6.0, b.Grad[1, 1], Precision);
        }

        [Fact]
        public void Tensor_RequiresGradFalse_DoesNotCalculateGradient()
        {
            // Arrange
            using var matA = new FastMatrix<double>(2, 2);
            using var matB = new FastMatrix<double>(2, 2);
            matA.AsSpan().Fill(1.0);
            matB.AsSpan().Fill(2.0);

            // Wyłączamy śledzenie gradientu dla A (np. gdy to są tylko dane wejściowe, a nie wagi)
            using var a = new Tensor(matA, requiresGrad: false); 
            using var b = new Tensor(matB, requiresGrad: true);

            // Act
            using var c = TensorMath.MatMul(a, b);
            FillGradient(c, 1.0);
            c.Backward();

            // Assert
            // Gradient dla a nie powinien być tknięty (pozostaje zerem)
            Assert.Equal(0.0, a.Grad[0, 0]);
            
            // Gradient dla b policzony normalnie
            Assert.True(b.Grad[0, 0] > 0.0);
        }
        
        [Fact]
        public void ReLU_ForwardAndBackward_FiltersNegativeValuesAndGradients()
        {
            // Arrange
            using var matA = new FastMatrix<double>(2, 2);
            matA[0, 0] = 5.0;  matA[0, 1] = -2.0;
            matA[1, 0] = -0.1; matA[1, 1] = 10.0;

            using var a = new Tensor(matA);

            // Act - Forward
            using var relu = TensorMath.ReLU(a);

            // Assert - Forward (Ujemne wyzerowane, dodatnie przepuszczone)
            Assert.Equal(5.0, relu.Data[0, 0]);
            Assert.Equal(0.0, relu.Data[0, 1]);
            Assert.Equal(0.0, relu.Data[1, 0]);
            Assert.Equal(10.0, relu.Data[1, 1]);

            // Symulujemy gradient przychodzący z góry równy 2.0 dla wszystkich komórek
            FillGradient(relu, 2.0);

            // Act - Backward
            relu.Backward();

            // Assert - Backward 
            // (Gradient przepływa tylko tam, gdzie wejście było > 0)
            Assert.Equal(2.0, a.Grad[0, 0]); // Przepuszczony
            Assert.Equal(0.0, a.Grad[0, 1]); // Zablokowany (bo x <= 0)
            Assert.Equal(0.0, a.Grad[1, 0]); // Zablokowany
            Assert.Equal(2.0, a.Grad[1, 1]); // Przepuszczony
        }

        [Fact]
        public void MSE_ForwardAndBackward_CalculatesCorrectLossAndDerivative()
        {
            // Arrange
            using var matPred = new FastMatrix<double>(2, 1);
            matPred[0, 0] = 3.0; 
            matPred[1, 0] = 5.0;

            using var matTarget = new FastMatrix<double>(2, 1);
            matTarget[0, 0] = 1.0; 
            matTarget[1, 0] = 9.0;

            // N = 2
            // MSE = ((3-1)^2 + (5-9)^2) / 2 = (4 + 16) / 2 = 10.0
            
            // Pochodna dL/d(pred) = (2/N) * (pred - target)
            // dL/d(pred[0]) = (2/2) * (3 - 1) = 2.0
            // dL/d(pred[1]) = (2/2) * (5 - 9) = -4.0

            using var predictions = new Tensor(matPred, requiresGrad: true);
            using var targets = new Tensor(matTarget, requiresGrad: false);

            // Act - Forward
            using var loss = TensorMath.MSE(predictions, targets);

            // Assert - Forward
            Assert.Equal(10.0, loss.Data[0, 0], Precision);

            // Act - Backward
            // UWAGA: Funkcja straty zawsze startuje z gradientem bazowym 1.0
            loss.Grad[0, 0] = 1.0;
            loss.Backward();

            // Assert - Backward (Gradienty dla predykcji)
            Assert.Equal(2.0, predictions.Grad[0, 0], Precision);
            Assert.Equal(-4.0, predictions.Grad[1, 0], Precision);
        }
    }
}