namespace DevOnBike.Overfit.Core
{
    public sealed class ComputationGraph
    {
        private const int InitialCapacity = 4096;
        private TapeOp[] _tape = new TapeOp[InitialCapacity];
        private int _opCount;

        public bool IsRecording { get; set; } = true;

        public void Record(OpCode code, AutogradNode output, AutogradNode a, AutogradNode b = null,
            int i0 = 0, int i1 = 0, int i2 = 0, int i3 = 0, int i4 = 0, AutogradNode[] nodeContext = null)
        {
            if (!IsRecording)
            {
                return;
            }

            if (_opCount >= _tape.Length)
            {
                Array.Resize(ref _tape, _tape.Length * 2);
            }

            _tape[_opCount++] = new TapeOp(code, output, a, b, i0, i1, i2, i3, i4, nodeContext);
        }

        public void Backward(AutogradNode lossNode)
        {
            if (lossNode?.Grad == null)
            {
                return;
            }

            // Jedyne co robimy: ustawiamy gradient Loss na 1.0
            // Zerowanie gradientów NIE jest potrzebne tutaj, bo:
            //   - Parametry (w1, b1, ...) — zerowane przez optimizer.ZeroGrad() przed Forward
            //   - Intermediate nodes (l1, prediction, loss) — tworzone na nowo każdą epokę
            //     z using var, więc ich Grad jest świeży (zerowany w konstruktorze AutogradNode)
            lossNode.Grad.AsSpan().Fill(1f);

            for (var i = _opCount - 1; i >= 0; i--)
            {
                ExecuteBackward(in _tape[i]);
            }
        }

        public AutogradNode Add(AutogradNode left, AutogradNode right)
        {
            return TensorMath.Add(this, left, right);
        }

        public AutogradNode AddBias(AutogradNode input, AutogradNode bias)
        {
            return TensorMath.AddBias(this, input, bias);
        }

        public AutogradNode MatMul(AutogradNode left, AutogradNode right)
        {
            return TensorMath.MatMul(this, left, right);
        }

        public AutogradNode ReLU(AutogradNode input)
        {
            return TensorMath.ReLU(this, input);
        }

        public AutogradNode MeanSquaredError(AutogradNode prediction, AutogradNode target)
        {
            return TensorMath.MSELoss(this, prediction, target);
        }

        public AutogradNode DirectionalLoss(AutogradNode prediction, AutogradNode target, float gamma = 10f)
        {
            return TensorMath.DirectionalLoss(this, prediction, target, gamma);
        }

        private void ExecuteBackward(in TapeOp op)
        {
            switch (op.Code)
            {
                case OpCode.Add: TensorMath.AddBackward(op.A, op.B, op.Output); break;
                case OpCode.AddBias: TensorMath.AddBiasBackward(op.A, op.B, op.Output); break;
                case OpCode.MatMul: TensorMath.MatMulBackward(op.A, op.B, op.Output); break;
                case OpCode.ReLU: TensorMath.ReluBackward(op.A, op.Output); break;
                case OpCode.Dropout: TensorMath.DropoutBackward(op.A, op.B, op.Output); break;
                case OpCode.MSELoss: TensorMath.MSELossBackward(op.A, op.B, op.Output); break;
                case OpCode.SoftmaxCrossEntropy: TensorMath.SoftmaxCrossEntropyBackward(op.A, op.B, op.Output, op.NodeContext[0]); break;
                case OpCode.Conv2D: TensorMath.Conv2DBackward(op.A, op.B, op.Output, op.I0, op.I1, op.I2, op.I3, op.I4); break;
                case OpCode.MaxPool2D: TensorMath.MaxPool2DBackward(op.A, op.B, op.Output); break;
                case OpCode.GlobalAveragePool2D: TensorMath.GlobalAvgPool2DBackward(op.A, op.Output, op.I0, op.I1, op.I2); break;
                case OpCode.BatchNorm1D: TensorMath.BatchNorm1DBackward(op.A, op.Output, op.NodeContext[0], op.NodeContext[1], op.NodeContext[2], op.NodeContext[3]); break;
                case OpCode.Reshape: TensorMath.ReshapeBackward(op.A, op.Output); break;

                case OpCode.DirectionalLoss:
                    var gammaValue = BitConverter.Int32BitsToSingle(op.I0);
                    TensorMath.DirectionalLossBackward(op.A, op.B, op.Output, gammaValue);
                    break;
            }
        }

        public void Reset()
        {
            if (_opCount > 0)
            {
                Array.Clear(_tape, 0, _opCount);
            }

            _opCount = 0;
        }
    }
}