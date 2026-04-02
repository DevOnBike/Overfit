namespace DevOnBike.Overfit.Core
{
    public sealed class ComputationGraph
    {
        private const int InitialCapacity = 4096;
        private TapeOp[] _tape = new TapeOp[InitialCapacity];
        private int _opCount = 0;

        private static readonly ThreadLocal<ComputationGraph> _active = new(() => new ComputationGraph());
        public static ComputationGraph Active { get => _active.Value; set => _active.Value = value; }

        public bool IsRecording { get; set; } = true;

        public void Record(OpCode code, AutogradNode output, AutogradNode a, AutogradNode b = null,
            int i0 = 0, int i1 = 0, int i2 = 0, int i3 = 0, int i4 = 0, AutogradNode[] nodeContext = null)
        {
            if (!IsRecording) return;
            if (_opCount >= _tape.Length) Array.Resize(ref _tape, _tape.Length * 2);
            _tape[_opCount++] = new TapeOp(code, output, a, b, i0, i1, i2, i3, i4, nodeContext);
        }

        public void Backward(AutogradNode lossNode)
        {
            if (lossNode?.Grad == null) return;

            // Zerowanie gradientów przed fazą Backward (poprawka dla ref struct Span)
            for (var i = 0; i < _opCount; i++)
            {
                ref readonly var op = ref _tape[i];
                if (op.Output?.Grad != null) op.Output.Grad.AsSpan().Clear();
                if (op.A?.Grad != null) op.A.Grad.AsSpan().Clear();
                if (op.B?.Grad != null) op.B.Grad.AsSpan().Clear();
            }

            lossNode.Grad.AsSpan().Fill(1f);

            for (var i = _opCount - 1; i >= 0; i--)
            {
                ExecuteBackward(in _tape[i]);
            }
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
                case OpCode.SoftmaxCrossEntropy: TensorMath.SoftmaxCrossEntropyBackward(op.A, op.B, op.Output); break;
                case OpCode.Conv2D: TensorMath.Conv2DBackward(op.A, op.B, op.Output, op.I0, op.I1, op.I2, op.I3, op.I4); break;
                case OpCode.MaxPool2D: TensorMath.MaxPool2DBackward(op.A, op.B, op.Output); break;
                case OpCode.GlobalAveragePool2D: TensorMath.GlobalAvgPool2DBackward(op.A, op.Output, op.I0, op.I1, op.I2); break;
                case OpCode.BatchNorm1D: TensorMath.BatchNorm1DBackward(op.A, op.Output, op.NodeContext[0], op.NodeContext[1], op.NodeContext[2], op.NodeContext[3]); break;
            }
        }

        public void Reset()
        {
            if (_opCount > 0) Array.Clear(_tape, 0, _opCount);
            _opCount = 0;
        }
    }
}