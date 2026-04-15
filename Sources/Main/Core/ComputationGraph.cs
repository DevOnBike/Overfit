using System.Diagnostics;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.Diagnostics.Contracts;

namespace DevOnBike.Overfit.Core
{
    /// <summary>
    /// Manages the recording and execution of operations for automatic differentiation (Reverse Mode).
    /// </summary>
    public sealed class ComputationGraph
    {
        private const int InitialCapacity = 4096;

        private int _opCount;
        private TapeOp[] _tape = new TapeOp[InitialCapacity];

        public bool IsRecording { get; set; } = true;

        public int RecordedOpCount => _opCount;

        public void Record(
            OpCode code,
            AutogradNode output,
            AutogradNode a,
            AutogradNode b = null,
            int i0 = 0,
            int i1 = 0,
            int i2 = 0,
            int i3 = 0,
            int i4 = 0,
            AutogradNode[] nodeContext = null)
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

            if (OverfitDiagnostics.IsEnabled())
            {
                OverfitDiagnostics.Counter($"graph.op.{code}", 1);
            }
        }

        public void Backward(AutogradNode lossNode)
        {
            if (lossNode == null || !lossNode.RequiresGrad)
            {
                return;
            }

            long allocBefore = 0;
            long start = 0;
            var gc0Before = 0;
            var gc1Before = 0;
            var gc2Before = 0;

            if (OverfitDiagnostics.IsEnabled())
            {
                allocBefore = GC.GetTotalAllocatedBytes(false);
                start = Stopwatch.GetTimestamp();
                gc0Before = GC.CollectionCount(0);
                gc1Before = GC.CollectionCount(1);
                gc2Before = GC.CollectionCount(2);
            }

            lossNode.GradView.AsSpan().Fill(1f);

            for (var i = _opCount - 1; i >= 0; i--)
            {
                ExecuteBackward(in _tape[i]);
            }

            if (OverfitDiagnostics.IsEnabled())
            {
                var allocAfter = GC.GetTotalAllocatedBytes(false);
                var end = Stopwatch.GetTimestamp();

                OverfitDiagnostics.GraphCompleted(new GraphDiagnosticEvent(
                    TapeOpCount: _opCount,
                    BackwardMs: (end - start) * 1000.0 / Stopwatch.Frequency,
                    AllocatedBytes: allocAfter - allocBefore,
                    Gen0Collections: GC.CollectionCount(0) - gc0Before,
                    Gen1Collections: GC.CollectionCount(1) - gc1Before,
                    Gen2Collections: GC.CollectionCount(2) - gc2Before,
                    Phase: "backward",
                    IsTraining: true,
                    BatchSize: lossNode.DataView.Rank > 0 ? lossNode.DataView.GetDim(0) : 0));
            }
        }

        public AutogradNode Add(AutogradNode left, AutogradNode right) => TensorMath.Add(this, left, right);
        public AutogradNode AddBias(AutogradNode input, AutogradNode bias) => TensorMath.AddBias(this, input, bias);
        public AutogradNode MatMul(AutogradNode left, AutogradNode right) => TensorMath.MatMul(this, left, right);
        public AutogradNode ReLU(AutogradNode input) => TensorMath.ReLU(this, input);
        public AutogradNode MeanSquaredError(AutogradNode prediction, AutogradNode target) => TensorMath.MSELoss(this, prediction, target);
        public AutogradNode DirectionalLoss(AutogradNode prediction, AutogradNode target, float gamma = 10f) => TensorMath.DirectionalLoss(this, prediction, target, gamma);
        public AutogradNode Sigmoid(AutogradNode input) => TensorMath.Sigmoid(this, input);
        public AutogradNode Tanh(AutogradNode input) => TensorMath.Tanh(this, input);
        public AutogradNode Multiply(AutogradNode a, AutogradNode b) => TensorMath.Multiply(this, a, b);
        public AutogradNode GateSlice(AutogradNode gates, int hiddenSize, int gateIndex) => TensorMath.GateSlice(this, gates, hiddenSize, gateIndex);

        private void ExecuteBackward(in TapeOp op)
        {
            switch (op.Code)
            {
                case OpCode.Add:
                    TensorMath.AddBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.Subtract:
                    TensorMath.SubtractBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.AddBias:
                    TensorMath.AddBiasBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.MatMul:
                    TensorMath.MatMulBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.ReLU:
                    TensorMath.ReluBackward(op.A, op.Output);
                    break;
                case OpCode.Dropout:
                    TensorMath.DropoutBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.MseLoss:
                    TensorMath.MSELossBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.SoftmaxCrossEntropy:
                    TensorMath.SoftmaxCrossEntropyBackward(op.A, op.B, op.Output, op.NodeContext[0]);
                    break;
                case OpCode.Conv2D:
                    TensorMath.Conv2DBackward(op.A, op.B, op.Output, op.I0, op.I1, op.I2, op.I3, op.I4);
                    break;
                case OpCode.MaxPool2D:
                    TensorMath.MaxPool2DBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.GlobalAveragePool2D:
                    TensorMath.GlobalAvgPool2DBackward(op.A, op.Output, op.I0, op.I1, op.I2);
                    break;
                case OpCode.BatchNorm1D:
                    TensorMath.BatchNorm1DBackward(op.A, op.Output, op.NodeContext[0], op.NodeContext[1], op.NodeContext[2], op.NodeContext[3]);
                    break;
                case OpCode.Reshape:
                    TensorMath.ReshapeBackward(op.A, op.Output);
                    break;
                case OpCode.Sigmoid:
                    TensorMath.SigmoidBackward(op.A, op.Output);
                    break;
                case OpCode.Tanh:
                    TensorMath.TanhBackward(op.A, op.Output);
                    break;
                case OpCode.Multiply:
                    TensorMath.MultiplyBackward(op.A, op.B, op.Output);
                    break;
                case OpCode.GateSlice:
                    TensorMath.GateSliceBackward(op.A, op.Output, op.I1, op.I0);
                    break;
                case OpCode.TimestepSlice:
                    TensorMath.TimestepSliceBackward(op.A, op.Output, op.I0, op.I1, op.I2);
                    break;
                case OpCode.StackTimesteps:
                    TensorMath.StackTimestepsBackward(op.NodeContext, op.Output, op.I0, op.I1, op.I2);
                    break;
                case OpCode.RepeatVector:
                    TensorMath.RepeatVectorBackward(op.A, op.Output, op.I0, op.I1);
                    break;
                case OpCode.FusedLSTMStep:
                    TensorMath.FusedLSTMStepBackward(op.A, op.B, op.Output, op.NodeContext);
                    break;
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
