// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Core
{
    /// <summary>
    /// Manages the recording and execution of operations for automatic differentiation (Reverse Mode).
    /// </summary>
    public sealed class ComputationGraph
    {
        private const int InitialCapacity = 4096;
        private TapeOp[] _tape = new TapeOp[InitialCapacity];
        private int _opCount;

        public bool IsRecording { get; set; } = true;

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
            if (!IsRecording) return;

            if (_opCount >= _tape.Length)
            {
                Array.Resize(ref _tape, _tape.Length * 2);
            }

            _tape[_opCount++] = new TapeOp(code, output, a, b, i0, i1, i2, i3, i4, nodeContext);
        }

        /// <summary>
        /// Performs the backward pass, propagating gradients from the loss node through the tape.
        /// </summary>
        public void Backward(AutogradNode lossNode)
        {
            if (lossNode?.Grad == null) return;

            // Seed the gradient with 1.0. 
            // Manual zeroing of intermediate node gradients is not required because transient nodes 
            // are cleared upon allocation, and parameters are cleared by the optimizer.
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