// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Autograd
{
    /// <summary>
    /// Manages the recording and execution of operations for automatic differentiation (Reverse Mode).
    /// </summary>
    public sealed partial class ComputationGraph : IDisposable
    {
        private const int InitialCapacity = 4096;

        // Default remains the original large arena for local/runtime behavior.
        // GitHub Actions gets a smaller default to avoid unmanaged OOM in tests that create
        // many short-lived ComputationGraph instances without disposing them immediately.
        private const int DefaultTapeBufferElements = 50_000_000;
        private const int CiTapeBufferElements = 1_048_576;
        private const string TapeBufferElementsEnvironmentVariable = "OVERFIT_GRAPH_TAPE_BUFFER_ELEMENTS";

        private static readonly int OpCodeCount = Enum.GetValues<OpCode>().Length;

        private int _opCount;
        private TapeOp[] _tape = new TapeOp[InitialCapacity];
        private Conv2DWorkspace? _conv2DWorkspace;

        // Huge arena buffer: bypasses GC and ArrayPool for intermediate tensor storage.
        public readonly NativeBufferManaged<float> TapeBuffer;

        public ComputationGraph()
            : this(ResolveDefaultTapeBufferElements())
        {
        }

        public ComputationGraph(int tapeBufferElements)
        {
            if (tapeBufferElements <= 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(tapeBufferElements),
                    "Tape buffer size must be greater than zero.");
            }

            TapeBuffer = new NativeBufferManaged<float>(tapeBufferElements, clearMemory: false);
        }

        public bool IsRecording { get; set; } = true;

        public int RecordedOpCount => _opCount;

        /// <summary>
        /// Fast allocation of raw storage from the arena.
        /// Returns only storage, without shape metadata.
        /// </summary>
        public TensorStorage<float> AllocateIntermediate(int length)
        {
            return new TensorStorage<float>(TapeBuffer, length);
        }

        internal Conv2DWorkspace GetConv2DWorkspace(
            int batchSize,
            int inC,
            int outC,
            int h,
            int w,
            int k)
        {
            var outH = h - k + 1;
            var outW = w - k + 1;
            var kSqInC = k * k * inC;
            var spatialOut = outH * outW;
            var colLength = kSqInC * spatialOut;
            var partialWeightGradientLength = outC * kSqInC;
            var workerCount = Math.Max(1, Math.Min(Environment.ProcessorCount, Math.Max(1, batchSize)));

            _conv2DWorkspace ??= new Conv2DWorkspace();

            _conv2DWorkspace.Ensure(
                workerCount,
                colLength,
                partialWeightGradientLength);

            return _conv2DWorkspace;
        }

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
            AutogradNode c0 = null,
            AutogradNode c1 = null,
            AutogradNode c2 = null,
            AutogradNode c3 = null,
            AutogradNode c4 = null,
            int contextCount = 0,
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

            _tape[_opCount++] = new TapeOp(
                code,
                output,
                a,
                b,
                i0,
                i1,
                i2,
                i3,
                i4,
                c0,
                c1,
                c2,
                c3,
                c4,
                contextCount,
                nodeContext);

            OverfitTelemetry.RecordGraphRecordOp(code);
        }

        public void Backward(AutogradNode lossNode)
        {
            if (lossNode == null || !lossNode.RequiresGrad)
            {
                return;
            }

            lossNode.GradView.AsSpan().Fill(1f);

            for (var i = _opCount - 1; i >= 0; i--)
            {
                ExecuteBackward(in _tape[i]);
            }
        }

        /*
        public AutogradNode Add(AutogradNode left, AutogradNode right) => TensorMath.Add(this, left, right);

        public AutogradNode AddBias(AutogradNode input, AutogradNode bias) => TensorMath.AddBias(this, input, bias);

        public AutogradNode MatMul(AutogradNode left, AutogradNode right) => TensorMath.MatMul(this, left, right);

        public AutogradNode ReLU(AutogradNode input) => TensorMath.ReLU(this, input);

        public AutogradNode MeanSquaredError(AutogradNode prediction, AutogradNode target) => TensorMath.MSELoss(this, prediction, target);

        public AutogradNode DirectionalLoss(AutogradNode prediction, AutogradNode target, float gamma = 10f) =>
            TensorMath.DirectionalLoss(this, prediction, target, gamma);

        public AutogradNode Sigmoid(AutogradNode input) => TensorMath.Sigmoid(this, input);

        public AutogradNode Tanh(AutogradNode input) => TensorMath.Tanh(this, input);
*/
        public AutogradNode Multiply(AutogradNode a, AutogradNode b) => TensorMath.Multiply(this, a, b);

        public AutogradNode GateSlice(AutogradNode gates, int hiddenSize, int gateIndex) =>
            TensorMath.GateSlice(this, gates, hiddenSize, gateIndex);

        public AutogradNode AddInPlace(AutogradNode target, AutogradNode source)
        {
            TensorPrimitives.Add(
                target.DataView.AsSpan(),
                source.DataView.AsReadOnlySpan(),
                target.DataView.AsSpan());

            if (IsRecording)
            {
                Record(OpCode.AddInPlace, output: target, a: source);
            }

            return target;
        }

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

                case OpCode.Linear:
                    TensorMath.LinearBackward(op.A, op.B, op.C0, op.Output);
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
                    TensorMath.SoftmaxCrossEntropyBackward(op.A, op.B, op.Output, op.C0);
                    break;

                case OpCode.Conv2D:
                    TensorMath.Conv2DBackward(this, op.A, op.B, op.Output, op.I0, op.I1, op.I2, op.I3, op.I4);
                    break;

                case OpCode.MaxPool2D:
                    TensorMath.MaxPool2DBackward(op.A, op.B, op.Output);
                    break;

                case OpCode.GlobalAveragePool2D:
                    TensorMath.GlobalAvgPool2DBackward(op.A, op.Output, op.I0, op.I1, op.I2);
                    break;

                case OpCode.BatchNorm1D:
                    TensorMath.BatchNorm1DBackward(op.A, op.Output, op.C0, op.C1, op.C2, op.C3);
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
                    TensorMath.DirectionalLossBackward(op.A, op.B, op.Output, BitConverter.Int32BitsToSingle(op.I0));
                    break;

                case OpCode.AddInPlace:
                    if (op.A.RequiresGrad)
                    {
                        TensorPrimitives.Add(
                            op.A.GradView.AsSpan(),
                            op.Output.GradView.AsReadOnlySpan(),
                            op.A.GradView.AsSpan());
                    }

                    break;
            }
        }

        public void Reset()
        {
            for (var i = 0; i < _opCount; i++)
            {
                ref readonly var op = ref _tape[i];

                // Dispose by ownership — no hardcoded OpCode switch needed.
                //
                // op.Output  → GraphTemporary  (activation output of this op)
                // op.B, C0-C4 → GraphAuxiliary (e.g., maxIndices, probsNode, mean/invStd)
                //               or Parameter   (gamma/beta nodes) → do NOT dispose
                //               or null
                // op.A       → output of previous TapeOp, disposed by its own entry → skip
                //
                // By routing all allocation through graph factory methods (Etap 3),
                // every node has Ownership set correctly at birth. Reset() trusts it.
                DisposeIfGraphOwned(op.Output);
                DisposeIfGraphOwned(op.B);
                DisposeIfGraphOwned(op.C0);
                DisposeIfGraphOwned(op.C1);
                DisposeIfGraphOwned(op.C2);
                DisposeIfGraphOwned(op.C3);
                DisposeIfGraphOwned(op.C4);

                if (op.NodeContext != null)
                {
                    foreach (var node in op.NodeContext)
                    {
                        DisposeIfGraphOwned(node);
                    }
                }
            }

            if (_opCount > 0)
            {
                Array.Clear(_tape, 0, _opCount);
            }

            _opCount = 0;

            // Reset arena without involving GC.
            TapeBuffer.ResetOffset();
        }

        [System.Runtime.CompilerServices.MethodImpl(
            System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private static void DisposeIfGraphOwned(AutogradNode? node)
        {
            if (node is
                {
                    Ownership: AutogradNodeOwnership.GraphTemporary
                                  or AutogradNodeOwnership.GraphAuxiliary
                })
            {
                node.Dispose();
            }
        }



        public void Dispose()
        {
            _conv2DWorkspace?.Dispose();
            TapeBuffer.Dispose();
        }

        private static int ResolveDefaultTapeBufferElements()
        {
            var explicitValue = Environment.GetEnvironmentVariable(TapeBufferElementsEnvironmentVariable);

            if (int.TryParse(explicitValue, out var parsed) && parsed > 0)
            {
                return parsed;
            }

            var isGitHubActions = string.Equals(
                Environment.GetEnvironmentVariable("GITHUB_ACTIONS"),
                "true",
                StringComparison.OrdinalIgnoreCase);

            return isGitHubActions
                ? CiTapeBufferElements
                : DefaultTapeBufferElements;
        }
    }
}