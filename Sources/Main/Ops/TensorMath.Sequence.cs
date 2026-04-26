// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // ====================================================================
        // FUSED LSTM STEP
        // ====================================================================

        public static (AutogradNode hNew, AutogradNode cNew) FusedLSTMStep(ComputationGraph graph, AutogradNode x, AutogradNode hPrev, AutogradNode cPrev, AutogradNode W, AutogradNode U, AutogradNode B)
        {
            int batchSize = x.Shape.D0, hS = hPrev.Shape.D1;
            var gD = MatMulRaw(graph, x, W);
            var uh = MatMulRaw(graph, hPrev, U);

            var req = x.RequiresGrad || hPrev.RequiresGrad || cPrev.RequiresGrad || W.RequiresGrad || U.RequiresGrad || B.RequiresGrad;

            var cNode = AllocateNode(graph, new TensorShape(batchSize, hS), req, clearMemory: false);
            var hNode = AllocateNode(graph, new TensorShape(batchSize, hS), req, clearMemory: false);

            if (batchSize < BatchSequentialThreshold)
            {
                var gDS = gD.DataView.AsSpan();
                var uhS = uh.DataView.AsReadOnlySpan();
                var bS = B.DataView.AsReadOnlySpan();
                var cPrevS = cPrev.DataView.AsReadOnlySpan();
                var cnDS = cNode.DataView.AsSpan();
                var hnDS = hNode.DataView.AsSpan();

                for (var b = 0; b < batchSize; b++)
                {
                    ExecuteLSTMInner(b, hS, gDS, uhS, bS, cPrevS, cnDS, hnDS);
                }
            }
            else
            {
                Parallel.For(0, batchSize, b => ExecuteLSTMInner(b, hS,
                    gD.DataView.AsSpan(),
                    uh.DataView.AsReadOnlySpan(),
                    B.DataView.AsReadOnlySpan(),
                    cPrev.DataView.AsReadOnlySpan(),
                    cNode.DataView.AsSpan(),
                    hNode.DataView.AsSpan()));
            }

            if (graph != null && graph.IsRecording && req)
            {
                graph.Record(OpCode.FusedLSTMStep, hNode, x, hPrev, nodeContext: [cPrev, W, U, B, cNode, gD]);
            }
            else
            {
                if (!req)
                {
                    gD.Dispose();
                }
            }

            return (hNode, cNode);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ExecuteLSTMInner(int b, int hS, Span<float> gDS, ReadOnlySpan<float> uhS, ReadOnlySpan<float> bS, ReadOnlySpan<float> cPrevS, Span<float> cnDS, Span<float> hnDS)
        {
            var bg = gDS.Slice(b * 4 * hS, 4 * hS);
            TensorPrimitives.Add(bg, uhS.Slice(b * 4 * hS, 4 * hS), bg);
            TensorPrimitives.Add(bg, bS, bg);

            var f = bg.Slice(0, hS);
            var i = bg.Slice(hS, hS);
            var g = bg.Slice(2 * hS, hS);
            var o = bg.Slice(3 * hS, hS);
            TensorPrimitives.Sigmoid(f, f);
            TensorPrimitives.Sigmoid(i, i);
            TensorPrimitives.Tanh(g, g);
            TensorPrimitives.Sigmoid(o, o);

            var bcn = cnDS.Slice(b * hS, hS);
            TensorPrimitives.Multiply(f, cPrevS.Slice(b * hS, hS), bcn);
            TensorPrimitives.MultiplyAdd(i, g, bcn, bcn);

            var bhn = hnDS.Slice(b * hS, hS);
            TensorPrimitives.Tanh(bcn, bhn);
            TensorPrimitives.Multiply(o, bhn, bhn);
        }

        public static void FusedLSTMStepBackward(AutogradNode x, AutogradNode hPrev, AutogradNode hNew, AutogradNode[] ctx)
        {
            var cPrev = ctx[0];
            var W = ctx[1];
            var U = ctx[2];
            var B = ctx[3];
            var cNew = ctx[4];
            var gates = ctx[5];
            int batchSize = x.Shape.D0, hS = hPrev.Shape.D1;

            using var dGNode = AllocateNode(null, new TensorShape(batchSize, 4 * hS), false, clearMemory: false);

            Parallel.For(0, batchSize,
                () => new TensorStorage<float>(hS * 4, false),
                (b, state, arrNode) =>
                {
                    var dGS = dGNode.DataView.AsSpan().Slice(b * 4 * hS, 4 * hS);
                    var gs = gates.DataView.AsReadOnlySpan().Slice(b * 4 * hS, 4 * hS);
                    var f = gs.Slice(0, hS);
                    var i = gs.Slice(hS, hS);
                    var g = gs.Slice(2 * hS, hS);
                    var o = gs.Slice(3 * hS, hS);
                    var dh = hNew.GradView.AsReadOnlySpan().Slice(b * hS, hS);
                    var dc = cNew.GradView.AsSpan().Slice(b * hS, hS);
                    var tS = arrNode.AsSpan();
                    var tanhC = tS.Slice(0, hS);
                    var t1 = tS.Slice(hS, hS);
                    var t2 = tS.Slice(2 * hS, hS);

                    TensorPrimitives.Tanh(cNew.DataView.AsReadOnlySpan().Slice(b * hS, hS), tanhC);
                    TensorPrimitives.Subtract(1f, o, t1);
                    TensorPrimitives.Multiply(o, t1, t1);
                    TensorPrimitives.Multiply(dh, tanhC, t2);
                    TensorPrimitives.Multiply(t2, t1, dGS.Slice(3 * hS, hS));
                    TensorPrimitives.Multiply(tanhC, tanhC, t1);
                    TensorPrimitives.Subtract(1f, t1, t1);
                    TensorPrimitives.Multiply(dh, o, t2);
                    TensorPrimitives.MultiplyAdd(t2, t1, dc, dc);
                    TensorPrimitives.Multiply(g, g, t1);
                    TensorPrimitives.Subtract(1f, t1, t1);
                    TensorPrimitives.Multiply(dc, i, t2);
                    TensorPrimitives.Multiply(t2, t1, dGS.Slice(2 * hS, hS));
                    TensorPrimitives.Subtract(1f, i, t1);
                    TensorPrimitives.Multiply(i, t1, t1);
                    TensorPrimitives.Multiply(dc, g, t2);
                    TensorPrimitives.Multiply(t2, t1, dGS.Slice(hS, hS));
                    TensorPrimitives.Subtract(1f, f, t1);
                    TensorPrimitives.Multiply(f, t1, t1);
                    TensorPrimitives.Multiply(dc, cPrev.DataView.AsReadOnlySpan().Slice(b * hS, hS), t2);
                    TensorPrimitives.Multiply(t2, t1, dGS.Slice(0, hS));

                    if (cPrev.RequiresGrad)
                    {
                        var dcp = cPrev.GradView.AsSpan().Slice(b * hS, hS);
                        TensorPrimitives.MultiplyAdd(dc, f, dcp, dcp);
                    }

                    return arrNode;
                },
                arrNode => arrNode.Dispose());

            if (x.RequiresGrad)
            {
                MatMulAdd_A_BT_Raw(dGNode, false, W, false, x, true);
            }
            if (W.RequiresGrad)
            {
                MatMulAdd_AT_B_Raw(x, false, dGNode, false, W, true);
            }
            if (hPrev.RequiresGrad)
            {
                MatMulAdd_A_BT_Raw(dGNode, false, U, false, hPrev, true);
            }
            if (U.RequiresGrad)
            {
                MatMulAdd_AT_B_Raw(hPrev, false, dGNode, false, U, true);
            }

            if (B.RequiresGrad)
            {
                var dbS = B.GradView.AsSpan();
                var dGS_full = dGNode.DataView.AsReadOnlySpan();
                for (var b = 0; b < batchSize; b++)
                {
                    TensorPrimitives.Add(dbS, dGS_full.Slice(b * 4 * hS, 4 * hS), dbS);
                }
            }
        }

        // ====================================================================
        // REPEAT VECTOR
        // ====================================================================

        public static AutogradNode RepeatVector(ComputationGraph graph, AutogradNode input, int seqLen)
        {
            int batch = input.Shape.D0, hS = input.Shape.D1;
            var output = AllocateNode(graph, new TensorShape(batch, seqLen, hS), input.RequiresGrad, clearMemory: false);

            var inS = input.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();

            for (var b = 0; b < batch; b++)
            {
                var src = inS.Slice(b * hS, hS);
                for (var t = 0; t < seqLen; t++)
                {
                    src.CopyTo(outS.Slice(b * seqLen * hS + t * hS, hS));
                }
            }

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.RepeatVector, output, input, null, seqLen, hS);
            }

            return output;
        }

        public static void RepeatVectorBackward(AutogradNode input, AutogradNode output, int seqLen, int hS)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var batch = input.Shape.D0;
            var iGS = input.GradView.AsSpan();
            var oGS = output.GradView.AsReadOnlySpan();

            for (var b = 0; b < batch; b++)
            {
                var dst = iGS.Slice(b * hS, hS);
                for (var t = 0; t < seqLen; t++)
                {
                    TensorPrimitives.Add(dst, oGS.Slice(b * seqLen * hS + t * hS, hS), dst);
                }
            }
        }

        // ====================================================================
        // GATE SLICE
        // ====================================================================

        public static AutogradNode GateSlice(ComputationGraph graph, AutogradNode gates, int hiddenSize, int gateIndex)
        {
            var output = AllocateNode(graph, new TensorShape(gates.Shape.D0, hiddenSize), gates.RequiresGrad, clearMemory: false);
            int batch = gates.Shape.D0, stride = 4 * hiddenSize, offset = gateIndex * hiddenSize;

            var inS = gates.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();

            for (var b = 0; b < batch; b++)
            {
                inS.Slice(b * stride + offset, hiddenSize).CopyTo(outS.Slice(b * hiddenSize, hiddenSize));
            }

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.GateSlice, output, gates, null, gateIndex, hiddenSize);
            }

            return output;
        }

        public static void GateSliceBackward(AutogradNode gates, AutogradNode output, int hiddenSize, int gateIndex)
        {
            if (!gates.RequiresGrad)
            {
                return;
            }

            int batch = gates.Shape.D0, offset = gateIndex * hiddenSize, stride = 4 * hiddenSize;
            var gGS = gates.GradView.AsSpan();
            var oGS = output.GradView.AsReadOnlySpan();

            for (var b = 0; b < batch; b++)
            {
                var dst = gGS.Slice(b * stride + offset, hiddenSize);
                TensorPrimitives.Add(dst, oGS.Slice(b * hiddenSize, hiddenSize), dst);
            }
        }

        // ====================================================================
        // TIMESTEP SLICE & STACK TIMESTEPS
        // ====================================================================

        public static void TimestepSliceBackward(AutogradNode input, AutogradNode output, int t, int seqLen, int inputSize)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var igS = input.GradView.AsSpan();
            var ogS = output.GradView.AsReadOnlySpan();

            for (var b = 0; b < input.Shape.D0; b++)
            {
                var dst = igS.Slice(b * seqLen * inputSize + t * inputSize, inputSize);
                TensorPrimitives.Add(dst, ogS.Slice(b * inputSize, inputSize), dst);
            }
        }

        public static void StackTimestepsBackward(AutogradNode[] allH, AutogradNode output, int batch, int seqLen, int hiddenSize)
        {
            var ogS = output.GradView.AsReadOnlySpan();

            for (var t = 0; t < seqLen; t++)
            {
                if (!allH[t].RequiresGrad)
                {
                    continue;
                }

                var hGS = allH[t].GradView.AsSpan();

                for (var b = 0; b < batch; b++)
                {
                    var dst = hGS.Slice(b * hiddenSize, hiddenSize);
                    TensorPrimitives.Add(dst, ogS.Slice(b * seqLen * hiddenSize + t * hiddenSize, hiddenSize), dst);
                }
            }
        }
    }
}