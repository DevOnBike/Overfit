// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // ====================================================================
        // SOFTMAX CROSS ENTROPY
        // ====================================================================

        public static AutogradNode SoftmaxCrossEntropy(ComputationGraph graph, AutogradNode logits, AutogradNode target)
        {
            int rows = logits.DataView.GetDim(0), cols = logits.DataView.GetDim(1);
            var total = 0f;
            var probs = new FastTensor<float>(rows, cols, false);

            var inS = logits.DataView.AsReadOnlySpan();
            var targetS = target.DataView.AsReadOnlySpan();
            var probS = probs.GetView().AsSpan();

            for (var r = 0; r < rows; r++)
            {
                var pR = probS.Slice(r * cols, cols);
                TensorPrimitives.SoftMax(inS.Slice(r * cols, cols), pR);
                var tR = targetS.Slice(r * cols, cols);
                for (var c = 0; c < cols; c++)
                {
                    if (tR[c] > 0.5f)
                    {
                        total -= MathF.Log(MathF.Max(pR[c], 1e-7f));
                    }
                }
            }

            var resTensor = new FastTensor<float>(1, false);
            resTensor.GetView().AsSpan()[0] = total / rows;
            var output = new AutogradNode(resTensor, logits.RequiresGrad);

            if (logits.RequiresGrad)
            {
                graph?.Record(OpCode.SoftmaxCrossEntropy, output, logits, target, nodeContext: [new AutogradNode(probs)]);
            }
            else
            {
                probs.Dispose();
            }

            return output;
        }

        public static void SoftmaxCrossEntropyBackward(AutogradNode logits, AutogradNode target, AutogradNode output, AutogradNode probsNode)
        {
            int R = logits.DataView.GetDim(0), C = logits.DataView.GetDim(1);
            var s = output.GradView.AsReadOnlySpan()[0] / R;

            var probS = probsNode.DataView.AsReadOnlySpan();
            var tarS = target.DataView.AsReadOnlySpan();
            var logS = logits.GradView.AsSpan();

            for (var r = 0; r < R; r++)
            {
                var pS = probS.Slice(r * C, C);
                var tS = tarS.Slice(r * C, C);
                var gS = logS.Slice(r * C, C);
                TensorPrimitives.MultiplyAdd(pS, s, gS, gS);
                TensorPrimitives.MultiplyAdd(tS, -s, gS, gS);
            }
        }

        // ====================================================================
        // MSE LOSS
        // ====================================================================

        public static AutogradNode MSELoss(ComputationGraph graph, AutogradNode prediction, AutogradNode target)
        {
            var sz = prediction.DataView.Size;
            float mse;

            using (var diffBuf = new PooledBuffer<float>(sz, false))
            {
                var dS = diffBuf.Span;
                TensorPrimitives.Subtract(prediction.DataView.AsReadOnlySpan(), target.DataView.AsReadOnlySpan(), dS);
                mse = TensorPrimitives.Dot(dS, dS) / sz;
            }

            var resTensor = new FastTensor<float>(1, false);
            resTensor.GetView().AsSpan()[0] = mse;

            var output = new AutogradNode(resTensor, prediction.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.MseLoss, output, prediction, target);
            }

            return output;
        }

        public static void MSELossBackward(AutogradNode p, AutogradNode t, AutogradNode o)
        {
            var f = o.GradView.AsReadOnlySpan()[0] * (2f / p.DataView.Size);
            TensorPrimitives.MultiplyAdd(p.DataView.AsReadOnlySpan(), f, p.GradView.AsSpan(), p.GradView.AsSpan());
            TensorPrimitives.MultiplyAdd(t.DataView.AsReadOnlySpan(), -f, p.GradView.AsSpan(), p.GradView.AsSpan());
        }

        // ====================================================================
        // DIRECTIONAL LOSS
        // ====================================================================

        public static AutogradNode DirectionalLoss(ComputationGraph graph, AutogradNode prediction, AutogradNode target, float gamma = 10f)
        {
            var sz = prediction.DataView.Size;
            float loss;

            using (var tempBuf = new PooledBuffer<float>(sz, false))
            {
                var s = tempBuf.Span;
                TensorPrimitives.Subtract(prediction.DataView.AsReadOnlySpan(), target.DataView.AsReadOnlySpan(), s);
                var mse = TensorPrimitives.SumOfSquares(s);
                TensorPrimitives.Multiply(prediction.DataView.AsReadOnlySpan(), target.DataView.AsReadOnlySpan(), s);
                TensorPrimitives.Min(s, 0f, s);
                loss = (mse + TensorPrimitives.Sum(s) * -gamma) / sz;
            }

            var resTensor = new FastTensor<float>(1, false);
            resTensor.GetView().AsSpan()[0] = loss;

            var output = new AutogradNode(resTensor, prediction.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.DirectionalLoss, output, prediction, target, BitConverter.SingleToInt32Bits(gamma));
            }

            return output;
        }

        public static void DirectionalLossBackward(AutogradNode p, AutogradNode t, AutogradNode o, float gamma)
        {
            var s = o.GradView.AsReadOnlySpan()[0] / p.DataView.Size;
            var pG = p.GradView.AsSpan();
            var pD = p.DataView.AsReadOnlySpan();
            var tD = t.DataView.AsReadOnlySpan();
            var i = 0;

            if (Vector.IsHardwareAccelerated)
            {
                var vS = Vector<float>.Count;
                for (; i <= pD.Length - vS; i += vS)
                {
                    var vP = new Vector<float>(pD.Slice(i));
                    var vT = new Vector<float>(tD.Slice(i));
                    (new Vector<float>(pG.Slice(i)) + (vP - vT) * (2f * s) + Vector.ConditionalSelect(Vector.LessThan(vP * vT, Vector<float>.Zero), -vT * (gamma * s), Vector<float>.Zero)).CopyTo(pG.Slice(i));
                }
            }

            for (; i < pD.Length; i++)
            {
                pG[i] += 2f * (pD[i] - tD[i]) * s + (pD[i] * tD[i] < 0f ? -tD[i] * (gamma * s) : 0f);
            }
        }
    }
}
