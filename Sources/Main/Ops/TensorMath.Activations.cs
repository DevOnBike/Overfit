// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // ====================================================================
        // RELU
        // ====================================================================

        public static AutogradNode ReLU(ComputationGraph graph, AutogradNode input)
        {
            var res = AllocateLike(input, false);
            TensorPrimitives.Max(input.DataView.AsReadOnlySpan(), 0f, res.GetView().AsSpan());

            var output = new AutogradNode(res, input.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.ReLU, output, input);
            }

            return output;
        }

        public static void ReluBackward(AutogradNode input, AutogradNode output)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var inS = input.DataView.AsReadOnlySpan();
            var goS = output.GradView.AsReadOnlySpan();
            var giS = input.GradView.AsSpan();

            var i = 0;

            if (Vector.IsHardwareAccelerated)
            {
                var vCount = Vector<float>.Count;
                var vZero = Vector<float>.Zero;

                for (; i <= inS.Length - vCount; i += vCount)
                {
                    var vIn = new Vector<float>(inS.Slice(i));
                    var vMask = Vector.GreaterThan(vIn, vZero);
                    var vGradToPass = Vector.ConditionalSelect(vMask, new Vector<float>(goS.Slice(i)), vZero);
                    (new Vector<float>(giS.Slice(i)) + vGradToPass).CopyTo(giS.Slice(i));
                }
            }

            for (; i < inS.Length; i++)
            {
                if (inS[i] > 0f)
                {
                    giS[i] += goS[i];
                }
            }
        }

        // ====================================================================
        // SIGMOID
        // ====================================================================

        public static AutogradNode Sigmoid(ComputationGraph graph, AutogradNode input)
        {
            var res = AllocateLike(input, false);
            TensorPrimitives.Sigmoid(input.DataView.AsReadOnlySpan(), res.GetView().AsSpan());

            var output = new AutogradNode(res, input.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.Sigmoid, output, input);
            }

            return output;
        }

        public static void SigmoidBackward(AutogradNode input, AutogradNode output)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var outS = output.DataView.AsReadOnlySpan();
            var ogS = output.GradView.AsReadOnlySpan();
            var igS = input.GradView.AsSpan();

            Span<float> buffer = stackalloc float[StackAllocThreshold];

            for (var i = 0; i < igS.Length; i += StackAllocThreshold)
            {
                var c = Math.Min(StackAllocThreshold, igS.Length - i);
                var b = buffer.Slice(0, c);
                var o = outS.Slice(i, c);

                TensorPrimitives.Subtract(1f, o, b);
                TensorPrimitives.Multiply(o, b, b);
                TensorPrimitives.MultiplyAdd(ogS.Slice(i, c), b, igS.Slice(i, c), igS.Slice(i, c));
            }
        }

        // ====================================================================
        // TANH
        // ====================================================================

        public static AutogradNode Tanh(ComputationGraph graph, AutogradNode input)
        {
            var res = AllocateLike(input, false);
            TensorPrimitives.Tanh(input.DataView.AsReadOnlySpan(), res.GetView().AsSpan());

            var output = new AutogradNode(res, input.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.Tanh, output, input);
            }

            return output;
        }

        public static void TanhBackward(AutogradNode input, AutogradNode output)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var outS = output.DataView.AsReadOnlySpan();
            var ogS = output.GradView.AsReadOnlySpan();
            var igS = input.GradView.AsSpan();

            Span<float> buffer = stackalloc float[StackAllocThreshold];

            for (var i = 0; i < igS.Length; i += StackAllocThreshold)
            {
                var c = Math.Min(StackAllocThreshold, igS.Length - i);
                var b = buffer.Slice(0, c);
                var o = outS.Slice(i, c);

                TensorPrimitives.Multiply(o, o, b);
                TensorPrimitives.Subtract(1f, b, b);
                TensorPrimitives.MultiplyAdd(ogS.Slice(i, c), b, igS.Slice(i, c), igS.Slice(i, c));
            }
        }

        // ====================================================================
        // DROPOUT
        // ====================================================================

        public static AutogradNode Dropout(ComputationGraph graph, AutogradNode input, float probability, bool isTraining)
        {
            var res = AllocateLike(input, false);

            if (!isTraining || probability <= 0f)
            {
                input.DataView.AsReadOnlySpan().CopyTo(res.GetView().AsSpan());
                return new AutogradNode(res, input.RequiresGrad);
            }

            var mask = AllocateLike(input, false);
            var scale = 1f / (1f - probability);

            var inS = input.DataView.AsReadOnlySpan();
            var outS = res.GetView().AsSpan();
            var maskS = mask.GetView().AsSpan();

            for (var i = 0; i < inS.Length; i++)
            {
                var keep = Random.Shared.NextSingle() >= probability;
                maskS[i] = keep ? scale : 0f;
                outS[i] = inS[i] * maskS[i];
            }

            var output = new AutogradNode(res, input.RequiresGrad);
            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.Dropout, output, input, new AutogradNode(mask));
            }
            else
            {
                mask.Dispose();
            }

            return output;
        }

        public static void DropoutBackward(AutogradNode input, AutogradNode mask, AutogradNode output)
        {
            TensorPrimitives.MultiplyAdd(output.GradView.AsReadOnlySpan(), mask.DataView.AsReadOnlySpan(), input.GradView.AsSpan(), input.GradView.AsSpan());
        }
    }
}
