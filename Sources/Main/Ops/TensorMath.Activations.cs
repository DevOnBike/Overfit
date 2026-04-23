// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics;
using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        // ====================================================================
        // RELU
        // ====================================================================

        public static AutogradNode ReLU(ComputationGraph graph, AutogradNode input)
        {
            var output = AllocateNode(graph, input.Shape, input.RequiresGrad, clearMemory: false);

            // Używamy naszego zoptymalizowanego Kernela!
            TensorKernels.Relu(input.DataView.AsReadOnlySpan(), output.DataView.AsSpan());

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.ReLU, output, input);
            }

            return output;
        }

        public static void ReluBackward(AutogradNode input, AutogradNode output)
        {
            if (!input.RequiresGrad) return;

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
                if (inS[i] > 0f) giS[i] += goS[i];
            }
        }

        // ====================================================================
        // SIGMOID
        // ====================================================================

        public static AutogradNode Sigmoid(ComputationGraph graph, AutogradNode input)
        {
            var output = AllocateNode(graph, input.Shape, input.RequiresGrad, clearMemory: false);
            TensorPrimitives.Sigmoid(input.DataView.AsReadOnlySpan(), output.DataView.AsSpan());

            if (output.RequiresGrad) graph?.Record(OpCode.Sigmoid, output, input);
            return output;
        }

        public static void SigmoidBackward(AutogradNode input, AutogradNode output)
        {
            if (!input.RequiresGrad) return;

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
            var output = AllocateNode(graph, input.Shape, input.RequiresGrad, clearMemory: false);
            TensorPrimitives.Tanh(input.DataView.AsReadOnlySpan(), output.DataView.AsSpan());

            if (output.RequiresGrad) graph?.Record(OpCode.Tanh, output, input);
            return output;
        }

        public static void TanhBackward(AutogradNode input, AutogradNode output)
        {
            if (!input.RequiresGrad) return;

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
            var output = AllocateNode(graph, input.Shape, input.RequiresGrad, clearMemory: false);

            if (!isTraining || probability <= 0f)
            {
                input.DataView.AsReadOnlySpan().CopyTo(output.DataView.AsSpan());
                return output;
            }

            // Mask staje się pełnoprawnym węzłem z własną pamięcią na taśmie
            var mask = AllocateNode(graph, input.Shape, false, clearMemory: false);
            var scale = 1f / (1f - probability);

            var inS = input.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();
            var maskS = mask.DataView.AsSpan();

            for (var i = 0; i < inS.Length; i++)
            {
                var keep = Random.Shared.NextSingle() >= probability;
                maskS[i] = keep ? scale : 0f;
                outS[i] = inS[i] * maskS[i];
            }

            if (output.RequiresGrad)
            {
                graph?.Record(OpCode.Dropout, output, input, mask); // Mask jako node context B
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