// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Threading.Tasks;
using DevOnBike.Overfit.Intrinsics;
using DevOnBike.Overfit.Kernels;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Runtime;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Autograd
{
    public sealed partial class ComputationGraph
    {
        // ─────────────────────────────────────────────────────────────────
        // Instance methods — use when graph is guaranteed non-null (training)
        // ─────────────────────────────────────────────────────────────────

        /// <summary>
        /// Fully-connected linear transformation: output = input @ W + b
        /// PR5-7a: implementation moved from TensorMath.Linear.
        /// </summary>
        public AutogradNode Linear(
            AutogradNode input,
            AutogradNode weights,
            AutogradNode bias)
        {
            int N = input.Shape.D0, K = input.Shape.D1, M = weights.Shape.D1;
            var requiresGrad = input.RequiresGrad || weights.RequiresGrad || bias.RequiresGrad;
            var output = CreateTemporary(new TensorShape(N, M), requiresGrad, clearMemory: false);

            var outS  = output.DataView.AsSpan();
            var inS   = input.DataView.AsReadOnlySpan();
            var wS    = weights.DataView.AsReadOnlySpan();
            var biasS = bias.DataView.AsReadOnlySpan();

            LinearKernels.InitWithBias(outS, biasS, N, M);

            if ((long)N * K * M < LinearKernels.ForwardBatchedThreshold)
            {
                LinearKernels.ForwardBatched(inS, wS, outS, N, K, M);
            }
            else
            {
                Parallel.For(0, N, OverfitParallel.Options, i =>
                {
                    var rC  = output.DataView.AsSpan().Slice(i * M, M);
                    var rA  = input.DataView.AsReadOnlySpan().Slice(i * K, K);
                    var wS2 = weights.DataView.AsReadOnlySpan();

                    for (var k = 0; k < K; k++)
                    {
                        var aVal = rA[k];
                        if (aVal != 0f)
                        {
                            Intrinsics.Simd.MulAdd(wS2.Slice(k * M, M), aVal, rC);
                        }
                    }
                });
            }

            if (requiresGrad)
            {
                Record(OpCode.Linear, output, input, weights, c0: bias, contextCount: 1);
            }

            return output;
        }

        public AutogradNode Add(AutogradNode left, AutogradNode right)
            => TensorMath.Add(this, left, right);

        public AutogradNode AddBias(AutogradNode input, AutogradNode bias)
            => TensorMath.AddBias(this, input, bias);

        public AutogradNode Subtract(AutogradNode left, AutogradNode right)
            => TensorMath.Subtract(this, left, right);

        public AutogradNode MatMul(AutogradNode left, AutogradNode right)
            => TensorMath.MatMul(this, left, right);

        public AutogradNode Reshape(AutogradNode input, params int[] newShape)
            => TensorMath.Reshape(this, input, newShape);

        // ─────────────────────────────────────────────────────────────────
        // Null-safe static ops — use in layer Forward() where graph may be
        // null (inference path via model.Forward(null, input)).
        //
        // Pattern: ComputationGraph.XOp(graph, ...) replaces graph.X(...)
        // so that null graph is handled identically to old TensorMath.X(null, ...)
        // ─────────────────────────────────────────────────────────────────

        internal static AutogradNode LinearOp(
            ComputationGraph? graph,
            AutogradNode input,
            AutogradNode weights,
            AutogradNode bias)
        {
            if (graph != null)
            {
                return graph.Linear(input, weights, bias);
            }

            // Inference path (graph == null): standalone allocation, no tape.
            int N = input.Shape.D0, K = input.Shape.D1, M = weights.Shape.D1;
            var storage = new TensorStorage<float>(N * M, clearMemory: false);
            var output = AutogradNode.CreateBorrowed(storage, new TensorShape(N, M));

            LinearKernels.InitWithBias(
                output.DataView.AsSpan(),
                bias.DataView.AsReadOnlySpan(), N, M);

            LinearKernels.ForwardBatched(
                input.DataView.AsReadOnlySpan(),
                weights.DataView.AsReadOnlySpan(),
                output.DataView.AsSpan(), N, K, M);

            return output;
        }

        internal static AutogradNode ReluOp(
            ComputationGraph? graph,
            AutogradNode input)
        {
            if (graph != null)
            {
                return graph.Relu(input);
            }

            // Inference path: standalone allocation.
            var storage = new TensorStorage<float>(input.Shape.Size, clearMemory: false);
            var output = AutogradNode.CreateBorrowed(storage, input.Shape);
            TensorKernels.Relu(input.DataView.AsReadOnlySpan(), output.DataView.AsSpan());
            return output;
        }

        internal static AutogradNode Conv2DOp(
            ComputationGraph? graph,
            AutogradNode input,
            AutogradNode weights,
            int inC, int outC, int h, int w, int k)
            => TensorMath.Conv2D(graph!, input, weights, inC, outC, h, w, k);

        internal static AutogradNode MaxPool2DOp(
            ComputationGraph? graph,
            AutogradNode input,
            int channels, int h, int w, int pool)
            => TensorMath.MaxPool2D(graph!, input, channels, h, w, pool);

        internal static AutogradNode GlobalAveragePool2DOp(
            ComputationGraph? graph,
            AutogradNode input,
            int channels, int h, int w)
            => TensorMath.GlobalAveragePool2D(graph!, input, channels, h, w);

        internal static AutogradNode BatchNorm1DOp(
            ComputationGraph? graph,
            AutogradNode input,
            AutogradNode gamma,
            AutogradNode beta,
            TensorStorage<float> runningMean,
            TensorStorage<float> runningVar,
            float momentum,
            float eps,
            bool isTraining)
            => TensorMath.BatchNorm1D(graph!, input, gamma, beta,
                                      runningMean, runningVar, momentum, eps, isTraining);

        /// <summary>
        /// Adds source into target in-place. When graph is non-null, records tape for backward.
        /// When graph is null (inference), performs the addition without recording.
        /// Replaces <c>graph.AddInPlace(target, source)</c> in layer Forward() methods.
        /// </summary>
        internal static AutogradNode AddOp(
            ComputationGraph? graph,
            AutogradNode target,
            AutogradNode source)
        {
            if (graph != null)
            {
                return graph.AddInPlace(target, source);
            }

            // Inference: add in-place without tape
            TensorPrimitives.Add(
                target.DataView.AsSpan(),
                source.DataView.AsReadOnlySpan(),
                target.DataView.AsSpan());

            return target;
        }
    }
}
