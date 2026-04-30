// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.Autograd
{
    public sealed partial class ComputationGraph
    {
        // ─────────────────────────────────────────────────────────────────
        // Instance methods — use when graph is guaranteed non-null (training)
        // ─────────────────────────────────────────────────────────────────

        public AutogradNode Linear(
            AutogradNode input,
            AutogradNode weights,
            AutogradNode bias)
            => TensorMath.Linear(this, input, weights, bias);

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
            => TensorMath.Linear(graph!, input, weights, bias);

        internal static AutogradNode ReluOp(
            ComputationGraph? graph,
            AutogradNode input)
            => TensorMath.ReLU(graph!, input);

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
