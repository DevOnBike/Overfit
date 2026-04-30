// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Element-wise addition of two tensors. Used for residual / skip connections.
    ///
    /// In DAG inference, the runtime manages both inputs explicitly:
    /// <see cref="ForwardInference(ReadOnlySpan{float}, ReadOnlySpan{float}, Span{float})"/>
    /// is the primary path. <see cref="ForwardInference(ReadOnlySpan{float}, Span{float})"/>
    /// (single-input IModule contract) is not meaningful for Add — it copies input to output
    /// as a no-op and is only included for interface compliance.
    ///
    /// In training graphs: <see cref="Forward(ComputationGraph, AutogradNode, AutogradNode)"/>
    /// performs the two-input forward pass with tape recording.
    /// </summary>
    public sealed class OnnxAddLayer : IModule
    {
        public bool IsTraining { get; private set; } = true;

        public void Train()
        {
            IsTraining = true;
        }

        public void Eval()
        {
            IsTraining = false;
        }

        // ── Two-input interface (primary for DAG) ──────────────────────────

        /// <summary>
        /// Computes output = left + right (element-wise) without autograd.
        /// Both spans must have equal length.
        /// </summary>
        public void ForwardInference(
            ReadOnlySpan<float> left,
            ReadOnlySpan<float> right,
            Span<float> output)
        {
            TensorPrimitives.Add(left, right, output);
        }

        /// <summary>
        /// Computes output = input + right node (element-wise) with autograd recording.
        /// </summary>
        public AutogradNode Forward(
            ComputationGraph graph,
            AutogradNode left,
            AutogradNode right)
        {
            return TensorMath.Add(graph, left, right);
        }

        // ── IModule single-input contract (no-op for Add) ─────────────────

        /// <summary>Single-input IModule path: copies input to output (no-op for Add).</summary>
        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            return input;
        }

        /// <summary>Single-input inference path: copies input to output (no-op for Add).</summary>
        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            input.CopyTo(output);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            return [];
        }

        public void InvalidateParameterCaches() { }

        public void Save(BinaryWriter bw) { }

        public void Load(BinaryReader br) { }

        public void Dispose() { }
    }
}
