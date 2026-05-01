// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Layer Normalization — normalises over the last dimension per token.
    ///
    /// Used in every Transformer block (before or after Attention and FFN).
    /// Unlike BatchNorm: no running statistics, no train/eval mode difference,
    /// normalises over features not batch.
    ///
    /// Parameters:
    ///   Gamma — learnable scale [normalizedShape], initialised to 1.
    ///   Beta  — learnable shift [normalizedShape], initialised to 0.
    ///
    /// Input:  [N, T, C] or [N, C] where C = normalizedShape.
    /// Output: same shape as input.
    /// </summary>
    public sealed class LayerNormLayer : IModule
    {
        private readonly int _normalizedShape;
        private readonly float _eps;

        private AutogradNode? _gammaNode;
        private AutogradNode? _betaNode;

        public LayerNormLayer(int normalizedShape, float eps = 1e-5f)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(normalizedShape);

            _normalizedShape = normalizedShape;
            _eps             = eps;

            Gamma = new Parameter(new TensorShape(normalizedShape), requiresGrad: true, clearData: false);
            Gamma.DataSpan.Fill(1f);  // scale = 1 by default

            Beta = new Parameter(new TensorShape(normalizedShape), requiresGrad: true, clearData: true);
            // bias = 0 by default (clearData: true)
        }

        /// <summary>Learnable per-feature scale (gamma), shape [normalizedShape].</summary>
        public Parameter Gamma { get; }

        /// <summary>Learnable per-feature shift (beta), shape [normalizedShape].</summary>
        public Parameter Beta { get; }

        public float Eps => _eps;

        public bool IsTraining { get; private set; } = true;

        public void Train() => IsTraining = true;

        public void Eval() => IsTraining = false;

        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            // Validate last dimension matches normalizedShape
            var lastDim = input.Shape[input.Shape.Rank - 1];
            if (lastDim != _normalizedShape)
            {
                throw new ArgumentException(
                    $"LayerNorm expects last dim {_normalizedShape}, got {lastDim}. " +
                    $"Input shape: {input.Shape}");
            }

            // Cached view nodes — no using, tape holds references.
            _gammaNode ??= Gamma.AsNode();
            _betaNode  ??= Beta.AsNode();

            return TensorMath.LayerNorm(graph, input, _gammaNode, _betaNode, _eps);
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            // Single-sample inference — reuse Forward logic without tape.
            var shape = new TensorShape(_normalizedShape);
            var C     = _normalizedShape;
            var numRows = input.Length / C;

            for (var r = 0; r < numRows; r++)
            {
                var inRow  = input.Slice(r * C, C);
                var outRow = output.Slice(r * C, C);
                var mu = 0f;

                for (var i = 0; i < C; i++) mu += inRow[i];
                mu /= C;

                var variance = 0f;
                for (var i = 0; i < C; i++) { var d = inRow[i] - mu; variance += d * d; }
                variance /= C;

                var inv = 1f / MathF.Sqrt(variance + _eps);
                var gammaS = Gamma.DataReadOnlySpan;
                var betaS  = Beta.DataReadOnlySpan;

                for (var i = 0; i < C; i++)
                {
                    outRow[i] = gammaS[i] * ((inRow[i] - mu) * inv) + betaS[i];
                }
            }
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            yield return Gamma.AsNode();
            yield return Beta.AsNode();
        }

        public IEnumerable<Parameter> TrainableParameters()
        {
            yield return Gamma;
            yield return Beta;
        }

        public void InvalidateParameterCaches()
        {
            _gammaNode = null;
            _betaNode  = null;
        }

        public void Save(BinaryWriter bw)
        {
            Gamma.Save(bw);
            Beta.Save(bw);
        }

        public void Load(BinaryReader br)
        {
            Gamma.Load(br);
            Beta.Load(br);
        }

        public void Dispose()
        {
            _gammaNode?.Dispose();
            _betaNode?.Dispose();
            Gamma.Dispose();
            Beta.Dispose();
        }
    }
}
