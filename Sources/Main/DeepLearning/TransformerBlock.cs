// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Parameters;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// Transformer Block (Pre-Layer Norm variant, as in GPT-2 / modern GPT).
    ///
    /// Architecture:
    ///   x = x + Attention(LayerNorm1(x))
    ///   x = x + FFN(LayerNorm2(x))
    ///
    /// Pre-LN (normalize before sublayer) is more training-stable than the
    /// original Post-LN used in "Attention Is All You Need". GPT-2 and later
    /// models use Pre-LN. GPT-1 used Post-LN — both are supported via the
    /// <see cref="PreLayerNorm"/> property.
    ///
    /// Parameters:
    ///   LayerNorm1: gamma, beta  [dModel]
    ///   MultiHeadAttention: Wq, Wk, Wv, Wo, Bo  (4*dModel² + dModel)
    ///   LayerNorm2: gamma, beta  [dModel]
    ///   FeedForward: W1, b1, W2, b2  (dModel*dFF*2 + dFF + dModel)
    ///
    /// For GPT-1: dModel=768, nHeads=12, dFF=3072 (=4*dModel).
    /// Total params per block: ~7M.
    /// </summary>
    public sealed class TransformerBlock : IModule
    {
        private readonly int _dModel;

        public TransformerBlock(
            int dModel,
            int nHeads,
            int dFF,
            bool causalMask    = true,
            bool preLayerNorm  = true,
            float lnEps        = 1e-5f)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dModel);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(nHeads);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dFF);

            _dModel       = dModel;
            PreLayerNorm  = preLayerNorm;

            Norm1     = new LayerNormLayer(dModel, lnEps);
            Attention = new MultiHeadAttentionLayer(dModel, nHeads, causalMask);
            Norm2     = new LayerNormLayer(dModel, lnEps);
            FFN       = new FeedForwardLayer(dModel, dFF);
        }

        /// <summary>Pre-LayerNorm (true) or Post-LayerNorm (false, original GPT-1).</summary>
        public bool PreLayerNorm { get; }

        public LayerNormLayer         Norm1     { get; }
        public MultiHeadAttentionLayer Attention { get; }
        public LayerNormLayer         Norm2     { get; }
        public FeedForwardLayer       FFN       { get; }

        public bool IsTraining { get; private set; } = true;

        public void Train()
        {
            IsTraining = true;
            Norm1.Train(); Attention.Train(); Norm2.Train(); FFN.Train();
        }

        public void Eval()
        {
            IsTraining = false;
            Norm1.Eval(); Attention.Eval(); Norm2.Eval(); FFN.Eval();
        }

        /// <summary>
        /// Forward: [B, T, dModel] → [B, T, dModel].
        ///
        /// Pre-LN:
        ///   x = x + Attention(LN1(x))
        ///   x = x + FFN(LN2(x))
        ///
        /// Post-LN (original GPT-1):
        ///   x = LN1(x + Attention(x))
        ///   x = LN2(x + FFN(x))
        /// </summary>
        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            if (input.Shape.D2 != _dModel)
            {
                throw new ArgumentException(
                    $"Expected d_model={_dModel}, got {input.Shape.D2}. Input: {input.Shape}");
            }

            if (PreLayerNorm)
            {
                return ForwardPreLN(graph, input);
            }

            return ForwardPostLN(graph, input);
        }

        private AutogradNode ForwardPreLN(ComputationGraph graph, AutogradNode x)
        {
            // x = x + Attention(LN1(x))
            var ln1Out  = Norm1.Forward(graph, x);
            var attnOut = Attention.Forward(graph, ln1Out);
            var x2      = Residual(x, attnOut);

            // x = x + FFN(LN2(x))
            var ln2Out  = Norm2.Forward(graph, x2);
            var ffnOut  = FFN.Forward(graph, ln2Out);
            return Residual(x2, ffnOut);
        }

        private AutogradNode ForwardPostLN(ComputationGraph graph, AutogradNode x)
        {
            // x = LN1(x + Attention(x))
            var attnOut = Attention.Forward(graph, x);
            var sum1    = Residual(x, attnOut);
            var x2      = Norm1.Forward(graph, sum1);

            // x = LN2(x + FFN(x))
            var ffnOut = FFN.Forward(graph, x2);
            var sum2   = Residual(x2, ffnOut);
            return Norm2.Forward(graph, sum2);
        }

        /// <summary>
        /// Element-wise residual addition: out = x + sublayer_out.
        /// Not tracked on tape (structural add without grad — gradient flows through
        /// both paths via the chain rule naturally since both x and sublayer_out
        /// are on the tape through their own ops).
        /// </summary>
        private static AutogradNode Residual(AutogradNode x, AutogradNode sublayerOut)
        {
            var size    = x.DataView.Size;
            var storage = new TensorStorage<float>(size, clearMemory: false);
            var xS      = x.DataView.AsReadOnlySpan();
            var sS      = sublayerOut.DataView.AsReadOnlySpan();
            var outS    = storage.AsSpan();

            TensorPrimitives.Add(xS, sS, outS);

            // requiresGrad: true if either input needs grad so backward can propagate
            var rg   = x.RequiresGrad || sublayerOut.RequiresGrad;
            var node = new AutogradNode(storage, x.Shape, requiresGrad: rg);
            return node;
        }

        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
            => throw new NotSupportedException(
                "TransformerBlock.ForwardInference(Span) is not supported. " +
                "Use Forward(ComputationGraph, AutogradNode).");

        public IEnumerable<AutogradNode> Parameters()
        {
            foreach (var p in Norm1.Parameters())
            {
                yield return p;
            }
            foreach (var p in Attention.Parameters())
            {
                yield return p;
            }
            foreach (var p in Norm2.Parameters())
            {
                yield return p;
            }
            foreach (var p in FFN.Parameters())
            {
                yield return p;
            }
        }

        public IEnumerable<Parameter> TrainableParameters()
        {
            foreach (var p in Norm1.TrainableParameters())
            {
                yield return p;
            }
            foreach (var p in Attention.TrainableParameters())
            {
                yield return p;
            }
            foreach (var p in Norm2.TrainableParameters())
            {
                yield return p;
            }
            foreach (var p in FFN.TrainableParameters())
            {
                yield return p;
            }
        }

        public void InvalidateParameterCaches()
        {
            Norm1.InvalidateParameterCaches();
            Attention.InvalidateParameterCaches();
            Norm2.InvalidateParameterCaches();
            FFN.InvalidateParameterCaches();
        }

        public void Save(BinaryWriter bw)
        {
            Norm1.Save(bw); Attention.Save(bw); Norm2.Save(bw); FFN.Save(bw);
        }

        public void Load(BinaryReader br)
        {
            Norm1.Load(br); Attention.Load(br); Norm2.Load(br); FFN.Load(br);
        }

        public void Dispose()
        {
            Norm1.Dispose(); Attention.Dispose(); Norm2.Dispose(); FFN.Dispose();
        }
    }
}
