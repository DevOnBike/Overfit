// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        /// <summary>
        /// Layer Normalization forward pass.
        ///
        /// Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
        ///
        /// Normalizes over the last dimension (features), independently per token.
        /// Input shape: [N, T, C] or [N, C] — normalization applied over the last dim C.
        ///
        /// Unlike BatchNorm:
        ///   - No running statistics (no train/eval mode difference)
        ///   - Normalizes over features, not batch
        ///   - Standard for Transformers (used in every GPT layer)
        ///
        /// Auxiliary outputs stored on tape:
        ///   mean    [numRows]   — per-token mean, needed for backward
        ///   invStd  [numRows]   — per-token 1/sqrt(var+eps), needed for backward
        /// </summary>
        public static AutogradNode LayerNorm(
            ComputationGraph graph,
            AutogradNode input,
            AutogradNode gamma,
            AutogradNode beta,
            float eps = 1e-5f)
        {
            // Flatten to [numRows, C] for uniform handling of [N,C] and [N,T,C].
            // Normalise over last dimension — works for [N,C] and [N,T,C].
            var C = input.Shape[input.Shape.Rank - 1];
            var numRows = input.Shape.Size / C;

            var requiresGrad = input.RequiresGrad || gamma.RequiresGrad || beta.RequiresGrad;
            var output = AllocateNode(graph, input.Shape, requiresGrad, clearMemory: false);

            // Per-row statistics — GraphAuxiliary, disposed by graph.Reset().
            var mean = AllocateNode(graph, new TensorShape(numRows), requiresGrad: false, clearMemory: false);
            var invStd = AllocateNode(graph, new TensorShape(numRows), requiresGrad: false, clearMemory: false);

            var inS = input.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();
            var gammaS = gamma.DataView.AsReadOnlySpan();
            var betaS = beta.DataView.AsReadOnlySpan();
            var meanS = mean.DataView.AsSpan();
            var invStdS = invStd.DataView.AsSpan();

            for (var r = 0; r < numRows; r++)
            {
                var row = inS.Slice(r * C, C);
                var outRow = outS.Slice(r * C, C);

                // Mean
                var mu = TensorPrimitives.Sum(row) / C;
                meanS[r] = mu;

                // Variance (biased)
                var variance = 0f;
                for (var i = 0; i < C; i++)
                {
                    var diff = row[i] - mu;
                    variance += diff * diff;
                }
                variance /= C;

                var inv = 1f / MathF.Sqrt(variance + eps);
                invStdS[r] = inv;

                // Normalize + scale + shift
                for (var i = 0; i < C; i++)
                {
                    outRow[i] = gammaS[i] * ((row[i] - mu) * inv) + betaS[i];
                }
            }

            if (requiresGrad)
            {
                graph?.Record(
                    OpCode.LayerNorm, output, input,
                    c0: gamma, c1: beta, c2: mean, c3: invStd,
                    contextCount: 4);
            }
            else
            {
                mean.Dispose();
                invStd.Dispose();
            }

            return output;
        }

        /// <summary>
        /// Layer Normalization backward pass.
        ///
        /// Gradients for input, gamma, beta — computed analytically.
        /// Derivation follows the standard LN backward:
        ///   dL/dx = (1/C) * invStd * (C * dL/dy_hat - sum(dL/dy_hat) - y_hat * sum(dL/dy_hat * y_hat))
        ///   where y_hat = (x - mean) * invStd  (normalised, before gamma/beta)
        /// </summary>
        public static void LayerNormBackward(
            AutogradNode input,
            AutogradNode output,
            AutogradNode gamma,
            AutogradNode beta,
            AutogradNode mean,
            AutogradNode invStd)
        {
            // Normalise over last dimension — works for [N,C] and [N,T,C].
            var C = input.Shape[input.Shape.Rank - 1];
            var numRows = input.Shape.Size / C;

            var inS = input.DataView.AsReadOnlySpan();
            var dOutS = output.GradView.AsReadOnlySpan();
            var gammaS = gamma.DataView.AsReadOnlySpan();
            var meanS = mean.DataView.AsReadOnlySpan();
            var invStdS = invStd.DataView.AsReadOnlySpan();

            for (var r = 0; r < numRows; r++)
            {
                var row = inS.Slice(r * C, C);
                var dOut = dOutS.Slice(r * C, C);

                var mu = meanS[r];
                var inv = invStdS[r];

                // Compute normalised values y_hat = (x - mu) * invStd
                // and accumulate dBeta, dGamma
                var sumDOut = 0f;
                var sumDOutYh = 0f;

                for (var i = 0; i < C; i++)
                {
                    var yHat = (row[i] - mu) * inv;
                    sumDOut += dOut[i] * gammaS[i];
                    sumDOutYh += dOut[i] * gammaS[i] * yHat;

                    // dBeta accumulation
                    if (beta.RequiresGrad)
                    {
                        beta.GradView.AsSpan()[i] += dOut[i];
                    }

                    // dGamma accumulation
                    if (gamma.RequiresGrad)
                    {
                        gamma.GradView.AsSpan()[i] += dOut[i] * yHat;
                    }
                }

                // dInput
                if (input.RequiresGrad)
                {
                    var dIn = input.GradView.AsSpan().Slice(r * C, C);
                    var scale = inv / C;

                    for (var i = 0; i < C; i++)
                    {
                        var yHat = (row[i] - mu) * inv;
                        dIn[i] += scale * (C * dOut[i] * gammaS[i] - sumDOut - yHat * sumDOutYh);
                    }
                }
            }
        }
    }
}