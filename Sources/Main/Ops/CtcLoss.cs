// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Ops
{
    /// <summary>
    /// Connectionist Temporal Classification (CTC) loss (Graves et al. 2006) — the loss for training
    /// sequence models on <b>unsegmented</b> data, where the output length differs from and is not
    /// aligned to the target (line OCR, speech). Given per-timestep class scores and a target label
    /// sequence, it sums the probability of <i>every</i> alignment (via a special <c>blank</c> symbol +
    /// the forward–backward algorithm over the label lattice) and returns the negative log-likelihood,
    /// plus the gradient w.r.t. the logits.
    ///
    /// <para>Single-sequence, caller-owned buffers, no hidden allocation beyond the two log-space
    /// lattices it must keep (forward-backward is inherently <c>O(T·S)</c> memory). Operates on raw
    /// logits — the log-softmax is taken internally and the returned gradient is w.r.t. those logits,
    /// so it drops straight onto a model's output. All arithmetic is in log-space for numerical
    /// stability over long sequences.</para>
    /// </summary>
    public static class CtcLoss
    {
        private const float NegInf = float.NegativeInfinity;

        /// <summary>
        /// Computes the CTC loss for one sequence and (optionally) the gradient w.r.t. the logits.
        /// </summary>
        /// <param name="logits">
        /// Row-major <c>[timeSteps × classCount]</c> raw scores: row <c>t</c> holds the class scores at
        /// timestep <c>t</c> (pre-softmax).
        /// </param>
        /// <param name="timeSteps">Number of input timesteps <c>T</c>.</param>
        /// <param name="classCount">Number of classes <c>C</c>, including the blank.</param>
        /// <param name="target">
        /// Target label indices, each in <c>[0, classCount)</c> and not equal to <paramref name="blankIndex"/>.
        /// May be empty (the model should then emit only blanks).
        /// </param>
        /// <param name="blankIndex">The CTC blank class index (commonly <c>classCount - 1</c> or <c>0</c>).</param>
        /// <param name="logitGradients">
        /// Optional output, same length as <paramref name="logits"/>; when non-empty it is <b>overwritten</b>
        /// with <c>∂loss/∂logits</c> (the classic <c>softmax − posterior</c> form). Pass an empty span to
        /// skip the gradient (loss only).
        /// </param>
        /// <returns>
        /// The loss <c>−ln p(target | logits)</c>. Returns <see cref="float.PositiveInfinity"/> when the
        /// target cannot fit in <paramref name="timeSteps"/> (no valid alignment); the gradient is then
        /// the plain softmax (a sane fallback).
        /// </returns>
        public static float Forward(
            ReadOnlySpan<float> logits,
            int timeSteps,
            int classCount,
            ReadOnlySpan<int> target,
            int blankIndex,
            Span<float> logitGradients)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(timeSteps);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(classCount);
            ArgumentOutOfRangeException.ThrowIfNegative(blankIndex);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(blankIndex, classCount);

            var tc = checked(timeSteps * classCount);
            if (logits.Length < tc)
            {
                throw new ArgumentException($"logits length {logits.Length} < timeSteps*classCount {tc}.", nameof(logits));
            }
            var wantGrad = !logitGradients.IsEmpty;
            if (wantGrad && logitGradients.Length < tc)
            {
                throw new ArgumentException($"logitGradients length {logitGradients.Length} < {tc}.", nameof(logitGradients));
            }
            for (var i = 0; i < target.Length; i++)
            {
                if ((uint)target[i] >= (uint)classCount)
                {
                    throw new ArgumentOutOfRangeException(nameof(target), $"label {target[i]} out of range [0,{classCount}).");
                }
                if (target[i] == blankIndex)
                {
                    throw new ArgumentException($"target[{i}] equals the blank index {blankIndex}.", nameof(target));
                }
            }

            // Log-softmax per timestep into logp[T*C].
            var logp = new float[tc];
            for (var t = 0; t < timeSteps; t++)
            {
                var baseT = t * classCount;
                var max = NegInf;
                for (var k = 0; k < classCount; k++)
                {
                    var v = logits[baseT + k];
                    if (v > max) { max = v; }
                }
                var sum = 0f;
                for (var k = 0; k < classCount; k++)
                {
                    sum += MathF.Exp(logits[baseT + k] - max);
                }
                var logZ = max + MathF.Log(sum);
                for (var k = 0; k < classCount; k++)
                {
                    logp[baseT + k] = logits[baseT + k] - logZ;
                }
            }

            // Extended label: blank l0 blank l1 ... blank  (length S = 2L+1).
            var labelLength = target.Length;
            var s = 2 * labelLength + 1;
            var ext = new int[s];
            for (var i = 0; i < labelLength; i++)
            {
                ext[2 * i] = blankIndex;
                ext[2 * i + 1] = target[i];
            }
            ext[s - 1] = blankIndex;

            // The minimum timesteps needed = S minus the number of adjacent-duplicate label pairs that
            // can share a blank... simpler sufficient check: need T >= labelLength + repeats + 1-ish.
            // We just detect "no valid alignment" from a −inf final α.

            // Forward α (log) over [T*S].
            var alpha = new float[timeSteps * s];
            alpha.AsSpan().Fill(NegInf);
            alpha[0] = logp[ext[0]];                 // blank at s=0
            if (s > 1) { alpha[1] = logp[ext[1]]; }  // first real label

            for (var t = 1; t < timeSteps; t++)
            {
                var prev = (t - 1) * s;
                var cur = t * s;
                var baseT = t * classCount;
                // A label at position si can be reached from si, si-1, and (si-2 when it is a distinct
                // non-blank, i.e. a transition that skips a separating blank).
                var sStart = Math.Max(0, s - 2 * (timeSteps - t));   // prune unreachable tail
                for (var si = sStart; si < s; si++)
                {
                    var a = alpha[prev + si];
                    if (si >= 1) { a = LogSumExp(a, alpha[prev + si - 1]); }
                    if (si >= 2 && ext[si] != blankIndex && ext[si] != ext[si - 2])
                    {
                        a = LogSumExp(a, alpha[prev + si - 2]);
                    }
                    alpha[cur + si] = a + logp[baseT + ext[si]];
                }
            }

            var last = (timeSteps - 1) * s;
            var logProb = LogSumExp(alpha[last + s - 1], s > 1 ? alpha[last + s - 2] : NegInf);

            if (float.IsNegativeInfinity(logProb))
            {
                // No alignment fits T timesteps. Fall back to plain softmax gradient.
                if (wantGrad)
                {
                    for (var i = 0; i < tc; i++) { logitGradients[i] = MathF.Exp(logp[i]); }
                }
                return float.PositiveInfinity;
            }

            if (!wantGrad)
            {
                return -logProb;
            }

            // Backward β (log).
            var beta = new float[timeSteps * s];
            beta.AsSpan().Fill(NegInf);
            var lastBase = (timeSteps - 1) * classCount;
            beta[last + s - 1] = logp[lastBase + ext[s - 1]];
            if (s > 1) { beta[last + s - 2] = logp[lastBase + ext[s - 2]]; }

            for (var t = timeSteps - 2; t >= 0; t--)
            {
                var next = (t + 1) * s;
                var cur = t * s;
                var baseT = t * classCount;
                var sEnd = Math.Min(s - 1, 2 * (t + 1));   // prune unreachable head
                for (var si = sEnd; si >= 0; si--)
                {
                    var b = beta[next + si];
                    if (si + 1 < s) { b = LogSumExp(b, beta[next + si + 1]); }
                    if (si + 2 < s && ext[si] != blankIndex && ext[si] != ext[si + 2])
                    {
                        b = LogSumExp(b, beta[next + si + 2]);
                    }
                    beta[cur + si] = b + logp[baseT + ext[si]];
                }
            }

            // Gradient: grad[t,k] = softmax[t,k] − posterior[t,k],
            //   posterior[t,k] = Σ_{s: ext[s]=k} exp(α[t,s] + β[t,s] − logp[t,k] − logProb).
            // (α and β each include logp[t,ext[s]] once, so their product double-counts it — hence the
            //  − logp[t,k] correction.) Accumulate the per-class lattice mass in log-space first.
            Span<float> classAcc = classCount <= 256 ? stackalloc float[classCount] : new float[classCount];
            for (var t = 0; t < timeSteps; t++)
            {
                var cur = t * s;
                var baseT = t * classCount;
                for (var k = 0; k < classCount; k++) { classAcc[k] = NegInf; }

                for (var si = 0; si < s; si++)
                {
                    var ab = alpha[cur + si] + beta[cur + si];
                    if (!float.IsNegativeInfinity(ab))
                    {
                        var k = ext[si];
                        classAcc[k] = LogSumExp(classAcc[k], ab);
                    }
                }

                for (var k = 0; k < classCount; k++)
                {
                    var softmax = MathF.Exp(logp[baseT + k]);
                    var posterior = float.IsNegativeInfinity(classAcc[k])
                        ? 0f
                        : MathF.Exp(classAcc[k] - logp[baseT + k] - logProb);
                    logitGradients[baseT + k] = softmax - posterior;
                }
            }

            return -logProb;
        }

        private static float LogSumExp(float a, float b)
        {
            if (float.IsNegativeInfinity(a)) { return b; }
            if (float.IsNegativeInfinity(b)) { return a; }
            var max = a > b ? a : b;
            return max + MathF.Log(MathF.Exp(a - max) + MathF.Exp(b - max));
        }
    }
}
