// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Opt-in per-component decode profiler (the "measure before you optimize" tool).
    /// Splits single-token decode wall time into <see cref="Component.Attention"/> /
    /// <see cref="Component.Ffn"/> / <see cref="Component.LmHead"/> /
    /// <see cref="Component.Sampler"/> (plus an "other" remainder = norms / residuals /
    /// embed / final-norm) so the next optimization targets the <i>measured</i>
    /// bottleneck, not a guessed one.
    ///
    /// <para><b>Off by default.</b> When <see cref="Enabled"/> is false every hook is a
    /// single predicted-false branch — no timestamp, no allocation — so the production
    /// decode hot path is unaffected. Accumulators are a static singleton, so it profiles
    /// <b>one</b> decode stream at a time (a diagnostic, not a concurrent-safe meter).</para>
    /// </summary>
    public static class DecodeProfiler
    {
        /// <summary>Hot-path decode components timed independently.</summary>
        public enum Component
        {
            /// <summary>Whole <c>CachedMultiHeadAttention.Decode</c> (QKV + attend + Wo).</summary>
            Attention = 0,

            /// <summary>Whole FFN block (SwiGLU / GeLU / MoE).</summary>
            Ffn = 1,

            /// <summary>LM-head logits projection.</summary>
            LmHead = 2,

            /// <summary>Token selection (greedy / sampling / constraint mask + accept).</summary>
            Sampler = 3,

            /// <summary>FFN gate+up projection (matmul, incl. fused activation-quantize) + gate activation.
            /// A sub-slice of <see cref="Ffn"/> — overlaps it, so it inflates the report's "other" remainder.</summary>
            FfnGateUp = 4,

            /// <summary>FFN down projection (the single biggest matmul). Sub-slice of <see cref="Ffn"/>.</summary>
            FfnDown = 5,

            /// <summary>FFN gate*up element-wise multiply. Sub-slice of <see cref="Ffn"/>.</summary>
            FfnMultiply = 6,
        }

        private const int ComponentCount = 7;

        private static readonly long[] _ticks = new long[ComponentCount];
        private static readonly long[] _calls = new long[ComponentCount];
        private static long _tokenTicks;
        private static long _tokenStart;
        private static long _tokens;

        /// <summary>Master switch. Leave <c>false</c> in production; flip on around a
        /// measured decode run, then read <see cref="Report"/>.</summary>
        public static bool Enabled;

        /// <summary>Timestamp to pass to a matching <see cref="Stop"/>. Cheap no-op when off.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static long Start() => Enabled ? Stopwatch.GetTimestamp() : 0L;

        /// <summary>Accumulate elapsed ticks (and one call) for <paramref name="component"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Stop(Component component, long start)
        {
            if (!Enabled)
            {
                return;
            }

            _ticks[(int)component] += Stopwatch.GetTimestamp() - start;
            _calls[(int)component]++;
        }

        /// <summary>Mark the start of one decoded token (the per-token wall clock).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void BeginToken()
        {
            if (!Enabled)
            {
                return;
            }

            _tokenStart = Stopwatch.GetTimestamp();
        }

        /// <summary>Close the current token and add it to the token total + count.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void EndToken()
        {
            if (!Enabled)
            {
                return;
            }

            _tokenTicks += Stopwatch.GetTimestamp() - _tokenStart;
            _tokens++;
        }

        /// <summary>Clear all accumulators (call before the measured segment).</summary>
        public static void Reset()
        {
            Array.Clear(_ticks);
            Array.Clear(_calls);
            _tokenTicks = 0;
            _tokenStart = 0;
            _tokens = 0;
        }

        /// <summary>Tokens counted since the last <see cref="Reset"/>.</summary>
        public static long Tokens => _tokens;

        /// <summary>Per-token breakdown: ms and % of token wall time, plus calls/token.</summary>
        public static string Report()
        {
            var sb = new StringBuilder();
            var toMs = 1000.0 / Stopwatch.Frequency;
            var tokens = _tokens == 0 ? 1 : _tokens;
            var tokenMs = _tokenTicks * toMs / tokens;

            sb.AppendLine($"=== DecodeProfiler ({_tokens} tokens) ===");
            sb.AppendLine(
                $"  total / token : {tokenMs,9:F3} ms   ({(tokenMs > 0 ? 1000.0 / tokenMs : 0),6:F2} tok/s)");

            long accounted = 0;
            for (var i = 0; i < ComponentCount; i++)
            {
                var ms = _ticks[i] * toMs / tokens;
                accounted += _ticks[i];
                var pct = _tokenTicks > 0 ? 100.0 * _ticks[i] / _tokenTicks : 0;
                var perTok = (double)_calls[i] / tokens;
                sb.AppendLine($"  {ComponentName(i),-13} : {ms,9:F3} ms   {pct,5:F1}%   ({perTok,5:F0}/tok)");
            }

            var otherTicks = _tokenTicks - accounted;
            var otherMs = otherTicks * toMs / tokens;
            var otherPct = _tokenTicks > 0 ? 100.0 * otherTicks / _tokenTicks : 0;
            sb.AppendLine(
                $"  {"other",-13} : {otherMs,9:F3} ms   {otherPct,5:F1}%   (norms/residual/embed/finalnorm)");
            return sb.ToString();
        }

        private static string ComponentName(int i) => i switch
        {
            0 => "attention",
            1 => "ffn",
            2 => "lm_head",
            3 => "sampler",
            4 => "ffn_gateup",
            5 => "ffn_down",
            6 => "ffn_multiply",
            _ => "?",
        };
    }
}
