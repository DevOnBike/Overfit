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
    /// Opt-in per-component profiler for the BATCHED PREFILL path — the counterpart to
    /// <see cref="DecodeProfiler"/>, which hooks only single-token decode and therefore reports nothing about
    /// time-to-first-token.
    ///
    /// <para><b>Why this exists.</b> Prefill was measured at 3.76× behind llama.cpp on the same file
    /// (144 vs 541.7 tok/s, 672-token prompt), and that gap decomposes as 2.34× kernel quality at equal
    /// instruction set × 1.60× AVX-512. Deciding what to fix first requires knowing whether the 2.34× sits in
    /// the FFN matmuls or in the attention path — where Q and O are dispatched <b>once per head</b>, each
    /// re-quantizing the same loop-invariant activation matrix. Guessing that split has already been wrong
    /// three times, so it gets measured.</para>
    ///
    /// <para><b>Off by default.</b> When <see cref="Enabled"/> is false every hook is one predicted-false
    /// branch — no timestamp, no allocation. Hooks sit at layer and projection granularity (tens per request),
    /// never per element, so even the branch is immeasurable. Accumulators are a static singleton: this
    /// profiles <b>one</b> prefill at a time and is a diagnostic, not a concurrent-safe meter.</para>
    /// </summary>
    public static class PrefillProfiler
    {
        /// <summary>Prefill components timed independently.</summary>
        public enum Component
        {
            /// <summary>Whole batched attention block for one layer. TOP-LEVEL.</summary>
            Attention = 0,

            /// <summary>Whole batched FFN for one layer (SwiGLU / MoE). TOP-LEVEL.</summary>
            Ffn = 1,

            /// <summary>K and V projections — once per KV group, not per head. Sub-slice of <see cref="Attention"/>.</summary>
            AttnKv = 2,

            /// <summary>Q projection — dispatched once PER HEAD over the same activations. Sub-slice of <see cref="Attention"/>.</summary>
            AttnQ = 3,

            /// <summary>Causal scores + weighted sum over the KV cache. Sub-slice of <see cref="Attention"/>.</summary>
            AttnScores = 4,

            /// <summary>Output projection — also dispatched per head. Sub-slice of <see cref="Attention"/>.</summary>
            AttnOut = 5,

            /// <summary>FFN gate+up projections and the gate activation. Sub-slice of <see cref="Ffn"/>.</summary>
            FfnGateUp = 6,

            /// <summary>FFN down projection. Sub-slice of <see cref="Ffn"/>.</summary>
            FfnDown = 7,
        }

        private const int ComponentCount = 8;
        private const int LastTopLevel = (int)Component.Ffn;

        private static readonly long[] _ticks = new long[ComponentCount];
        private static readonly long[] _calls = new long[ComponentCount];
        private static long _requestTicks;
        private static long _requestStart;
        private static long _requests;
        private static long _rows;

        /// <summary>Master switch. Leave <c>false</c> in production; flip on around a measured prefill.</summary>
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

        /// <summary>Mark the start of one prefill request over <paramref name="rows"/> prompt tokens.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void BeginRequest(int rows)
        {
            if (!Enabled)
            {
                return;
            }

            _requestStart = Stopwatch.GetTimestamp();
            _rows += rows;
        }

        /// <summary>Close the current request and add it to the totals.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void EndRequest()
        {
            if (!Enabled)
            {
                return;
            }

            _requestTicks += Stopwatch.GetTimestamp() - _requestStart;
            _requests++;
        }

        /// <summary>Clear all accumulators (call before the measured segment).</summary>
        public static void Reset()
        {
            Array.Clear(_ticks);
            Array.Clear(_calls);
            _requestTicks = 0;
            _requestStart = 0;
            _requests = 0;
            _rows = 0;
        }

        /// <summary>Prompt tokens prefilled since the last <see cref="Reset"/>.</summary>
        public static long Rows => _rows;

        /// <summary>
        /// Per-request breakdown: ms, % of prefill wall time, and calls. Also prints the prefill rate in
        /// tok/s, which is directly comparable to <c>llama-bench -p N -n 0</c>.
        /// </summary>
        public static string Report()
        {
            var sb = new StringBuilder();
            var toMs = 1000.0 / Stopwatch.Frequency;
            var requests = _requests == 0 ? 1 : _requests;
            var requestMs = _requestTicks * toMs / requests;
            var rowsPerRequest = (double)_rows / requests;

            sb.AppendLine($"=== PrefillProfiler ({_requests} request(s), {rowsPerRequest:F0} tokens each) ===");
            sb.AppendLine(
                $"  total/request : {requestMs,9:F1} ms   ({(requestMs > 0 ? rowsPerRequest * 1000.0 / requestMs : 0),7:F0} tok/s)");

            // Only TOP-LEVEL components count toward "accounted" — the sub-slices overlap their parent, and
            // summing them too would subtract the same work twice and drive `other` negative. DecodeProfiler
            // shipped with exactly that bug (it read -67%), so the same mistake is not repeated here.
            long accounted = 0;
            for (var i = 0; i < ComponentCount; i++)
            {
                var ms = _ticks[i] * toMs / requests;
                if (i <= LastTopLevel)
                {
                    accounted += _ticks[i];
                }

                var pct = _requestTicks > 0 ? 100.0 * _ticks[i] / _requestTicks : 0;
                var perRequest = (double)_calls[i] / requests;
                var indent = i > LastTopLevel ? "  " : string.Empty;
                sb.AppendLine(
                    $"  {indent + ComponentName(i),-14} : {ms,9:F1} ms   {pct,5:F1}%   ({perRequest,6:F0} calls)");
            }

            var otherTicks = _requestTicks - accounted;
            var otherMs = otherTicks * toMs / requests;
            var otherPct = _requestTicks > 0 ? 100.0 * otherTicks / _requestTicks : 0;
            sb.AppendLine(
                $"  {"other",-14} : {otherMs,9:F1} ms   {otherPct,5:F1}%   (norms/residual/embed/finalnorm/RoPE)");
            return sb.ToString();
        }

        private static string ComponentName(int i) => i switch
        {
            0 => "attention",
            1 => "ffn",
            2 => "attn_kv",
            3 => "attn_q",
            4 => "attn_scores",
            5 => "attn_out",
            6 => "ffn_gateup",
            7 => "ffn_down",
            _ => "?",
        };
    }
}
