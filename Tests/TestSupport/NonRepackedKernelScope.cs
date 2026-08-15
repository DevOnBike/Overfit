// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.TestSupport
{
    /// <summary>
    /// Forces the NON-repacked batched (prefill) kernels for the duration of the scope, so a parity
    /// assertion compares two runs of ONE kernel rather than two different ones. The only writer of
    /// <see cref="BatchedQuantProjection.DisableRepackedKernelsForParity"/> in the test tree.
    ///
    /// <para>Without it a batched-vs-single-token parity test silently stops testing what it claims: the
    /// repacked <c>block_q*_Kx8</c> GEMMs associate their reduction differently from the per-row kernels the
    /// single-token path uses, so they are NOT bit-identical — measured at <c>maxAbsLogitDiff ~ 0.44</c> on
    /// Qwen-3B, enough to flip an argmax. The repacked kernels are held to end-to-end coherence instead
    /// (<c>BatchedPrefillParityTests.RepackedPrefill_AgreesWithNonRepacked_OnArgmax</c>).</para>
    ///
    /// <para>The reason the hook exists at all: a <c>*.gguf.repack</c> sidecar sets <c>IsPrepacked</c> and
    /// switches the repacked path on <b>regardless of the env flag</b>, which is how
    /// <c>BatchedPrefillParityTests</c> came to be failing unnoticed for two days — it is <c>[LongFact]</c>,
    /// so it never ran.</para>
    ///
    /// <para><b>It restores the previous value, not <c>false</c>.</b> Two private near-twins of this type
    /// (in <c>BatchedPrefillParityTests</c> and <c>PrefixKvCacheParityTests</c>) each reset the flag to a
    /// constant, which is an unconditional reset and not a scope: nesting two of them left the flag wrong
    /// after the inner one exited. Both are deleted; this is the shared replacement their own comments asked
    /// for once a third case appeared.</para>
    ///
    /// <para><b>Thread affinity.</b> The flag is <c>[ThreadStatic]</c>, so the scope binds the thread that
    /// opened it. Drive the engine from that same thread — an <c>async</c> continuation or a
    /// <c>Task.Run</c> inside the scope silently gets the repacked kernel back.</para>
    /// </summary>
    internal readonly struct NonRepackedKernelScope : IDisposable
    {
        private readonly bool _previous;

        public NonRepackedKernelScope()
        {
            _previous = BatchedQuantProjection.DisableRepackedKernelsForParity;
            BatchedQuantProjection.DisableRepackedKernelsForParity = true;
        }

        public void Dispose() => BatchedQuantProjection.DisableRepackedKernelsForParity = _previous;
    }
}
