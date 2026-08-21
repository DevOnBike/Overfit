// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.TestSupport
{
    /// <summary>
    /// Sets <see cref="BatchedQuantProjection.UseTiledPrefillQ4K"/> for the duration of the scope and puts the
    /// previous value back, so a test that needs the register-tiled Q4_K prefill GEMM on (or off) says so in one
    /// line instead of an open-coded try/finally. Mirrors <see cref="NonRepackedKernelScope"/>.
    ///
    /// <para><b>It restores the previous value, not a constant.</b> An unconditional reset is not a scope: two
    /// nested ones leave the flag wrong after the inner one exits. That defect is why the sibling type exists in
    /// this shape.</para>
    ///
    /// <para><b>Thread affinity, and it is the point of the mechanism.</b> The flag is a per-thread override over
    /// a process-wide default, so this scope binds the thread that opened it — which is what stops a
    /// <c>[ModelFact]</c> in one class from switching the kernel under a fast <c>[Fact]</c> in another while
    /// xunit runs their collections in parallel. Drive the engine from the same thread: an <c>async</c>
    /// continuation or a <c>Task.Run</c> inside the scope silently reads the default instead.</para>
    ///
    /// <para>Restoring writes an explicit override equal to whatever was read on construction, so a thread that
    /// held no override ends up holding one that equals the default. That is value-identical and stays so: the
    /// default is <see cref="Q4KGemvKernel.TiledPrefillEnabled"/>, a <c>static readonly</c> resolved once per
    /// process.</para>
    ///
    /// <para><b>Construct it, never <c>default</c> it.</b> <c>default(TiledPrefillQ4KScope)</c> skips both
    /// constructors, so its <c>Dispose</c> writes <c>false</c> rather than restoring anything.</para>
    ///
    /// <para>Six classes write this flag: <c>BatchedQuantProjectionTiledDispatchTests</c>,
    /// <c>BatchedQuantProjectionTiledPrefillFlagTests</c>, <c>RepackedSidecarEngineE2ETests</c>,
    /// <c>RepackedGemmReproducibilityDiagnostics</c>, <c>PerHeadTiledVsWeightStationaryDiagnostics</c> and
    /// <c>TinyBlasTiledPrefillE2EPhase3Tests</c> — the last three being the instruments one would reach for to
    /// investigate this very flag, and all of them <c>[ModelFact]</c>, so their windows are seconds rather than
    /// the fast test's milliseconds.</para>
    /// </summary>
    internal readonly struct TiledPrefillQ4KScope : IDisposable
    {
        private readonly bool _previous;

        /// <summary>Saves the current value and sets <paramref name="enabled"/>.</summary>
        public TiledPrefillQ4KScope(bool enabled)
        {
            _previous = BatchedQuantProjection.UseTiledPrefillQ4K;
            BatchedQuantProjection.UseTiledPrefillQ4K = enabled;
        }

        /// <summary>Saves the current value and sets nothing — for a body that toggles the flag itself.</summary>
        public TiledPrefillQ4KScope() => _previous = BatchedQuantProjection.UseTiledPrefillQ4K;

        public void Dispose() => BatchedQuantProjection.UseTiledPrefillQ4K = _previous;
    }
}
