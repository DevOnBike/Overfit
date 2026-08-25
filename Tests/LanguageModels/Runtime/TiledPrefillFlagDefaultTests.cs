// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Pins that <c>OVERFIT_TILED_PREFILL</c> — and the two flags resolved the same way — stay <b>opt-in</b>:
    /// unset means off, an unrecognised value means off, and only <c>1</c> or <c>true</c> means on.
    ///
    /// <para><b>Why this exists at all, and it is a measured gap rather than a guess.</b> On 2026-08-25 the
    /// flag was made default-on. A mutation that performs exactly that flip was run against the whole suite
    /// with this class excluded, and <b>0 of 2795 tests noticed</b>. So nothing pinned the default in either
    /// direction, and the change that would have moved it silently is not hypothetical — it was written,
    /// measured and reverted the same day (`XC-119`).</para>
    ///
    /// <para><b>What the revert was for, so nobody re-proposes it from this file.</b> The kernel is worth
    /// 2.98× at <c>pp512</c>, but defaulting the flag on repacks every Q4_K tensor onto the heap: +1194 MiB
    /// of peak private commit and +186 ms once at the start of decode, which leaves a short CLI invocation
    /// 9.5% slower for 2.4× the memory. Conditions and fitted coefficients live in
    /// <c>docs/measured-baselines.md</c> under <c>XC-119</c>, not here.</para>
    ///
    /// <para><b>The variable itself is never touched.</b> Every case drives
    /// <see cref="Q4KGemvKernel.ResolveFlag"/> through a uniquely-named throw-away variable, because xunit
    /// runs collections in parallel and <c>Environment.SetEnvironmentVariable</c> is process-wide.</para>
    ///
    /// <para><b>Reach, stated because it is narrower than the class name.</b> The AVX2 gate cannot be
    /// exercised in-process — <c>Avx2.IsSupported</c> is fixed at JIT time — so on a box with AVX2 these pin
    /// the parse and the default, and on a box without it they pin only that everything resolves to
    /// <c>false</c>. The non-AVX2 arm was measured out of process instead, under <c>DOTNET_EnableAVX2=0</c>:
    /// 9.539 t/s flag-unset against 9.535 t/s flag-on, 292 against 294 MiB. Neither figure is asserted
    /// here.</para>
    /// </summary>
    public sealed class TiledPrefillFlagDefaultTests
    {
        /// <summary>
        /// The assertion the 2026-08-25 default flip breaks, and the one nothing in the suite carried before
        /// it. The variable is removed rather than set to the empty string, because "unset" has to mean unset.
        /// </summary>
        [Fact]
        public void UnsetVariable_ResolvesToOff()
        {
            var name = UniqueName();
            Environment.SetEnvironmentVariable(name, null);

            Assert.False(Q4KGemvKernel.ResolveFlag(name));
        }

        /// <summary>Only these two spellings turn an opt-in flag on, and only when AVX2 is present.</summary>
        [Theory]
        [InlineData("1")]
        [InlineData("true")]
        [InlineData("True")]
        [InlineData("TRUE")]
        public void AskingValue_ResolvesToOn_WhenAvx2IsPresent(string raw)
        {
            var name = UniqueName();

            try
            {
                Environment.SetEnvironmentVariable(name, raw);
                Assert.Equal(Avx2.IsSupported, Q4KGemvKernel.ResolveFlag(name));
            }
            finally
            {
                Environment.SetEnvironmentVariable(name, null);
            }
        }

        /// <summary>
        /// Everything else leaves it off — including <c>0</c> and <c>false</c>, which read as off here for the
        /// same reason <c>yes</c> does: they are simply not the ON table. That is worth pinning rather than
        /// assuming, because a default-ON flag would have to treat those three differently from each other.
        /// </summary>
        [Theory]
        [InlineData("0")]
        [InlineData("false")]
        [InlineData("FALSE")]
        [InlineData("yes")]
        [InlineData("on")]
        [InlineData("")]
        [InlineData(" 1")]
        public void AnyOtherValue_ResolvesToOff(string raw)
        {
            var name = UniqueName();

            try
            {
                Environment.SetEnvironmentVariable(name, raw);
                Assert.False(Q4KGemvKernel.ResolveFlag(name));
            }
            finally
            {
                Environment.SetEnvironmentVariable(name, null);
            }
        }

        /// <summary>The ON table, pinned directly so a change to it cannot hide behind the resolver.</summary>
        [Fact]
        public void TheOnTable_ClaimsNeitherTheUnsetCaseNorAnythingItDoesNotRecognise()
        {
            Assert.True(Q4KGemvKernel.IsTruthy("1"));
            Assert.True(Q4KGemvKernel.IsTruthy("true"));
            Assert.True(Q4KGemvKernel.IsTruthy("TRUE"));

            Assert.False(Q4KGemvKernel.IsTruthy(null));
            Assert.False(Q4KGemvKernel.IsTruthy(""));
            Assert.False(Q4KGemvKernel.IsTruthy("0"));
            Assert.False(Q4KGemvKernel.IsTruthy("false"));
            Assert.False(Q4KGemvKernel.IsTruthy("yes"));
        }

        /// <summary>
        /// The three shipped fields are wired to the opt-in resolver. Without this every case above could pass
        /// while <see cref="Q4KGemvKernel.TiledPrefillEnabled"/> was resolved some other way and the shipped
        /// default had moved anyway — which is exactly the change that went unnoticed by 2795 tests.
        /// </summary>
        [Theory]
        [InlineData(OverfitEnvironment.TiledPrefill)]
        [InlineData(OverfitEnvironment.RepackGemv)]
        [InlineData(OverfitEnvironment.RepackAttn)]
        public void TheShippedFields_AreOffUnlessTheirVariableAsksForThem(string variable)
        {
            var raw = Environment.GetEnvironmentVariable(variable);
            var expected = Avx2.IsSupported && Q4KGemvKernel.IsTruthy(raw);

            var actual = variable switch
            {
                OverfitEnvironment.TiledPrefill => Q4KGemvKernel.TiledPrefillEnabled,
                OverfitEnvironment.RepackGemv => Q4KGemvKernel.Enabled,
                _ => Q4KGemvKernel.AttnEnabled,
            };

            Assert.Equal(expected, actual);
        }

        private static string UniqueName() => "OVERFIT_TILED_PREFILL_TEST_" + Guid.NewGuid().ToString("N");
    }
}
