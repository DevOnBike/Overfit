// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    /// <summary>
    /// Pins <see cref="Q4KGemvKernel.GemmTiled512"/> against <see cref="Q4KGemvKernel.GemmTiled"/>.
    ///
    /// <para>The AVX-512 kernel packs two activation columns into one instruction — column <c>2p</c> in the low
    /// 256 bits, <c>2p+1</c> in the high — but performs each column's operations in the same order as the
    /// 256-bit kernel. Nothing is reassociated, so the results must be <b>bit-identical</b>, not merely close;
    /// these tests assert exact equality so that any future reordering shows up immediately rather than hiding
    /// inside a tolerance.</para>
    ///
    /// <para>Odd column counts are covered because the tail pair computes a duplicate of its low column in the
    /// high half and must discard it — an off-by-one there would silently overwrite the neighbouring column.</para>
    /// </summary>
    public sealed class Avx512PrefillParityTests
    {
        private const int InputSize = 512;
        private const int OutputSize = 64;

        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(3)]
        [InlineData(7)]
        [InlineData(8)]
        [InlineData(16)]
        public void GemmTiled512_IsBitIdenticalTo_GemmTiled(int cols)
        {
            if (!Avx512BW.IsSupported || !Avx512F.IsSupported)
            {
                // Nothing to compare on a machine without the wider kernel; the 256-bit path is covered
                // by Q4KTiledGemmParityTests regardless.
                return;
            }

            var (weight, quants, scales, bsums) = BuildInputs(cols);

            {
                var repacked = weight.EnsureRepacked();
                var reference = new float[cols * OutputSize];
                var wide = new float[cols * OutputSize];

                Q4KGemvKernel.GemmTiled(
                    repacked, OutputSize, InputSize, cols, quants, scales, bsums, reference);
                Q4KGemvKernel.GemmTiled512(
                    repacked, OutputSize, InputSize, cols, quants, scales, bsums, wide);

                Assert.Equal(reference, wide);
            }
        }

        [Fact]
        public void GemmTiled512_IsBitIdenticalTo_GemmTiled_WithBias()
        {
            if (!Avx512BW.IsSupported || !Avx512F.IsSupported)
            {
                // Nothing to compare on a machine without the wider kernel; the 256-bit path is covered
                // by Q4KTiledGemmParityTests regardless.
                return;
            }

            const int Cols = 8;
            var (weight, quants, scales, bsums) = BuildInputs(Cols);

            {
                var repacked = weight.EnsureRepacked();
                var bias = new float[OutputSize];
                var rng = new Random(4242);

                for (var i = 0; i < bias.Length; i++)
                {
                    bias[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                }

                var reference = new float[Cols * OutputSize];
                var wide = new float[Cols * OutputSize];

                Q4KGemvKernel.GemmTiled(
                    repacked, OutputSize, InputSize, Cols, quants, scales, bsums, reference, bias);
                Q4KGemvKernel.GemmTiled512(
                    repacked, OutputSize, InputSize, Cols, quants, scales, bsums, wide, bias);

                Assert.Equal(reference, wide);
            }
        }

        /// <summary>The pre-decoded scale path must not change the result either — it only widens F16 earlier.</summary>
        [Fact]
        public void GemmTiled512_IsBitIdenticalTo_GemmTiled_WithPrecomputedScales()
        {
            if (!Avx512BW.IsSupported || !Avx512F.IsSupported)
            {
                // Nothing to compare on a machine without the wider kernel; the 256-bit path is covered
                // by Q4KTiledGemmParityTests regardless.
                return;
            }

            const int Cols = 8;
            var (weight, quants, scales, bsums) = BuildInputs(Cols);

            {
                var repacked = weight.EnsureRepacked();
                var decoded = new float[(OutputSize / 8) * (InputSize / 256) * Q4KGemvKernel.DecodedScalesPerBlock];

                Q4KGemvKernel.DecodeBlockScales(repacked, OutputSize, InputSize, decoded);

                var reference = new float[Cols * OutputSize];
                var wide = new float[Cols * OutputSize];

                Q4KGemvKernel.GemmTiled(
                    repacked, OutputSize, InputSize, Cols, quants, scales, bsums, reference);
                Q4KGemvKernel.GemmTiled512(
                    repacked, OutputSize, InputSize, Cols, quants, scales, bsums, wide, [], decoded);

                Assert.Equal(reference, wide);
            }
        }

        /// <summary>
        /// The banded call, which is the path that was silently 256-bit until 2026-08-02.
        ///
        /// <para><c>BatchedQuantProjection.TiledBandChunk</c> had no <c>if (c.Avx512)</c> branch while its
        /// three siblings all did, so a machine configured for AVX-512 ran the narrow kernel on precisely the
        /// short-prompt case banding exists for. Wiring it up meant giving <c>GemmTiled512</c> the
        /// <c>groupStart</c>/<c>groupCount</c> parameters <c>GemmTiled</c> already had — new code, and this is
        /// what pins it.</para>
        ///
        /// <para>Every band is checked, and the union of the bands is checked against the unbanded result:
        /// a band that computed the wrong groups but computed them correctly would pass the first check
        /// alone.</para>
        /// </summary>
        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(4)]
        public void GemmTiled512_Banded_IsBitIdenticalTo_GemmTiled_Banded(int groupsPerBand)
        {
            if (!Avx512BW.IsSupported || !Avx512F.IsSupported)
            {
                return;
            }

            const int Cols = 8;

            var (weight, quants, scales, bsums) = BuildInputs(Cols);
            var repacked = weight.EnsureRepacked();
            var totalGroups = OutputSize / 8;

            var wholeReference = new float[Cols * OutputSize];
            var assembled = new float[Cols * OutputSize];

            Q4KGemvKernel.GemmTiled(
                repacked, OutputSize, InputSize, Cols, quants, scales, bsums, wholeReference);

            for (var groupStart = 0; groupStart < totalGroups; groupStart += groupsPerBand)
            {
                var groupCount = Math.Min(groupsPerBand, totalGroups - groupStart);

                var reference = new float[Cols * OutputSize];
                var wide = new float[Cols * OutputSize];

                Q4KGemvKernel.GemmTiled(
                    repacked, OutputSize, InputSize, Cols, quants, scales, bsums, reference, [],
                    groupStart, groupCount);

                Q4KGemvKernel.GemmTiled512(
                    repacked, OutputSize, InputSize, Cols, quants, scales, bsums, wide, [], [],
                    groupStart, groupCount);

                Assert.Equal(reference, wide);

                // Bands are disjoint, so accumulating them must reconstruct the whole matrix exactly.
                for (var c = 0; c < Cols; c++)
                {
                    for (var g = 0; g < groupCount; g++)
                    {
                        var row = (groupStart + g) * 8;

                        for (var r = row; r < row + 8; r++)
                        {
                            assembled[(c * OutputSize) + r] = wide[(c * OutputSize) + r];
                        }
                    }
                }
            }

            Assert.Equal(wholeReference, assembled);
        }

        private static (Q4KWeight Weight, sbyte[] Quants, float[] Scales, short[] Bsums) BuildInputs(int cols)
        {
            var rng = new Random(20260722);

            var f32 = new float[(long)OutputSize * InputSize];

            for (var i = 0; i < f32.Length; i++)
            {
                f32[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            var weight = new Q4KWeight(GgmlQuant.QuantizeQ4_K(f32, InputSize, OutputSize), InputSize, OutputSize);

            var spr = weight.SuperBlocksPerRow;
            var bsumsPerRow = spr * Q4KDotKernel.GroupsPerSuperBlock;

            var input = new float[cols * InputSize];

            for (var i = 0; i < input.Length; i++)
            {
                input[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            var quants = new sbyte[cols * InputSize];
            var scales = new float[cols * spr];
            var bsums = new short[cols * bsumsPerRow];

            for (var c = 0; c < cols; c++)
            {
                Q4KDotKernel.QuantizeActivationQ8K(
                    input.AsSpan(c * InputSize, InputSize),
                    quants.AsSpan(c * InputSize, InputSize),
                    scales.AsSpan(c * spr, spr),
                    bsums.AsSpan(c * bsumsPerRow, bsumsPerRow));
            }

            return (weight, quants, scales, bsums);
        }
    }
}
