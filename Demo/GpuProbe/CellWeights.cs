// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// The synthetic weight of one cell, in both forms: quantized to Q4_K (what arms C1 and C2 read) and
    /// decoded back to F32 (what arms C3 and C4 read, and what the device receives).
    /// <para>
    /// The F32 form is produced by DECODING the quantized form, not by keeping the pre-quantization
    /// floats. Quantization is lossy, so keeping the originals would leave C1 and C3 computing two
    /// different functions and the parity check comparing the device against the wrong reference.
    /// </para>
    /// <para>
    /// Held separately from <see cref="CellFixture"/> because it depends only on the cell and not on the
    /// token count, so the batch sweep reuses it rather than rebuilding it three times.
    /// </para>
    /// <para>
    /// It buys much less than it looks like it should, and the number is here so nobody re-derives the
    /// wrong reason. Measured 2026-08-21, one reading each, parity-only over the five production shapes
    /// on an AMD gfx1036 through ILGPU's OpenCL backend: <b>709.6 s</b> quantizing once per (cell, token
    /// count) pair, <b>697.3 s</b> quantizing once per cell. That is 1.7 %, which one reading cannot
    /// separate from noise. So the quantize was NOT the bottleneck of the run, and what is has NOT been
    /// decomposed. The split is kept because it is also the lower-peak-RAM shape, not because it is fast.
    /// </para>
    /// </summary>
    internal sealed class CellWeights
    {
        public CellWeights(Cell cell, int seed)
        {
            Cell = cell;

            var k = cell.K;
            var m = cell.M;
            var rnd = new Random(seed);

            // One buffer, used twice: random F32 in, quantized out, then decoded back over itself.
            F32 = new float[(long)m * k];
            for (var i = 0; i < F32.Length; i++)
            {
                F32[i] = (float)(rnd.NextDouble() * 2 - 1) * 0.08f;
            }

            Quantized = new Q4KWeight(GgmlQuant.QuantizeQ4_K(F32, k, m), k, m);
            for (var o = 0; o < m; o++)
            {
                Quantized.DecodeRow(o, F32.AsSpan(o * k, k));
            }
        }

        public Cell Cell { get; }

        public Q4KWeight Quantized { get; }

        public float[] F32 { get; }
    }
}
