// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio.Tts.Snac;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>End-to-end parity for the SNAC encoder: real weights encode a known waveform to the same int codes
    /// PyTorch produced (the reference <c>codes.bin</c> came from <c>model.encode(input)</c>). Validates the whole
    /// encode graph — conv downsampling stem, dilated residual units, depthwise output conv, avg-pool, in_proj,
    /// nearest-codebook assignment, residual subtraction. Needs <c>c:\snac</c> (Scripts/convert_snac.py); [LongFact].</summary>
    public sealed class SnacEncoderParityTests
    {
        [LongFact]
        public void Encode_RealAudio_ReproducesReferenceCodes()
        {
            TestModelPaths.Snac.RequireSafetensorsPath();

            var input = ReadF32(TestModelPaths.Snac.InputPath);
            var reference = ReadCodes(TestModelPaths.Snac.CodesPath);

            var snac = Snac.Load(TestModelPaths.Snac.Dir);
            var codes = snac.Encode(input);

            Assert.Equal(reference.Length, codes.Length);
            var total = 0;
            var matched = 0;
            for (var level = 0; level < reference.Length; level++)
            {
                Assert.Equal(reference[level].Length, codes[level].Length);
                for (var j = 0; j < reference[level].Length; j++)
                {
                    total++;
                    if (reference[level][j] == codes[level][j])
                    {
                        matched++;
                    }
                }
            }

            // Deterministic encode; allow a tiny tolerance for argmin ties at float precision.
            var agreement = matched / (double)total;
            Assert.True(agreement >= 0.98, $"encoder codes matched only {matched}/{total} ({agreement:P1})");
        }

        private static float[] ReadF32(string path)
        {
            var bytes = File.ReadAllBytes(path);
            var floats = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, floats, 0, floats.Length * 4);
            return floats;
        }

        private static int[][] ReadCodes(string path)
        {
            using var br = new BinaryReader(File.OpenRead(path));
            var levels = br.ReadInt32();
            var lengths = new int[levels];
            for (var i = 0; i < levels; i++)
            {
                lengths[i] = br.ReadInt32();
            }
            var codes = new int[levels][];
            for (var i = 0; i < levels; i++)
            {
                codes[i] = new int[lengths[i]];
                for (var j = 0; j < lengths[i]; j++)
                {
                    codes[i][j] = br.ReadInt32();
                }
            }
            return codes;
        }
    }
}
