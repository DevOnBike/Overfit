// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio;
using DevOnBike.Overfit.Audio.Tts.Snac;
using DevOnBike.Overfit.Tests.TestSupport;

namespace DevOnBike.Overfit.Tests.Audio
{
    /// <summary>End-to-end parity for the SNAC decoder: real <c>snac_24khz</c> weights decode a fixed set of real
    /// codes to a waveform that matches the reference (PyTorch) noise-off decode. Validates the whole graph —
    /// codebook gather, out_proj, repeat-interleave + cross-level sum, depthwise stem, transposed-conv upsampling,
    /// dilated residual units, output conv + tanh — against the source of truth. Needs <c>c:\snac</c> (see
    /// Scripts/convert_snac.py); [LongFact] by default.</summary>
    public sealed class SnacDecoderParityTests
    {
        [LongFact]
        public void Decode_RealCodes_MatchesReferenceNoiseOffDecode()
        {
            TestModelPaths.Snac.RequireSafetensorsPath();

            var codes = ReadCodes(TestModelPaths.Snac.CodesPath);
            var reference = ReadF32(TestModelPaths.Snac.ReferenceNoiseOffPath);

            var snac = Snac.Load(TestModelPaths.Snac.Dir);
            var audio = snac.Decode(codes, addNoise: false);

            Assert.Equal(reference.Length, audio.Length);

            // Deterministic path, but our float op-order differs from PyTorch's → expect near-identical, not bit-exact.
            var report = AudioSimilarity.Compare(reference, snac.SampleRate, audio, snac.SampleRate);
            Assert.True(report.SignalToNoiseRatioDb > 40.0,
                $"SNAC decode diverged from reference: {report}");
            Assert.True(report.Correlation > 0.9999, $"low correlation: {report}");
        }

        private static int[][] ReadCodes(string path)
        {
            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);
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

        private static float[] ReadF32(string path)
        {
            var bytes = File.ReadAllBytes(path);
            var floats = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, floats, 0, floats.Length * 4);
            return floats;
        }
    }
}
