// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Whisper;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Whisper
{
    /// <summary>
    /// Structural validation of <see cref="WhisperEncoder"/> on a tiny synthetic model (random weights, the
    /// real whisper.cpp tensor names + shapes): the conv stem → positional embed → N pre-LN blocks → ln_post
    /// pipeline produces the right output shape <c>[nCtx × nAudioState]</c>, all-finite, and deterministic.
    /// Numerical parity vs whisper.cpp is S5 (needs a real ggml-tiny).
    /// </summary>
    public sealed class WhisperEncoderTests
    {
        private const int NState = 8, NHead = 2, NLayer = 2, NMels = 4, Frames = 8, DFF = 4 * NState;

        private readonly ITestOutputHelper _out;
        public WhisperEncoderTests(ITestOutputHelper output) => _out = output;

        [Fact]
        public void Encode_ProducesCorrectShape_Finite_Deterministic()
        {
            var model = BuildTinyEncoder(seed: 3);
            var encoder = new WhisperEncoder(model);

            var rng = new Random(9);
            var mel = new float[NMels * Frames];
            for (var i = 0; i < mel.Length; i++) { mel[i] = (float)(rng.NextDouble() * 2 - 1); }

            var outA = encoder.Encode(mel, Frames, out var nCtx);
            var outB = encoder.Encode(mel, Frames, out _);

            var expectedCtx = (Frames + 2 * 1 - 3) / 2 + 1; // conv2 stride 2, pad 1, k 3
            _out.WriteLine($"nCtx={nCtx} (expected {expectedCtx}), output length {outA.Length}");
            Assert.Equal(expectedCtx, nCtx);
            Assert.Equal(nCtx * NState, outA.Length);

            for (var i = 0; i < outA.Length; i++)
            {
                Assert.True(float.IsFinite(outA[i]), $"output[{i}] not finite");
                Assert.Equal(outA[i], outB[i]); // deterministic
            }
        }

        private static WhisperModel BuildTinyEncoder(int seed)
        {
            var rng = new Random(seed);
            var t = new Dictionary<string, WhisperTensor>();

            void Add(string name, params int[] shape)
            {
                long n = 1;
                foreach (var d in shape) { n *= d; }
                var data = new float[n];
                for (var i = 0; i < n; i++) { data[i] = (float)(rng.NextDouble() * 2 - 1) * 0.2f; }
                t[name] = new WhisperTensor(shape, data);
            }

            var nCtx = (Frames + 2 * 1 - 3) / 2 + 1;
            Add("encoder.conv1.weight", NState, NMels, 3);
            Add("encoder.conv1.bias", NState);
            Add("encoder.conv2.weight", NState, NState, 3);
            Add("encoder.conv2.bias", NState);
            Add("encoder.positional_embedding", nCtx, NState);
            for (var b = 0; b < NLayer; b++)
            {
                var p = $"encoder.blocks.{b}.";
                Add(p + "attn_ln.weight", NState); Add(p + "attn_ln.bias", NState);
                Add(p + "attn.query.weight", NState, NState); Add(p + "attn.query.bias", NState);
                Add(p + "attn.key.weight", NState, NState);
                Add(p + "attn.value.weight", NState, NState); Add(p + "attn.value.bias", NState);
                Add(p + "attn.out.weight", NState, NState); Add(p + "attn.out.bias", NState);
                Add(p + "mlp_ln.weight", NState); Add(p + "mlp_ln.bias", NState);
                Add(p + "mlp.0.weight", DFF, NState); Add(p + "mlp.0.bias", DFF);
                Add(p + "mlp.2.weight", NState, DFF); Add(p + "mlp.2.bias", NState);
            }
            Add("encoder.ln_post.weight", NState); Add("encoder.ln_post.bias", NState);

            var config = new WhisperConfig(51865, nCtx, NState, NHead, NLayer, 448, NState, NHead, NLayer, NMels, false);
            return new WhisperModel(config, NMels, 1, new float[NMels], Array.Empty<string>(), t);
        }
    }
}
