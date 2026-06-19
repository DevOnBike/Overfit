// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Whisper;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Whisper
{
    /// <summary>
    /// Structural validation of <see cref="WhisperDecoder"/> on a tiny synthetic model (real whisper.cpp
    /// tensor names: causal self-attn + cross-attn to the encoder output + tied-embedding logits): forward
    /// produces vocab-sized finite deterministic logits, and the greedy <c>Decode</c> loop runs, terminates,
    /// and emits valid token ids. Numerical parity vs whisper.cpp is S5.
    /// </summary>
    public sealed class WhisperDecoderTests
    {
        private const int NState = 8, NHead = 2, NLayer = 2, NVocab = 40, NTextCtx = 16, NCtx = 4, DFF = 4 * NState;

        private readonly ITestOutputHelper _out;
        public WhisperDecoderTests(ITestOutputHelper output) => _out = output;

        [Fact]
        public void Forward_ProducesVocabLogits_Finite_Deterministic()
        {
            var model = BuildTinyDecoder(seed: 4);
            var dec = new WhisperDecoder(model);
            var encOut = RandomEncoderOut(seed: 11);

            var tokens = new[] { 1, 5, 9, 3 };
            var a = dec.Forward(tokens, encOut, NCtx);
            var b = dec.Forward(tokens, encOut, NCtx);

            Assert.Equal(NVocab, a.Length);
            for (var i = 0; i < a.Length; i++)
            {
                Assert.True(float.IsFinite(a[i]), $"logit {i} not finite");
                Assert.Equal(a[i], b[i]); // deterministic
            }
        }

        [Fact]
        public void Decode_GreedyLoop_Terminates_EmitsValidTokens()
        {
            var model = BuildTinyDecoder(seed: 7);
            var dec = new WhisperDecoder(model);
            var encOut = RandomEncoderOut(seed: 12);

            var prompt = new[] { 1, 2 };
            const int eot = NVocab - 1;
            var produced = dec.Decode(encOut, NCtx, prompt, endOfTranscript: eot, maxNewTokens: 8);

            _out.WriteLine($"produced {produced.Length} tokens: [{string.Join(",", produced)}]");
            Assert.True(produced.Length <= 8);
            foreach (var id in produced)
            {
                Assert.InRange(id, 0, NVocab - 1);
                Assert.NotEqual(eot, id); // eot is the stop signal, never emitted
            }
            // Deterministic.
            var again = dec.Decode(encOut, NCtx, prompt, eot, 8);
            Assert.Equal(produced, again);
        }

        [Fact]
        public void DecodeCached_MatchesUncached()
        {
            var model = BuildTinyDecoder(seed: 13);
            var dec = new WhisperDecoder(model);
            var encOut = RandomEncoderOut(seed: 21);
            var prompt = new[] { 1, 2 };
            const int eot = NVocab - 1;

            var slow = dec.Decode(encOut, NCtx, prompt, eot, maxNewTokens: 8);
            var fast = dec.DecodeCached(encOut, NCtx, prompt, eot, maxNewTokens: 8);

            _out.WriteLine($"uncached: [{string.Join(",", slow)}]  cached: [{string.Join(",", fast)}]");
            Assert.Equal(slow, fast);
        }

        private static float[] RandomEncoderOut(int seed)
        {
            var rng = new Random(seed);
            var enc = new float[NCtx * NState];
            for (var i = 0; i < enc.Length; i++)
            {
                enc[i] = (float)(rng.NextDouble() * 2 - 1);
            }
            return enc;
        }

        private static WhisperModel BuildTinyDecoder(int seed)
        {
            var rng = new Random(seed);
            var t = new Dictionary<string, WhisperTensor>();

            void Add(string name, params int[] shape)
            {
                long n = 1;
                foreach (var d in shape)
                {
                    n *= d;
                }
                var data = new float[n];
                for (var i = 0; i < n; i++)
                {
                    data[i] = (float)(rng.NextDouble() * 2 - 1) * 0.2f;
                }
                t[name] = new WhisperTensor(shape, data);
            }

            Add("decoder.token_embedding.weight", NVocab, NState);
            Add("decoder.positional_embedding", NTextCtx, NState);
            for (var b = 0; b < NLayer; b++)
            {
                var p = $"decoder.blocks.{b}.";
                Add(p + "attn_ln.weight", NState);
                Add(p + "attn_ln.bias", NState);
                Add(p + "attn.query.weight", NState, NState);
                Add(p + "attn.query.bias", NState);
                Add(p + "attn.key.weight", NState, NState);
                Add(p + "attn.value.weight", NState, NState);
                Add(p + "attn.value.bias", NState);
                Add(p + "attn.out.weight", NState, NState);
                Add(p + "attn.out.bias", NState);
                Add(p + "cross_attn_ln.weight", NState);
                Add(p + "cross_attn_ln.bias", NState);
                Add(p + "cross_attn.query.weight", NState, NState);
                Add(p + "cross_attn.query.bias", NState);
                Add(p + "cross_attn.key.weight", NState, NState);
                Add(p + "cross_attn.value.weight", NState, NState);
                Add(p + "cross_attn.value.bias", NState);
                Add(p + "cross_attn.out.weight", NState, NState);
                Add(p + "cross_attn.out.bias", NState);
                Add(p + "mlp_ln.weight", NState);
                Add(p + "mlp_ln.bias", NState);
                Add(p + "mlp.0.weight", DFF, NState);
                Add(p + "mlp.0.bias", DFF);
                Add(p + "mlp.2.weight", NState, DFF);
                Add(p + "mlp.2.bias", NState);
            }
            Add("decoder.ln.weight", NState);
            Add("decoder.ln.bias", NState);

            var config = new WhisperConfig(NVocab, NCtx, NState, NHead, NLayer, NTextCtx, NState, NHead, NLayer, 80, false);
            return new WhisperModel(config, 80, 1, new float[80], Array.Empty<string>(), t);
        }
    }
}
