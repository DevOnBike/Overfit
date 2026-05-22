// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Tests.LanguageModels.Chat
{
    /// <summary>
    /// Proves the turnkey <see cref="ChatSession"/> drives the Llama/Qwen runtime
    /// (<see cref="CachedLlamaSession"/>) now that it implements <see cref="ISlmSession"/> —
    /// the path the GGUF / safetensors loaders feed. A tiny random-weight Llama is built,
    /// loaded through <see cref="CachedLlamaInferenceEngine.Load"/>, and chatted with via
    /// a trivial char-level stub tokenizer (linguistic content is irrelevant; the point is
    /// that render → tokenize → prefill → decode → detokenize runs through the contract).
    /// </summary>
    public sealed class ChatSessionLlamaTests
    {
        private const int Vocab = 16, D = 8, Heads = 2, Kv = 1, Hd = D / Heads, DFF = 16, Layers = 1, Ctx = 256;
        private const int EotId = Vocab - 1;

        [Fact]
        public void ChatSession_DrivesLlamaRuntime_ProducesReplyAndAdvances()
        {
            using var engine = BuildTinyLlama(seed: 7);
            using var session = engine.CreateSession(Ctx);
            var tokenizer = new CharStubTokenizer();
            var chat = new ChatSession(session, tokenizer, new ChatTemplate(ChatTemplateFormat.ChatML));
            chat.AddSystem("You are concise.");

            var options = new GenerationOptions(
                maxNewTokens: 4,
                maxContextLength: Ctx,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: true,
                endOfTextTokenId: EotId);

            var streamed = new StringBuilder();
            var reply = chat.Send("Hello there", in options, s => streamed.Append(s));

            // The wiring ran end-to-end: history has the user + assistant turns, the
            // session consumed the prompt (CurrentPosition advanced past 0), and the
            // streamed callback received exactly what the return value reports.
            Assert.Equal(3, chat.History.Count);                 // system + user + assistant
            Assert.Equal("assistant", chat.History[^1].Role);
            Assert.True(session.CurrentPosition > 0);
            Assert.Equal(reply, streamed.ToString());
            Assert.True(session.CurrentPosition <= session.MaxContextLength);
        }

        [Fact]
        public void CachedLlamaSession_Generate_RespectsMaxNewTokens()
        {
            using var engine = BuildTinyLlama(seed: 3);
            using var session = engine.CreateSession(Ctx);

            int[] prompt = [1, 2, 3, 4];
            var output = new int[5];
            var options = new GenerationOptions(
                maxNewTokens: 5,
                maxContextLength: Ctx,
                sampling: SamplingOptions.Greedy,
                stopOnEndOfTextToken: false);

            var produced = session.Generate(prompt, output, in options);

            Assert.Equal(5, produced);
            Assert.Equal(prompt.Length + produced, session.CurrentPosition);

            var logits = new float[Vocab];
            session.GetLastLogits(logits);
            for (var i = 0; i < Vocab; i++)
            {
                Assert.False(float.IsNaN(logits[i]) || float.IsInfinity(logits[i]));
            }
        }

        // ── tiny random-weight Llama through the validated .bin load path ───────
        private static CachedLlamaInferenceEngine BuildTinyLlama(int seed)
        {
            var rng = new Random(seed);
            using var ms = new MemoryStream();
            using (var bw = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
            {
                bw.Write(0x4F565246u);                          // magic
                bw.Write(2);                                    // version
                bw.Write(Layers); bw.Write(D); bw.Write(Heads); bw.Write(Kv);
                bw.Write(Vocab); bw.Write(Ctx); bw.Write(DFF);
                bw.Write(1);                                    // use_rope
                bw.Write(10_000f);                              // rope_theta
                bw.Write((int)FeedForwardActivation.SwiGLU);
                bw.Write(0);                                    // tie_weights = false

                Rand(bw, rng, Vocab * D);                       // embed
                Rand(bw, rng, D); Zeros(bw, D);                 // attn norm gamma/beta
                for (var h = 0; h < Heads; h++) { Rand(bw, rng, D * Hd); Zeros(bw, Hd); }       // wq/bq
                for (var k = 0; k < Kv; k++)
                {
                    Rand(bw, rng, D * Hd); Zeros(bw, Hd);       // wk/bk
                    Rand(bw, rng, D * Hd); Zeros(bw, Hd);       // wv/bv
                }
                for (var h = 0; h < Heads; h++) { Rand(bw, rng, Hd * D); Zeros(bw, D); }        // wo/bo
                Rand(bw, rng, D); Zeros(bw, D);                 // ffn norm gamma/beta
                Rand(bw, rng, D * DFF);                         // gate
                Rand(bw, rng, D * DFF);                         // up
                Rand(bw, rng, DFF * D);                         // down
                Rand(bw, rng, D); Zeros(bw, D);                 // final norm gamma/beta
                Rand(bw, rng, Vocab * D);                       // lm head
            }

            ms.Position = 0;
            using var br = new BinaryReader(ms);
            return CachedLlamaInferenceEngine.Load(br);
        }

        private static void Rand(BinaryWriter bw, Random rng, int n)
        {
            for (var i = 0; i < n; i++) { bw.Write((float)(rng.NextDouble() - 0.5) * 0.2f); }
        }

        private static void Zeros(BinaryWriter bw, int n)
        {
            for (var i = 0; i < n; i++) { bw.Write(0f); }
        }

        // Char-level identity tokenizer: one token per char in [1, Vocab-2]; never emits
        // the unknown (0) or end-of-text (Vocab-1) ids from text, so prompts never self-stop.
        private sealed class CharStubTokenizer : ITokenizer
        {
            public int VocabularySize => Vocab;
            public int EndOfTextTokenId => EotId;
            public int UnknownTokenId => 0;
            public bool SupportsZeroAllocationEncode => true;
            public bool SupportsZeroAllocationDecode => true;

            public int CountTokens(ReadOnlySpan<char> text) => text.Length;

            public int Encode(ReadOnlySpan<char> text, Span<int> destination)
            {
                for (var i = 0; i < text.Length; i++)
                {
                    destination[i] = (text[i] % (Vocab - 2)) + 1;   // [1, Vocab-2]
                }
                return text.Length;
            }

            public int Decode(ReadOnlySpan<int> tokens, Span<char> destination)
            {
                for (var i = 0; i < tokens.Length; i++)
                {
                    destination[i] = (char)('a' + (tokens[i] % 26));
                }
                return tokens.Length;
            }

            public string DecodeToString(ReadOnlySpan<int> tokens)
            {
                var sb = new StringBuilder(tokens.Length);
                foreach (var t in tokens) { sb.Append((char)('a' + (t % 26))); }
                return sb.ToString();
            }
        }
    }
}
