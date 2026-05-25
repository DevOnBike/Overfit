// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.Tests.TestSupport;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Chat
{
    /// <summary>
    /// Validates the family-generic path on a NON-Qwen model (Llama-3 / Mistral) when one is
    /// present at <c>C:\llama3</c> (or <c>OVERFIT_LLAMA_DIR</c>) — confirms the generic
    /// <see cref="HuggingFaceBpeTokenizer"/> handles a different pre-tokenizer and the loader
    /// handles the family's dims. Both [LongFact] and no-op with a log when the model is
    /// absent (so they never fail for a missing artifact). The tokenizer test needs ONLY
    /// <c>tokenizer.json</c> (the minimal download); the chat test needs the weights too.
    /// </summary>
    public sealed class HuggingFaceLlamaModelTests
    {
        private readonly ITestOutputHelper _out;
        public HuggingFaceLlamaModelTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Tokenizer_RoundTrips_OnLlamaFamily()
        {
            if (!File.Exists(TestModelPaths.Llama.TokenizerJsonPath))
            {
                _out.WriteLine($"No tokenizer.json at {TestModelPaths.Llama.Dir} — skipping.");
                return;
            }

            var tok = HuggingFaceBpeTokenizer.Load(TestModelPaths.Llama.Dir);
            _out.WriteLine($"Llama-family vocab={tok.VocabularySize}, eos={tok.EndOfTextTokenId}");

            string[] inputs =
            [
                "Hello, world! How are you?",
                "The quick brown fox jumps over the lazy dog.",
                "Cześć, jak się masz? 🚀",
                "1234567",
            ];
            foreach (var input in inputs)
            {
                var ids = tok.Encode(input);
                Assert.NotEmpty(ids);
                Assert.Equal(input, tok.DecodeToString(ids));   // byte-level BPE round-trips
            }
        }

        // Strong LOADER correctness check for a BASE model (no instruct tuning): a raw greedy
        // completion of a near-deterministic factual prompt. A correctly mapped Llama-3 (incl.
        // the RoPE row-permute on Llama dims + GQA + tied LM head) says " Paris"; a mapping bug
        // yields word-salad. The chat test below is structural-only because the dropped model
        // is a base, not an instruct, model.
        [LongFact]
        public void Generate_RealLlama1B_FromSafetensors_CompletesCoherently()
        {
            if (!File.Exists(TestModelPaths.Llama.SafetensorsPath))
            {
                _out.WriteLine($"No model.safetensors at {TestModelPaths.Llama.Dir} — skipping.");
                return;
            }

            using var engine = SafetensorsLlamaLoader.Load(TestModelPaths.Llama.Dir, quantize: false);
            var tok = HuggingFaceBpeTokenizer.Load(TestModelPaths.Llama.Dir);
            using var session = engine.CreateSession(256);

            session.Reset(tok.Encode("The capital of France is"));
            var sampling = SamplingOptions.Greedy;
            var generated = new List<int>();
            for (var i = 0; i < 6 && !session.IsFull; i++)
            {
                var t = session.GenerateNextToken(in sampling);
                if (t == tok.EndOfTextTokenId) { break; }
                generated.Add(t);
            }
            var completion = tok.DecodeToString(CollectionsMarshal.AsSpan(generated));
            _out.WriteLine($"completion: '{completion}'");

            var logits = new float[engine.Config.VocabSize];
            session.GetLastLogits(logits);
            for (var i = 0; i < logits.Length; i++)
            {
                Assert.False(float.IsNaN(logits[i]) || float.IsInfinity(logits[i]));
            }
            Assert.Contains("Paris", completion, StringComparison.OrdinalIgnoreCase);
        }

        [LongFact]
        public void Chat_OnLlamaFamily_Responds()
        {
            if (!File.Exists(TestModelPaths.Llama.SafetensorsPath))
            {
                _out.WriteLine($"No model.safetensors at {TestModelPaths.Llama.Dir} — skipping.");
                return;
            }

            using var model = HuggingFaceChatModel.LoadFromDirectory(TestModelPaths.Llama.Dir, maxContextLength: 512, quantize: false);
            _out.WriteLine($"Detected chat format: {model.Format}");

            model.Chat.AddSystem("You are a concise assistant. Answer in one short sentence.");
            var options = new GenerationOptions(
                maxNewTokens: 32, maxContextLength: 512, sampling: SamplingOptions.Greedy, stopOnEndOfTextToken: true);

            var reply = model.Chat.Send("What is the capital of France?", in options);
            _out.WriteLine($"reply: '{reply}'");

            // Structural only — we don't control which Llama variant (base vs instruct) is dropped.
            Assert.False(string.IsNullOrWhiteSpace(reply), "Chat produced no text.");
            Assert.Equal("assistant", model.Chat.History[^1].Role);
        }
    }
}
