// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    [Trait("Category", "QwenDemo")]
    [Trait("Category", "Qwen")]
    public sealed class QwenGenerationDemoTests
    {
        private readonly ITestOutputHelper _out;

        public QwenGenerationDemoTests(ITestOutputHelper output)
        {
            _out = output;
        }

        private const string ModelPath = "d:/qwen/qwen.bin";
        private const string TokenizerDir = "d:/qwen/";
        private const int MaxNewTokens = 200;
        private const int MaxCtx = 512;

        private bool TryLoad(out CachedLlamaInferenceEngine? engine, out QwenTokenizer? tok)
        {
            engine = null;
            tok = null;
            if (!File.Exists(ModelPath))
            {
                _out.WriteLine($"SKIPPED: model not found at {ModelPath}");
                return false;
            }
            if (!File.Exists(Path.Combine(TokenizerDir, "tokenizer.json")))
            {
                _out.WriteLine($"SKIPPED: tokenizer not found in {TokenizerDir}");
                return false;
            }
            engine = CachedLlamaInferenceEngine.Load(ModelPath);
            tok = QwenTokenizer.Load(TokenizerDir);
            return true;
        }

        private int[] Generate(
            CachedLlamaInferenceEngine engine, CachedLlamaSession session,
            QwenTokenizer tok, int[] prompt, int maxNew, bool stream = true)
        {
            session.Reset(prompt);
            var sampling = SamplingOptions.Greedy;
            var output = new List<int>();
            var sb = new System.Text.StringBuilder();

            for (var i = 0; i < maxNew; i++)
            {
                if (session.IsFull) break;
                var token = session.GenerateNextToken(in sampling);
                output.Add(token);
                if (stream) sb.Append(tok.DecodeToken(token));
                if (token == QwenTokenizer.EndOfText || token == QwenTokenizer.ImEnd) break;
            }

            if (stream) _out.WriteLine(sb.ToString());
            return output.ToArray();
        }

        // ── Tests ──────────────────────────────────────────────────────────

        [Fact]
        public void Demo_EncodeDecodeRoundtrip_Sanity()
        {
            if (!TryLoad(out _, out var tok)) return;

            const string text = "What is 7 * 8?";
            var tokens = tok!.Encode(text);
            var decoded = tok.Decode(tokens);

            _out.WriteLine($"Input   : '{text}'");
            _out.WriteLine($"Tokens  : [{string.Join(", ", tokens)}]");
            _out.WriteLine($"Decoded : '{decoded}'");

            var chat = tok.BuildChatPrompt(text);
            _out.WriteLine($"Chat prompt ({chat.Length} tokens): [{string.Join(", ", chat)}]");
            _out.WriteLine($"Expected first token: {QwenTokenizer.ImStart} (<|im_start|>), actual: {chat[0]}");

            Assert.Equal(text, decoded);
            Assert.Equal(QwenTokenizer.ImStart, chat[0]);
        }

        [Fact]
        public void Demo_SimpleGreeting_GeneratesResponse()
        {
            if (!TryLoad(out var engine, out var tok)) return;
            using (engine)
            {
                using var session = engine!.CreateSession(MaxCtx);
                var prompt = tok!.BuildChatPrompt("Hello! What is your name?");
                _out.WriteLine($"Prompt: {prompt.Length} tokens");
                _out.WriteLine("=== RESPONSE ===");
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var output = Generate(engine, session, tok, prompt, MaxNewTokens);
                sw.Stop();
                _out.WriteLine($"Tokens={output.Length}  {output.Length * 1000.0 / sw.ElapsedMilliseconds:F1} tok/s");
                Assert.NotEmpty(output);
            }
        }

        [Fact]
        public void Demo_MathQuestion_GeneratesResponse()
        {
            if (!TryLoad(out var engine, out var tok)) return;
            using (engine)
            {
                using var session = engine!.CreateSession(MaxCtx);
                var prompt = tok!.BuildChatPrompt("What is 7 * 8?");
                _out.WriteLine("=== RESPONSE ===");
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var output = Generate(engine, session, tok, prompt, MaxNewTokens);
                sw.Stop();
                _out.WriteLine($"{output.Length * 1000.0 / sw.ElapsedMilliseconds:F1} tok/s");
                _out.WriteLine($"First 20 IDs: [{string.Join(", ", output.Take(20))}]");
                Assert.NotEmpty(output);
            }
        }

        [Fact]
        public void Demo_PolishQuestion_GeneratesResponse()
        {
            if (!TryLoad(out var engine, out var tok)) return;
            using (engine)
            {
                using var session = engine!.CreateSession(MaxCtx);
                var prompt = tok!.BuildChatPrompt(
                    "Napisz po polsku: jaka jest stolica Polski?",
                    systemPrompt: "Jesteś pomocnym asystentem mówiącym po polsku.");
                _out.WriteLine("=== RESPONSE ===");
                var output = Generate(engine, session, tok, prompt, MaxNewTokens);
                Assert.NotEmpty(output);
            }
        }

        [Fact]
        public void Demo_SpeedBenchmark_ReportsTokensPerSecond()
        {
            if (!TryLoad(out var engine, out var tok)) return;
            using (engine)
            {
                using var session = engine!.CreateSession(MaxCtx);
                var prompt = tok!.BuildChatPrompt("Write a short poem about C# programming.");

                var sw = System.Diagnostics.Stopwatch.StartNew();
                var output = Generate(engine, session, tok, prompt, MaxNewTokens, stream: false);
                sw.Stop();

                var tokPerSec = output.Length * 1000.0 / sw.ElapsedMilliseconds;
                _out.WriteLine($"Generated : {output.Length} tokens");
                _out.WriteLine($"Speed     : {tokPerSec:F1} tok/s  ({sw.ElapsedMilliseconds / (double)output.Length:F1} ms/tok)");
                _out.WriteLine($"Config    : {engine.Config.NLayers}L d={engine.Config.DModel} h={engine.Config.NHeads}/{engine.Config.KvHeads}");
                _out.WriteLine(tok.Decode(output));

                Assert.True(tokPerSec > 0);
            }
        }
    }
}
