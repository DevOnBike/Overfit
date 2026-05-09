// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime
{
    [Trait("Category", "QwenDemo")]
    [Trait("Category", "Qwen")]
    public sealed class QwenGenerationDemoTests
    {
        private const string ModelPath = "d:/qwen/qwen.bin";
        private const string TokenizerDir = "d:/qwen/";
        private const int MaxNewTokens = 200;
        private const int MaxCtx = 512;

        private static bool TryLoad(out CachedLlamaInferenceEngine? engine, out QwenTokenizer? tok)
        {
            engine = null;
            tok = null;
            if (!File.Exists(ModelPath))
            {
                Console.WriteLine($"SKIPPED: model not found at {ModelPath}");
                return false;
            }
            if (!File.Exists(Path.Combine(TokenizerDir, "tokenizer.json")))
            {
                Console.WriteLine($"SKIPPED: tokenizer not found in {TokenizerDir}");
                return false;
            }
            engine = CachedLlamaInferenceEngine.Load(ModelPath);
            tok = QwenTokenizer.Load(TokenizerDir);
            return true;
        }

        private static int[] Generate(
            CachedLlamaInferenceEngine engine, CachedLlamaSession session,
            QwenTokenizer tok, int[] prompt, int maxNew, Action<string>? onToken = null)
        {
            session.Reset(prompt);
            var sampling = SamplingOptions.Greedy;
            var output = new List<int>();
            for (var i = 0; i < maxNew; i++)
            {
                if (session.IsFull)
                {
                    break;
                }
                var token = session.GenerateNextToken(in sampling);
                output.Add(token);
                onToken?.Invoke(tok.DecodeToken(token));
                if (token == QwenTokenizer.EndOfText || token == QwenTokenizer.ImEnd)
                {
                    break;
                }
            }
            return output.ToArray();
        }

        [Fact]
        public void Demo_SimpleGreeting_GeneratesResponse()
        {
            if (!TryLoad(out var engine, out var tok))
            {
                return;
            }
            using (engine)
            {
                using var session = engine!.CreateSession(MaxCtx);
                var prompt = tok!.BuildChatPrompt("Hello! What is your name?");
                Console.WriteLine("=== PROMPT ===\n" + tok.Decode(prompt));
                Console.WriteLine("\n=== RESPONSE ===");
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var output = Generate(engine, session, tok, prompt, MaxNewTokens, t => Console.Write(t));
                sw.Stop();
                Console.WriteLine($"\n\nTokens={output.Length}  {output.Length * 1000.0 / sw.ElapsedMilliseconds:F1} tok/s");
                Assert.NotEmpty(output);
            }
        }

        [Fact]
        public void Demo_MathQuestion_AnswerContains56()
        {
            if (!TryLoad(out var engine, out var tok))
            {
                return;
            }
            using (engine)
            {
                using var session = engine!.CreateSession(MaxCtx);
                var prompt = tok!.BuildChatPrompt("What is 7 * 8?");
                Console.WriteLine("=== RESPONSE ===");
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var output = Generate(engine, session, tok, prompt, MaxNewTokens, t => Console.Write(t));
                sw.Stop();
                var response = tok.Decode(output);
                Console.WriteLine($"\n\n{output.Length * 1000.0 / sw.ElapsedMilliseconds:F1} tok/s");
                Assert.NotEmpty(output);
                Assert.Contains("56", response);
            }
        }

        [Fact]
        public void Demo_PolishQuestion_GeneratesResponse()
        {
            if (!TryLoad(out var engine, out var tok))
            {
                return;
            }
            using (engine)
            {
                using var session = engine!.CreateSession(MaxCtx);
                var prompt = tok!.BuildChatPrompt(
                    "Napisz po polsku: jaka jest stolica Polski?",
                    systemPrompt: "Jesteś pomocnym asystentem mówiącym po polsku.");
                Console.WriteLine("=== RESPONSE ===");
                var output = Generate(engine, session, tok, prompt, MaxNewTokens, t => Console.Write(t));
                Console.WriteLine();
                Assert.NotEmpty(output);
            }
        }

        [Fact]
        public void Demo_MultiTurn_RemembersName()
        {
            if (!TryLoad(out var engine, out var tok))
            {
                return;
            }
            using (engine)
            {
                using var session = engine!.CreateSession(MaxCtx);

                var prompt1 = tok!.BuildChatPrompt("My name is Maciej. Remember it.");
                Console.WriteLine("=== TURN 1 ===");
                var reply1 = Generate(engine, session, tok, prompt1, 80, t => Console.Write(t));
                Console.WriteLine();

                var history =
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" +
                    "<|im_start|>user\nMy name is Maciej. Remember it.<|im_end|>\n" +
                    "<|im_start|>assistant\n" + tok.Decode(reply1) + "<|im_end|>\n" +
                    "<|im_start|>user\nWhat is my name?<|im_end|>\n" +
                    "<|im_start|>assistant\n";

                Console.WriteLine("\n=== TURN 2 ===");
                var reply2 = Generate(engine, session, tok, tok.Encode(history), 80, t => Console.Write(t));
                Console.WriteLine();

                Assert.NotEmpty(reply2);
                Assert.Contains("Maciej", tok.Decode(reply2));
            }
        }

        [Fact]
        public void Demo_SpeedBenchmark_ReportsTokensPerSecond()
        {
            if (!TryLoad(out var engine, out var tok))
            {
                return;
            }
            using (engine)
            {
                using var session = engine!.CreateSession(MaxCtx);
                var prompt = tok!.BuildChatPrompt(
                    "Write a detailed explanation of how neural networks work, " +
                    "including backpropagation, activation functions, and training.");

                var sw = System.Diagnostics.Stopwatch.StartNew();
                var output = Generate(engine, session, tok, prompt, MaxNewTokens);
                sw.Stop();

                var tokPerSec = output.Length * 1000.0 / sw.ElapsedMilliseconds;
                Console.WriteLine($"Generated : {output.Length} tokens");
                Console.WriteLine($"Speed     : {tokPerSec:F1} tok/s  " +
                                  $"({sw.ElapsedMilliseconds / (double)output.Length:F1} ms/tok)");
                Console.WriteLine($"Config    : {engine.Config.NLayers}L " +
                                  $"d={engine.Config.DModel} h={engine.Config.NHeads}/{engine.Config.KvHeads}");
                Console.WriteLine($"\n{tok.Decode(output)}");

                Assert.True(tokPerSec > 0);
            }
        }
    }
}
