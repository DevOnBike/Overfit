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
        public QwenGenerationDemoTests(ITestOutputHelper output) => _out = output;

        private const string ModelPath = "c:/qwen3b/qwen.bin";
        private const string TokenizerDir = "c:/qwen3b/";
        private const int MaxNewTokens = 200;
        private const int MaxCtx = 512;

        private static readonly SamplingOptions GreedySampling = SamplingOptions.GreedyWithPenalty(1.15f, 64);

        private static readonly SamplingOptions Temp03Sampling = new(
            strategy: SamplingStrategy.TopP,
            temperature: 0.3f,
            topK: 0,
            topP: 0.9f,
            seed: 42);

        private bool TryLoad(out CachedLlamaInferenceEngine? engine, out QwenTokenizer? tok)
        {
            engine = null; tok = null;
            if (!File.Exists(ModelPath))
            {
                _out.WriteLine("SKIPPED: brak modelu w test_fixtures/qwen.bin");
                return false;
            }
            if (!File.Exists(Path.Combine(TokenizerDir, "tokenizer.json")))
            {
                _out.WriteLine("SKIPPED: brak tokenizera w test_fixtures/tokenizer/");
                return false;
            }
            engine = CachedLlamaInferenceEngine.Load(ModelPath);
            tok = QwenTokenizer.Load(TokenizerDir);
            return true;
        }

        private string Generate(
            CachedLlamaSession session,
            QwenTokenizer tok,
            int[] prompt,
            int maxNew,
            in SamplingOptions sampling)
        {
            session.Reset(prompt);
            var sb = new System.Text.StringBuilder();
            var lastTok = -1;
            var repCnt = 0;

            for (var i = 0; i < maxNew; i++)
            {
                if (session.IsFull)
                {
                    break;
                }
                var token = session.GenerateNextToken(in sampling);

                if (token == QwenTokenizer.EndOfText || token == QwenTokenizer.ImEnd)
                {
                    break;
                }

                sb.Append(tok.DecodeToken(token));

                // Prosta detekcja pętli: 6 identycznych tokenów z rzędu → stop
                if (token == lastTok) { repCnt++; if (repCnt >= 6)
                    {
                        break;
                    }
                }
                else { lastTok = token; repCnt = 0; }
            }
            return sb.ToString();
        }

        /// <summary>Format bez system message: &lt;|im_start|&gt;user\n{q}&lt;|im_end|&gt;\n&lt;|im_start|&gt;assistant\n</summary>
        private static int[] NoSystemPrompt(QwenTokenizer tok, string question)
            => new[] { QwenTokenizer.ImStart, 872, 198 }
                .Concat(tok.Encode(question))
                .Concat(new[] { QwenTokenizer.ImEnd, 198, QwenTokenizer.ImStart, 77091, 198 })
                .ToArray();

        [Fact]
        public void Demo_EncodeDecodeRoundtrip_Sanity()
        {
            if (!TryLoad(out _, out var tok))
            {
                return;
            }
            const string text = "What is 7 * 8?";
            var decoded = tok!.Decode(tok.Encode(text));
            _out.WriteLine($"Input  : '{text}'");
            _out.WriteLine($"Decoded: '{decoded}'");
            var chat = tok.BuildChatPrompt(text);
            _out.WriteLine($"Chat prompt ({chat.Length} tokens): [{string.Join(", ", chat.Take(6))}...]");
            Assert.Equal(text, decoded);
        }

        /// <summary>Greedy bez system message — najlepsza szansa dla 0.5B FP32.</summary>
        [Fact]
        public void Demo_Math_NoSystem_Greedy()
        {
            if (!TryLoad(out var engine, out var tok))
            {
                return;
            }
            using (engine)
            {
                using var session = engine!.CreateSession(MaxCtx);
                var prompt = NoSystemPrompt(tok!, "What is 2+2?");
                _out.WriteLine($"Prompt ({prompt.Length} tokens) — bez system message");
                _out.WriteLine("=== RESPONSE ===");
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var resp = Generate(session, tok!, prompt, MaxNewTokens, in GreedySampling);
                sw.Stop();
                _out.WriteLine(resp);
                _out.WriteLine($"\n{resp.Length} chars  {sw.ElapsedMilliseconds}ms");
                Assert.NotEmpty(resp);
            }
        }

        /// <summary>Temperature 0.3 bez system message — może być lepiej niż greedy.</summary>
        [Fact]
        public void Demo_Math_NoSystem_Temperature()
        {
            if (!TryLoad(out var engine, out var tok))
            {
                return;
            }
            using (engine)
            {
                using var session = engine!.CreateSession(MaxCtx);
                var prompt = NoSystemPrompt(tok!, "What is 2+2?");
                _out.WriteLine("=== RESPONSE (temp=0.3, top-p=0.9) ===");
                var resp = Generate(session, tok!, prompt, MaxNewTokens, in Temp03Sampling);
                _out.WriteLine(resp);
                Assert.NotEmpty(resp);
            }
        }

        /// <summary>Z pełnym system message Qwen2.5-Instruct.</summary>
        [Fact]
        public void Demo_Math_WithSystem_Greedy()
        {
            if (!TryLoad(out var engine, out var tok))
            {
                return;
            }
            using (engine)
            {
                using var session = engine!.CreateSession(MaxCtx);
                var prompt = tok!.BuildChatPrompt("What is 2+2?");
                _out.WriteLine($"Prompt ({prompt.Length} tokens) — z system message");
                _out.WriteLine("=== RESPONSE ===");
                var resp = Generate(session, tok, prompt, MaxNewTokens, in GreedySampling);
                _out.WriteLine(resp);
                Assert.NotEmpty(resp);
            }
        }

        [Fact]
        public void Demo_Polish_NoSystem()
        {
            if (!TryLoad(out var engine, out var tok))
            {
                return;
            }
            using (engine)
            {
                using var session = engine!.CreateSession(MaxCtx);
                var prompt = NoSystemPrompt(tok!, "Jaka jest stolica Polski?");
                _out.WriteLine("=== RESPONSE ===");
                var resp = Generate(session, tok, prompt, MaxNewTokens, in GreedySampling);
                _out.WriteLine(resp);
                Assert.NotEmpty(resp);
            }
        }

        [Fact]
        public void Demo_SpeedBenchmark()
        {
            if (!TryLoad(out var engine, out var tok))
            {
                return;
            }
            using (engine)
            {
                using var session = engine!.CreateSession(MaxCtx);
                var prompt = NoSystemPrompt(tok!, "Write a short poem about programming.");
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var resp = Generate(session, tok, prompt, MaxNewTokens, in GreedySampling);
                sw.Stop();
                var tps = resp.Length > 0 ? resp.Length * 1000.0 / sw.ElapsedMilliseconds : 0;
                _out.WriteLine($"Config: {engine.Config.NLayers}L d={engine.Config.DModel} h={engine.Config.NHeads}/{engine.Config.KvHeads}");
                _out.WriteLine($"Speed : {tps:F1} chars/s  ({sw.ElapsedMilliseconds}ms)");
                _out.WriteLine($"\n{resp}");
            }
        }
    }
}
