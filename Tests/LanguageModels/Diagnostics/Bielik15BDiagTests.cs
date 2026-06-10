// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Diagnostic for the community Bielik-1.5B-v3.0 imatrix Q4_K_M GGUF that produced word salad through the
    /// server. Isolates template-vs-weights: a RAW completion (no chat template) of a Polish prefix. Coherent
    /// continuation → the chat path is at fault; garbage → the file/loader path is at fault.
    /// [LongFact] — needs C:\bielik.
    /// </summary>
    public sealed class Bielik15BDiagTests
    {
        private const string Path = @"C:\bielik\bielik-1.5b-v3.0-instruct-q4_k_m-imat.gguf";
        private readonly ITestOutputHelper _out;

        public Bielik15BDiagTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void RawCompletion_PolishPrefix()
        {
            if (!File.Exists(Path)) { _out.WriteLine("missing gguf"); return; }

            using var engine = CachedLlamaInferenceEngine.LoadGguf(Path);
            var tok = GgufTokenizer.Load(Path);

            var prompt = tok.Encode("Stolicą Polski jest");
            _out.WriteLine($"prompt ids: [{string.Join(",", prompt)}]");

            using var session = engine.CreateSession(256);
            session.Reset(prompt);
            var sampling = SamplingOptions.Greedy;
            var ids = new List<int>();
            for (var i = 0; i < 24 && !session.IsFull; i++)
            {
                ids.Add(session.GenerateNextToken(in sampling));
            }
            _out.WriteLine($"gen ids: [{string.Join(",", ids)}]");
            _out.WriteLine($"RAW: \"{tok.Decode(ids.ToArray())}\"");

            // ChatML path too — what the server effectively renders.
            var chat = tok.Encode("<|im_start|>user\nJakie jest największe miasto w Polsce?<|im_end|>\n<|im_start|>assistant\n");
            using var session2 = engine.CreateSession(256);
            session2.Reset(chat);
            var ids2 = new List<int>();
            for (var i = 0; i < 40 && !session2.IsFull; i++)
            {
                var t = session2.GenerateNextToken(in sampling);
                if (t == 4) { break; }
                ids2.Add(t);
            }
            _out.WriteLine($"CHAT: \"{tok.Decode(ids2.ToArray())}\"");
        }
    }
}
