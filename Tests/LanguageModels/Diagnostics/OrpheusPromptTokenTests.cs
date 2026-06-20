// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Diagnostics
{
    /// <summary>
    /// Dumps how the Orpheus GGUF tokenizer maps the control strings/ids we need for the canonical prompt
    /// (start_of_human 128259, end_of_text 128009, end_of_human 128260, start_of_ai 128261, start_of_speech
    /// 128257). Confirms whether "&lt;|audio|&gt;"/"&lt;|eot_id|&gt;" already encode to the right ids and whether
    /// the audio-priming ids exist — grounding the prompt-format fix. [LongFact] — needs C:\orpheus.
    /// </summary>
    public sealed class OrpheusPromptTokenTests
    {
        private const string Path = @"C:\orpheus\orpheus-3b-0.1-ft-q4_k_m.gguf";
        private readonly ITestOutputHelper _out;
        public OrpheusPromptTokenTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Dump_Control_Token_Ids()
        {
            if (!File.Exists(Path))
            {
                _out.WriteLine("missing orpheus gguf");
                return;
            }

            var tok = GgufTokenizer.Load(Path);

            void Show(string s)
            {
                var ids = tok.Encode(s);
                _out.WriteLine($"encode(\"{s}\") = [{string.Join(", ", ids)}]  ({ids.Length} tok)");
            }

            Show("<|audio|>");
            Show("<|eot_id|>");
            Show("<custom_token_0>");
            Show("tara: hello");

            // Do the canonical numeric ids decode back to recognizable control tokens?
            foreach (var id in new[] { 128257, 128259, 128260, 128261, 128009, 128258 })
            {
                _out.WriteLine($"decode({id}) = \"{tok.DecodeToken(id)}\"");
            }

            Assert.True(true);
        }
    }
}
