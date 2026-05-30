// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Loading;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// Loads the Polish Bielik-4.5B-v3.0-Instruct Q8_0 GGUF (no sibling tokenizer — exercises the
    /// embedded GgufTokenizer fallback in OverfitClient.LoadGguf) and asserts coherent Polish output.
    /// [LongFact] — needs C:\bielik. Flip to [Fact] to run.
    /// </summary>
    public sealed class BielikSmokeTests
    {
        private const string Path = @"C:\bielik\Bielik-4.5B-v3.0-Instruct.Q8_0.gguf";
        private readonly ITestOutputHelper _out;
        public BielikSmokeTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Bielik_Loads_And_Generates_Polish()
        {
            if (!File.Exists(Path)) { _out.WriteLine("missing Bielik gguf"); return; }

            using (var reader = new GgufReader(Path))
            {
                _out.WriteLine("architecture : " + reader.GetMeta("general.architecture", "?"));
                _out.WriteLine("tokenizer    : " + reader.GetMeta("tokenizer.ggml.model", "?"));
                var ct = reader.GetMeta("tokenizer.chat_template", "");
                var marker = ct.Contains("<|im_start|>") ? "ChatML"
                    : ct.Contains("<|start_header_id|>") ? "Llama3"
                    : ct.Contains("[INST]") ? "Mistral" : "OTHER/none";
                _out.WriteLine("chat template: " + marker);
            }

            using var client = OverfitClient.LoadGguf(Path);
            client.AddSystem("Jesteś zwięzłym, pomocnym asystentem. Odpowiadaj po polsku.");
            var reply = client.Send("Jaka jest stolica Polski? Odpowiedz jednym zdaniem.");
            var stats = client.Chat.LastStats;

            _out.WriteLine("REPLY: " + reply);
            _out.WriteLine($"tok/s {stats.TokensPerSecond:F1} | prompt {stats.PromptTokens} | gen {stats.GeneratedTokens} | alloc {stats.AllocatedBytes} B");

            Assert.False(string.IsNullOrWhiteSpace(reply));
            Assert.True(stats.GeneratedTokens > 0);
        }
    }
}
