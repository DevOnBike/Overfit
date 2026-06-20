// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// A/B decode-throughput comparison for Bielik-4.5B: Q8_0 (~4.8 GB) vs Q4_K_M (~2.7 GB) on the same
    /// Polish chat prompt. Decode is memory-bandwidth-bound, so the lighter K-quant should be markedly
    /// faster. Prints tok/s + a snippet of each reply (to confirm Q4_K_M stays coherent).
    /// [LongFact] — needs both GGUFs at C:\bielik. Flip to [Fact] to run.
    /// </summary>
    public sealed class BielikSpeedTests
    {
        private const string Q8 = @"C:\bielik\Bielik-4.5B-v3.0-Instruct.Q8_0.gguf";
        private const string Q4 = @"C:\bielik\Bielik-4.5B-v3.0-Instruct-Q4_K_M.gguf";
        private const string Prompt = "Wymień trzy największe miasta w Polsce i opisz każde jednym zdaniem.";

        private readonly ITestOutputHelper _out;
        public BielikSpeedTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Bielik_Q4KM_vs_Q8_DecodeSpeed()
        {
            Bench(Q8, "Q8_0  ");
            Bench(Q4, "Q4_K_M");
        }

        private void Bench(string path, string label)
        {
            if (!File.Exists(path))
            {
                _out.WriteLine($"{label}: missing {path}");
                return;
            }

            using var client = OverfitClient.LoadGguf(path, mmap: true);
            client.AddSystem("Jesteś zwięzłym asystentem. Odpowiadaj po polsku.");

            // Warm-up turn (excluded), then the measured turn.
            client.Send("Cześć.");
            client.Reset();
            client.AddSystem("Jesteś zwięzłym asystentem. Odpowiadaj po polsku.");

            var reply = client.Send(Prompt);
            var s = client.Chat.LastStats;

            var snippet = reply.Replace("\n", " ");
            if (snippet.Length > 110)
            {
                snippet = snippet[..110] + "…";
            }
            _out.WriteLine($"{label}: {s.TokensPerSecond,5:F1} tok/s | gen {s.GeneratedTokens} | alloc {s.AllocatedBytes} B | {snippet}");

            Assert.False(string.IsNullOrWhiteSpace(reply));
        }
    }
}
