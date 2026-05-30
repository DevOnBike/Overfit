// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using System.Text.Json;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Constraints;
using DevOnBike.Overfit.LanguageModels.Tools;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Loading
{
    /// <summary>
    /// The real open question for Bielik: do the CONSTRAINED paths (tool calling + guaranteed JSON)
    /// work on its SPM (llama) tokenizer? The constraints build their per-token text table by decoding
    /// every single token id (SPM uses the ▁ space marker and &lt;0xNN&gt; byte fallbacks), so chat
    /// working does not by itself prove the masks behave. This loads Bielik-4.5B Q8_0 and asserts a
    /// valid Polish tool call (with the right argument schema) plus parseable JSON.
    /// [LongFact] — needs C:\bielik. Flip to [Fact] to run.
    /// </summary>
    public sealed class BielikAgenticTests
    {
        private const string Path = @"C:\bielik\Bielik-4.5B-v3.0-Instruct.Q8_0.gguf";
        private readonly ITestOutputHelper _out;
        
        public BielikAgenticTests(ITestOutputHelper output) => _out = output;

        private static readonly ToolDefinition[] Tools =
        [
            new(
                "lookup_customer",
                "Odczytaj dane istniejącego klienta (plan, region) po adresie email. Nic nie zmienia.",
                [new ToolParameter("email")]),
            new(
                "create_ticket",
                "Załóż NOWE zgłoszenie serwisowe — tylko gdy użytkownik wprost prosi o założenie/utworzenie zgłoszenia.",
                [
                    new ToolParameter("customerEmail"),
                    new ToolParameter("subject"),
                    new ToolParameter("priority"),
                ]),
        ];

        [LongFact]
        public void Bielik_ToolCalling_And_Json_Work_OnSpm()
        {
            if (!File.Exists(Path)) { _out.WriteLine("missing Bielik gguf"); return; }

            using var client = OverfitClient.LoadGguf(Path);

            // ── Tool routing: a "lookup" request and a "create ticket" request ──
            var lookup = CallTool(client, "Na jakim planie jest klient anna@firma.pl?");
            _out.WriteLine($"[lookup]  tool={lookup.Name}  args={lookup.Arguments}");

            var ticket = CallTool(client, "Załóż zgłoszenie reklamacyjne o wysokim priorytecie dla klienta anna@firma.pl.");
            _out.WriteLine($"[ticket]  tool={ticket.Name}  args={ticket.Arguments}");

            // ── Guaranteed JSON ──
            var json = client.Complete("Zwróć obiekt JSON z polami imie oraz miasto dla Anny, która mieszka w Krakowie.",
                constraint: new JsonGrammarConstraint(client.Tokenizer));
            _out.WriteLine($"[json]    {json}");

            // Assertions: both calls parse, route correctly, and carry the exact schema keys.
            Assert.Equal("lookup_customer", lookup.Name);
            using (var la = JsonDocument.Parse(lookup.Arguments))
            {
                Assert.True(la.RootElement.TryGetProperty("email", out _), "lookup args missing 'email'");
            }

            Assert.Equal("create_ticket", ticket.Name);
            using (var ta = JsonDocument.Parse(ticket.Arguments))
            {
                Assert.True(ta.RootElement.TryGetProperty("customerEmail", out _), "ticket args missing 'customerEmail'");
                Assert.True(ta.RootElement.TryGetProperty("subject", out _), "ticket args missing 'subject'");
                Assert.True(ta.RootElement.TryGetProperty("priority", out _), "ticket args missing 'priority'");
            }

            // JSON mode output must be well-formed by construction.
            using var jd = JsonDocument.Parse(json);
            Assert.Equal(JsonValueKind.Object, jd.RootElement.ValueKind);
        }

        private static ToolCall CallTool(OverfitClient client, string message)
        {
            var constraint = new ToolCallConstraint(Tools, client.Tokenizer);
            var reply = client.Complete(BuildToolPrompt(message), constraint: constraint);
            
            if (!ToolCall.TryParse(reply, out var call))
            {
                throw new InvalidOperationException($"Constrained reply did not parse as a tool call: '{reply}'.");
            }
            return call;
        }

        private static string BuildToolPrompt(string message)
        {
            var sb = new StringBuilder();
            
            sb.AppendLine("Wybierz JEDNO narzędzie pasujące do prośby, a następnie podaj jego argumenty:");
            foreach (var tool in Tools)
            {
                sb.Append("- ").Append(tool.Name).Append(": ").AppendLine(tool.Description);
            }
            sb.AppendLine();
            sb.Append("Prośba: ").AppendLine(message);
            
            return sb.ToString();
        }
    }
}
