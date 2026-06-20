// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Chat;
using DevOnBike.Overfit.LanguageModels.Constraints;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Tokenizers;
using DevOnBike.Overfit.LanguageModels.Tools;
using Xunit.Abstractions;

namespace DevOnBike.Overfit.Tests.LanguageModels.Runtime.Parity
{
    /// <summary>
    /// Probe: does the smallest local Qwen (2.5-0.5B-Instruct safetensors at C:\qwen3b) do the agentic
    /// tasks? Constrained decoding forces valid STRUCTURE + args + JSON — the 0.5B handles all of that.
    /// What it does NOT handle reliably is tool ROUTING (which tool): measured ~4/8 here, and it
    /// systematically over-picks <c>lookup_customer</c> (the first-listed tool) for create-ticket
    /// requests. Prompt wording barely moves it — tool selection is below a 0.5B's reasoning ceiling.
    /// Conclusion that drives the demo default: use 1.5B+ for agentic; the Local Agent demo ships 3B
    /// (which routes the same 8 cases 6/6 — proven in the demo). [LongFact] — needs the model.
    /// </summary>
    public sealed class SmallModelAgenticProbeTests
    {
        private const string Dir = @"C:\qwen3b"; // model.safetensors here is Qwen2.5-0.5B-Instruct
        private readonly ITestOutputHelper _out;
        public SmallModelAgenticProbeTests(ITestOutputHelper output) => _out = output;

        [LongFact]
        public void Qwen05B_Agentic_Probe()
        {
            if (!File.Exists(Path.Combine(Dir, "model.safetensors")))
            {
                _out.WriteLine("missing 0.5B");
                return;
            }

            using var engine = SafetensorsLlamaLoader.Load(Dir);
            using var session = engine.CreateSession(2048);
            ITokenizer tok = new QwenChatTokenizer(QwenTokenizer.Load(Dir));
            var template = ChatTemplate.Detect("<|im_start|>");
            var chat = new ChatSession(session, tok, template, ["<|im_end|>", "\n<|im_start|>"]);
            var opts = new GenerationOptions(64, 2048, SamplingOptions.Greedy, true, tok.EndOfTextTokenId);

            var tools = new List<ToolDefinition>
            {
                new("lookup_customer", "Read an existing customer's account details (plan, region, months active) by email. Use this to ANSWER QUESTIONS about a customer. Changes nothing. Arguments: { \"email\": string }."),
                new("create_ticket", "Open a NEW support ticket. Use ONLY when the user explicitly asks to open/create/file/raise a ticket. Arguments: { \"customerEmail\": string, \"subject\": string, \"priority\": \"low\"|\"normal\"|\"high\" }."),
            };

            string SelectedTool(string userMsg)
            {
                chat.ResetConversation();
                var prompt = "Pick the ONE tool whose purpose matches the request, then supply its arguments.\n" +
                             "- lookup_customer — read/answer questions about a customer (plan, region, details).\n" +
                             "- create_ticket — open a NEW support ticket (only when explicitly asked).\n\nRequest: " + userMsg;
                var reply = chat.Send(prompt, in opts, constraint: new ToolCallConstraint(tools, tok));
                return ToolCall.TryParse(reply, out var call) ? call.Name : "PARSE_FAIL";
            }

            var cases = new (string Msg, string Expect)[]
            {
                ("What plan is ada@northwind.example on?", "lookup_customer"),
                ("Is sam@brightlabs.example in the EU?", "lookup_customer"),
                ("How many months has jo@tinkergarden.example been active?", "lookup_customer"),
                ("Tell me ada@northwind.example's region.", "lookup_customer"),
                ("Open a high priority ticket for sam@brightlabs.example about a failed SSO login.", "create_ticket"),
                ("File a ticket for ada@northwind.example: billing error on last invoice.", "create_ticket"),
                ("Create a ticket: jo@tinkergarden.example cannot reset password.", "create_ticket"),
                ("Raise a support ticket for sam@brightlabs.example, export is broken.", "create_ticket"),
            };

            var correct = 0;
            var lines = new System.Text.StringBuilder();
            foreach (var (msg, expect) in cases)
            {
                var got = SelectedTool(msg);
                if (got == expect)
                {
                    correct++;
                }
                lines.Append(got == expect ? "OK  " : "XX  ").Append("want=").Append(expect).Append(" got=").Append(got).Append("  | ").AppendLine(msg);
            }
            // Reported, not asserted: this is a capability probe, not a regression gate. Observed ~4/8 —
            // all lookups right, create-ticket routing unreliable. The demo default (3B) is the fix.
            _out.WriteLine($"0.5B tool-SELECTION: {correct}/{cases.Length}\n{lines}");
        }
    }
}
