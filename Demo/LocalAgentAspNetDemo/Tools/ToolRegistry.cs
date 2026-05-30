// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Tools;

namespace DevOnBike.Overfit.Demo.LocalAgent.Tools
{
    /// <summary>
    /// The C# tools the agent may call, plus the in-memory back-office data they act on. Each tool is
    /// a <see cref="ToolDefinition"/> (name + description the model reads) paired with a
    /// <c>Func&lt;JsonElement, string&gt;</c> handler (the actual C# code that runs when the model
    /// picks it). Constrained decoding forces the model to emit exactly one of these tool names with a
    /// well-formed JSON arguments object; this registry deserialises the arguments and dispatches.
    ///
    /// In a real deployment these handlers would hit your CRM, ticketing system, or internal APIs.
    /// Here they operate on a seeded in-memory store so the demo runs with no external dependencies.
    /// </summary>
    public sealed class ToolRegistry
    {
        private readonly List<ToolDefinition> _definitions = [];
        private readonly Dictionary<string, Func<JsonElement, string>> _handlers = new(StringComparer.Ordinal);

        private readonly Dictionary<string, Customer> _customersByEmail;
        private readonly List<Ticket> _tickets = [];
        private int _nextTicketNumber = 1042;

        public ToolRegistry()
        {
            _customersByEmail = SeedCustomers();

            // Description = the prompt hint the model uses to ROUTE (the constraint masks on the tool NAME,
            // not the description). Purpose-first wording — "what it's for", contrasted against the other
            // tool — is what makes routing reliable. (A 0.5B routes the wrong tool here regardless of
            // wording — tool selection is below its reasoning ceiling; use 1.5B+ for agentic. See
            // SmallModelAgenticProbeTests.)
            // Parameters are enforced by the constraint: the model is forced to emit exactly these keys,
            // in order, as strings — so the handler can never receive an invented or missing argument.
            Register(
                new ToolDefinition(
                    "lookup_customer",
                    "Read an existing customer's account details (plan, region, months active) by email. " +
                    "Use this to ANSWER QUESTIONS about a customer. Changes nothing.",
                    [new ToolParameter("email")]),
                LookupCustomer);

            Register(
                new ToolDefinition(
                    "create_ticket",
                    "Open a NEW support ticket. Use ONLY when the user explicitly asks to open / create / " +
                    "file / raise a ticket for a problem.",
                    [
                        new ToolParameter("customerEmail"),
                        new ToolParameter("subject"),
                        new ToolParameter("priority"),
                    ]),
                CreateTicket);
        }

        /// <summary>The tool menu presented to the model (drives the name-enum in the constraint).</summary>
        public IReadOnlyList<ToolDefinition> Definitions => _definitions;

        /// <summary>
        /// Dispatches a parsed <see cref="ToolCall"/> to its handler. The arguments string is guaranteed
        /// well-formed JSON by the constraint, but field presence/typing is validated here (that's the
        /// handler's job until per-tool JSON-Schema argument enforcement lands).
        /// </summary>
        public string Dispatch(ToolCall call)
        {
            if (!_handlers.TryGetValue(call.Name, out var handler))
            {
                return JsonError($"Unknown tool '{call.Name}'.");
            }

            JsonElement args;
            try
            {
                using var doc = JsonDocument.Parse(string.IsNullOrWhiteSpace(call.Arguments) ? "{}" : call.Arguments);
                args = doc.RootElement.Clone();
            }
            catch (JsonException ex)
            {
                return JsonError($"Arguments were not valid JSON: {ex.Message}");
            }

            return handler(args);
        }

        private void Register(ToolDefinition definition, Func<JsonElement, string> handler)
        {
            _definitions.Add(definition);
            _handlers[definition.Name] = handler;
        }

        // ── Tool: lookup_customer ─────────────────────────────────────────────

        private string LookupCustomer(JsonElement args)
        {
            if (!TryGetString(args, "email", out var email))
            {
                return JsonError("Missing required string argument 'email'.");
            }

            if (!_customersByEmail.TryGetValue(email.Trim().ToLowerInvariant(), out var customer))
            {
                return JsonSerializer.Serialize(new { found = false, email });
            }

            return JsonSerializer.Serialize(new
            {
                found = true,
                customer.Email,
                customer.Name,
                customer.Plan,
                customer.Region,
                customer.MonthsActive,
            });
        }

        // ── Tool: create_ticket ───────────────────────────────────────────────

        private string CreateTicket(JsonElement args)
        {
            if (!TryGetString(args, "customerEmail", out var email))
            {
                return JsonError("Missing required string argument 'customerEmail'.");
            }
            if (!TryGetString(args, "subject", out var subject))
            {
                return JsonError("Missing required string argument 'subject'.");
            }

            var priority = TryGetString(args, "priority", out var p) ? NormalisePriority(p) : "normal";

            var ticket = new Ticket(
                Id: $"TICK-{_nextTicketNumber++}",
                CustomerEmail: email.Trim(),
                Subject: subject.Trim(),
                Priority: priority,
                Status: "open");
            _tickets.Add(ticket);

            return JsonSerializer.Serialize(new
            {
                created = true,
                ticket.Id,
                ticket.CustomerEmail,
                ticket.Subject,
                ticket.Priority,
                ticket.Status,
            });
        }

        // Accepts English and Polish priority words (the Bielik preset's model answers in Polish, e.g.
        // "wysoki"); anything unrecognised falls back to "normal".
        private static string NormalisePriority(string value) =>
            value.Trim().ToLowerInvariant() switch
            {
                "low" or "niski" => "low",
                "high" or "wysoki" or "urgent" or "pilny" => "high",
                "normal" or "normalny" or "średni" or "sredni" => "normal",
                _ => "normal",
            };

        // ── Helpers + seed data ───────────────────────────────────────────────

        private static bool TryGetString(JsonElement args, string name, out string value)
        {
            value = string.Empty;
            if (args.ValueKind != JsonValueKind.Object ||
                !args.TryGetProperty(name, out var prop) ||
                prop.ValueKind != JsonValueKind.String)
            {
                return false;
            }

            var s = prop.GetString();
            if (string.IsNullOrWhiteSpace(s)) { return false; }
            value = s;
            return true;
        }

        private static string JsonError(string message) =>
            JsonSerializer.Serialize(new { error = message });

        private static Dictionary<string, Customer> SeedCustomers() => new(StringComparer.Ordinal)
        {
            ["ada@northwind.example"] = new Customer("ada@northwind.example", "Ada Northwind", "Enterprise", "EU", 27),
            ["sam@brightlabs.example"] = new Customer("sam@brightlabs.example", "Sam Bright", "Business", "US", 8),
            ["jo@tinkergarden.example"] = new Customer("jo@tinkergarden.example", "Jo Tinker", "Starter", "EU", 2),
        };
    }

}
