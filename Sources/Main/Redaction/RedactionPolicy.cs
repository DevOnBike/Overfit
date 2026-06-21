// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// Maps each redaction category to an <see cref="RedactionAction"/> (what the gateway should do when it is
    /// detected). Unknown categories fall back to <see cref="DefaultAction"/> — <b>fail-closed</b> by default
    /// (<see cref="RedactionAction.Redact"/>), so a new detector is never silently allowed through. The production
    /// map is a per-deployment policy; <see cref="Default"/> ships a conservative starting point.
    /// </summary>
    public sealed class RedactionPolicy
    {
        private readonly IReadOnlyDictionary<string, RedactionAction> _actions;

        public RedactionPolicy(
            IReadOnlyDictionary<string, RedactionAction> actions,
            RedactionAction defaultAction = RedactionAction.Redact)
        {
            ArgumentNullException.ThrowIfNull(actions);

            _actions = actions;
            DefaultAction = defaultAction;
        }

        /// <summary>Action for a category not present in the map — fail-closed (Redact) by default.</summary>
        public RedactionAction DefaultAction
        {
            get;
        }

        /// <summary>The action for <paramref name="category"/>, or <see cref="DefaultAction"/> if unmapped.</summary>
        public RedactionAction ActionFor(string category)
        {
            ArgumentNullException.ThrowIfNull(category);
            return _actions.TryGetValue(category, out var action) ? action : DefaultAction;
        }

        /// <summary>
        /// A conservative default policy over <see cref="DefaultRedactionRules"/>'s categories: hard secrets
        /// (private keys, vendor/API keys, JWTs) are <see cref="RedactionAction.Block"/> — they must never leave the
        /// box even redacted; PII (email, card, SSN) and internal IPs are <see cref="RedactionAction.Redact"/>.
        /// Unknown categories fail closed to Redact.
        /// </summary>
        public static RedactionPolicy Default()
        {
            var map = new Dictionary<string, RedactionAction>(StringComparer.Ordinal)
            {
                ["PRIVATE_KEY"] = RedactionAction.Block,
                ["AWS_KEY"] = RedactionAction.Block,
                ["API_KEY"] = RedactionAction.Block,
                ["JWT"] = RedactionAction.Block,
                ["EMAIL"] = RedactionAction.Redact,
                ["CREDIT_CARD"] = RedactionAction.Redact,
                ["SSN"] = RedactionAction.Redact,
                ["IPV4"] = RedactionAction.Redact,
                // Polish national identifiers (checksum-validated) — PII, redacted.
                ["PESEL"] = RedactionAction.Redact,
                ["NIP"] = RedactionAction.Redact,
                ["REGON"] = RedactionAction.Redact,
                ["IBAN_PL"] = RedactionAction.Redact
            };

            return new RedactionPolicy(map, RedactionAction.Redact);
        }
    }
}
