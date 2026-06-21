// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using System.Text.RegularExpressions;

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// Removes sensitive spans from text by applying a set of <see cref="RedactionRule"/>s, replacing each match
    /// with a stable placeholder and reporting what was removed. The core of the outbound redaction gateway:
    /// scan a payload before it leaves the box, forward the cleaned text, audit the findings, optionally
    /// <see cref="Restore"/> the originals on the response coming back.
    ///
    /// Overlapping matches are resolved deterministically — earliest start wins, and the longer span wins on a tie —
    /// so the result is stable regardless of rule order.
    /// </summary>
    public sealed class Redactor
    {
        private readonly RedactionRule[] _rules;

        public Redactor(RedactionRule[] rules)
        {
            ArgumentNullException.ThrowIfNull(rules);
            _rules = rules;
        }

        /// <summary>A redactor preloaded with the conservative <see cref="DefaultRedactionRules"/> starter set.</summary>
        public static Redactor CreateDefault() => new(DefaultRedactionRules.All());

        /// <summary>
        /// Redacts <paramref name="input"/>: returns the cleaned text plus the ordered list of removed spans.
        /// </summary>
        public RedactionResult Redact(string input)
        {
            ArgumentNullException.ThrowIfNull(input);

            if (input.Length == 0)
            {
                return new RedactionResult(input, []);
            }

            var spans = CollectSpans(input);

            if (spans.Count == 0)
            {
                return new RedactionResult(input, []);
            }

            var builder = new StringBuilder(input.Length);
            var matches = new List<RedactionMatch>();
            var counters = new Dictionary<string, int>(StringComparer.Ordinal);
            var cursor = 0;

            foreach (var span in spans)
            {
                // Skip spans that overlap one we already took.
                if (span.Start < cursor)
                {
                    continue;
                }

                builder.Append(input, cursor, span.Start - cursor);

                var index = counters.GetValueOrDefault(span.Category);
                counters[span.Category] = index + 1;
                var placeholder = $"[REDACTED_{span.Category}_{index}]";

                builder.Append(placeholder);
                matches.Add(new RedactionMatch(span.Category, span.Value, placeholder, span.Start, span.Length));
                cursor = span.Start + span.Length;
            }

            builder.Append(input, cursor, input.Length - cursor);
            return new RedactionResult(builder.ToString(), matches);
        }

        /// <summary>
        /// Applies a <see cref="RedactionPolicy"/>: detects all spans, then — per the policy's action for each
        /// category — replaces <see cref="RedactionAction.Redact"/> spans with placeholders, leaves Allow/Alert
        /// spans untouched, and flags the request as <see cref="RedactionDecision.Blocked"/> if any span's category
        /// is <see cref="RedactionAction.Block"/> (the gateway then refuses it).
        /// </summary>
        public RedactionDecision Redact(string input, RedactionPolicy policy)
        {
            ArgumentNullException.ThrowIfNull(input);
            ArgumentNullException.ThrowIfNull(policy);

            if (input.Length == 0)
            {
                return new RedactionDecision(input, [], false, []);
            }

            var spans = CollectSpans(input);

            if (spans.Count == 0)
            {
                return new RedactionDecision(input, [], false, []);
            }

            var builder = new StringBuilder(input.Length);
            var redacted = new List<RedactionMatch>();
            var blockedCategories = new List<string>();
            var counters = new Dictionary<string, int>(StringComparer.Ordinal);
            var cursor = 0;

            foreach (var span in spans)
            {
                if (span.Start < cursor)
                {
                    continue;
                }

                var action = policy.ActionFor(span.Category);

                if (action == RedactionAction.Block && !blockedCategories.Contains(span.Category))
                {
                    blockedCategories.Add(span.Category);
                }

                // Only Redact substitutes; Allow / Alert / Block leave the original in place (a Block refuses the
                // whole request upstream, so its text is never forwarded anyway).
                if (action == RedactionAction.Redact)
                {
                    builder.Append(input, cursor, span.Start - cursor);

                    var index = counters.GetValueOrDefault(span.Category);
                    counters[span.Category] = index + 1;
                    var placeholder = $"[REDACTED_{span.Category}_{index}]";

                    builder.Append(placeholder);
                    redacted.Add(new RedactionMatch(span.Category, span.Value, placeholder, span.Start, span.Length));
                    cursor = span.Start + span.Length;
                }
            }

            builder.Append(input, cursor, input.Length - cursor);
            return new RedactionDecision(builder.ToString(), redacted, blockedCategories.Count > 0, blockedCategories);
        }

        /// <summary>
        /// Scans a model <em>response</em> for sensitive content the model itself produced (a leaked secret, an
        /// echoed document from RAG, a hallucinated identifier) and masks every span whose policy action is
        /// <see cref="RedactionAction.Redact"/> or <see cref="RedactionAction.Block"/> — what is not allowed to leave
        /// the box is not allowed back in either. Allow/Alert spans are left in place.
        ///
        /// <para>Run this <strong>before</strong> <see cref="Restore"/>: at that point the caller's own values are
        /// still placeholders, so only genuinely model-generated content is detected. The mask token
        /// (<c>[REDACTED-RESPONSE-…]</c>) is deliberately distinct from the request placeholder format so a later
        /// <see cref="Restore"/> never rewrites it.</para>
        /// </summary>
        public RedactionResult ScanResponse(string input, RedactionPolicy policy)
        {
            ArgumentNullException.ThrowIfNull(input);
            ArgumentNullException.ThrowIfNull(policy);

            if (input.Length == 0)
            {
                return new RedactionResult(input, []);
            }

            var spans = CollectSpans(input);

            if (spans.Count == 0)
            {
                return new RedactionResult(input, []);
            }

            var builder = new StringBuilder(input.Length);
            var masked = new List<RedactionMatch>();
            var counters = new Dictionary<string, int>(StringComparer.Ordinal);
            var cursor = 0;

            foreach (var span in spans)
            {
                if (span.Start < cursor)
                {
                    continue;
                }

                var action = policy.ActionFor(span.Category);

                // Mask anything the policy would Redact or Block; leave Allow/Alert content visible to the client.
                if (action == RedactionAction.Redact || action == RedactionAction.Block)
                {
                    builder.Append(input, cursor, span.Start - cursor);

                    var index = counters.GetValueOrDefault(span.Category);
                    counters[span.Category] = index + 1;
                    var placeholder = $"[REDACTED-RESPONSE-{span.Category}-{index}]";

                    builder.Append(placeholder);
                    masked.Add(new RedactionMatch(span.Category, span.Value, placeholder, span.Start, span.Length));
                    cursor = span.Start + span.Length;
                }
            }

            builder.Append(input, cursor, input.Length - cursor);
            return new RedactionResult(builder.ToString(), masked);
        }

        // Detects every rule match, sorted deterministically (earliest start first, longer span first on a tie).
        // Overlap resolution happens at substitution time via the cursor.
        private List<Span> CollectSpans(string input)
        {
            var spans = new List<Span>();

            foreach (var rule in _rules)
            {
                foreach (Match match in rule.Pattern.Matches(input))
                {
                    if (match.Length == 0)
                    {
                        continue;
                    }

                    // Precision gate: a checksum-validated rule (PESEL, NIP, Luhn…) only counts if the value passes.
                    if (rule.Validator is not null && !rule.Validator(match.Value))
                    {
                        continue;
                    }

                    spans.Add(new Span(match.Index, match.Length, rule.Category, match.Value));
                }
            }

            spans.Sort(static (a, b) =>
                a.Start != b.Start ? a.Start.CompareTo(b.Start) : b.Length.CompareTo(a.Length));

            return spans;
        }

        /// <summary>
        /// Reverses redaction: substitutes each placeholder in <paramref name="redacted"/> back to its original
        /// value. Used to re-hydrate a model response so the caller sees a coherent answer about the entities that
        /// were hidden from the upstream. Whether a deployment restores or keeps the redaction is a policy choice.
        /// </summary>
        public static string Restore(string redacted, IReadOnlyList<RedactionMatch> matches)
        {
            ArgumentNullException.ThrowIfNull(redacted);
            ArgumentNullException.ThrowIfNull(matches);

            if (matches.Count == 0)
            {
                return redacted;
            }

            var result = redacted;

            foreach (var match in matches)
            {
                result = result.Replace(match.Placeholder, match.Original, StringComparison.Ordinal);
            }

            return result;
        }

        private readonly struct Span
        {
            public Span(int start, int length, string category, string value)
            {
                Start = start;
                Length = length;
                Category = category;
                Value = value;
            }

            public int Start
            {
                get;
            }

            public int Length
            {
                get;
            }

            public string Category
            {
                get;
            }

            public string Value
            {
                get;
            }
        }
    }
}
