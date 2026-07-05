// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Constraints;

namespace DevOnBike.Overfit.LanguageModels.Skills.Optimization
{
    /// <summary>
    /// Real <see cref="ISkillEditor"/> — a local "optimizer" model proposes one revised instruction text from
    /// the current skill's failures, its reply forced to <c>{reasoning, revised_instructions}</c> by
    /// <see cref="JsonSchemaConstraint"/> so the edit is always machine-usable (a cloud editor can drift off
    /// format). Runs through <see cref="OverfitClient.Complete"/> (stateless), and is told the already-rejected
    /// rewrites so it doesn't loop. A capable optimizer model (7B+) gives better edits; the held-out selection
    /// gate in <see cref="SkillOptimizer"/> makes even a weak editor safe (bad edits are simply rejected).
    /// </summary>
    public sealed class OverfitSkillEditor : ISkillEditor
    {
        private const string EditSchema =
            "{\"type\":\"object\",\"additionalProperties\":false,"
            + "\"properties\":{\"reasoning\":{\"type\":\"string\"},\"revised_instructions\":{\"type\":\"string\"}},"
            + "\"required\":[\"reasoning\",\"revised_instructions\"]}";

        private readonly OverfitClient _optimizer;
        private readonly int _maxFailuresShown;

        public OverfitSkillEditor(OverfitClient optimizer, int maxFailuresShown = 5)
        {
            ArgumentNullException.ThrowIfNull(optimizer);
            _optimizer = optimizer;
            _maxFailuresShown = maxFailuresShown;
        }

        public string? Propose(
            string currentInstructions,
            IReadOnlyList<CaseFailure> failures,
            IReadOnlyList<string> rejectedRevisions)
        {
            ArgumentNullException.ThrowIfNull(currentInstructions);
            ArgumentNullException.ThrowIfNull(failures);
            ArgumentNullException.ThrowIfNull(rejectedRevisions);

            var sb = new StringBuilder();
            sb.Append("You are improving a skill's instructions so a model follows them more reliably. Make ONE ")
              .Append("small, targeted change that would fix the failures below; keep the instructions concise. ")
              .Append("Reply ONLY with JSON: {\"reasoning\": string, \"revised_instructions\": string}.\n\n");
            sb.Append("CURRENT INSTRUCTIONS:\n").Append(currentInstructions).Append("\n\n");
            sb.Append("FAILURES (prompt -> what the skill produced):\n");

            var shown = Math.Min(failures.Count, _maxFailuresShown);
            for (var i = 0; i < shown; i++)
            {
                sb.Append("- \"").Append(failures[i].Prompt).Append("\" -> \"").Append(failures[i].Output).Append("\"\n");
            }

            if (rejectedRevisions.Count > 0)
            {
                sb.Append("\nDo NOT repeat any of these already-rejected rewrites:\n");
                for (var i = 0; i < rejectedRevisions.Count; i++)
                {
                    sb.Append("- ").Append(rejectedRevisions[i]).Append('\n');
                }
            }

            _optimizer.Reset();
            var json = _optimizer.Complete(
                sb.ToString(), constraint: new JsonSchemaConstraint(_optimizer.Tokenizer, EditSchema));

            try
            {
                using var doc = JsonDocument.Parse(json);
                if (doc.RootElement.TryGetProperty("revised_instructions", out var revised))
                {
                    var text = revised.GetString();
                    return string.IsNullOrWhiteSpace(text) ? null : text;
                }
            }
            catch (JsonException)
            {
                // unparseable → signal "no edit this round"
            }

            return null;
        }
    }
}
