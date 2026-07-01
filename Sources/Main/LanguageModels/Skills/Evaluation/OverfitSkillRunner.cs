// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.LanguageModels.Tools;

namespace DevOnBike.Overfit.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// Real <see cref="ISkillRunner"/> over a local model (<see cref="OverfitClient"/>). Each run generates the
    /// answer with the skill's instructions injected as a system message (skill ON) or without them (OFF), using
    /// the client's greedy sampling so runs are byte-reproducible. Generation goes through
    /// <see cref="OverfitClient.Complete"/> (stateless — it renders only the system turn(s) + this one prompt),
    /// so eval cases stay isolated and don't bleed into each other.
    ///
    /// <para>When trigger measurement is enabled, a separate constrained pass makes the model pick exactly one of
    /// {the skill as a tool, a "none" tool} via <see cref="ToolCallConstraint"/> — a hard, deterministic routing
    /// decision graded against <see cref="SkillEvalCase.ShouldTrigger"/>. Because the skill's description drives
    /// trigger accuracy, this is the highest-signal deterministic check, and it's reproducible locally.</para>
    /// </summary>
    public sealed class OverfitSkillRunner : ISkillRunner
    {
        private const string NoneTool = "none";

        private readonly OverfitClient _client;
        private readonly string _toolName;
        private readonly string _skillDescription;
        private readonly string _skillInstructions;
        private readonly bool _measureTrigger;

        public OverfitSkillRunner(
            OverfitClient client,
            string skillName,
            string skillDescription,
            string skillInstructions,
            bool measureTrigger = true)
        {
            ArgumentNullException.ThrowIfNull(client);
            ArgumentException.ThrowIfNullOrEmpty(skillName);

            _client = client;
            _skillDescription = skillDescription ?? string.Empty;
            _skillInstructions = skillInstructions ?? string.Empty;
            _toolName = SanitizeToolName(skillName);
            _measureTrigger = measureTrigger;
        }

        public SkillRunResult Run(string prompt, bool skillEnabled)
        {
            ArgumentNullException.ThrowIfNull(prompt);

            // Trigger is only meaningful when the skill is on the table (a routing decision, ON runs only).
            var triggered = _measureTrigger && skillEnabled ? MeasureTrigger(prompt) : (bool?)null;

            _client.Reset();
            
            if (skillEnabled && _skillInstructions.Length > 0)
            {
                _client.AddSystem(_skillInstructions);
            }

            var tokens = 0;
            var stopwatch = ValueStopwatch.StartNew();
            var output = _client.Complete(prompt, onText: _ => tokens++);

            return new SkillRunResult(output, triggered, tokens, stopwatch.GetElapsedTime().TotalMilliseconds);
        }

        // Deterministic skill routing: constrain the model to emit {"name": "<skill|none>", ...} and read the pick.
        private bool MeasureTrigger(string prompt)
        {
            var tools = new List<ToolDefinition>
            {
                new(_toolName, _skillDescription),
                new(NoneTool, "Choose this when no specialized skill is needed; answer the user directly."),
            };
            // ToolCallConstraint carries per-generation state, so a fresh instance is used for each routing call.
            var constraint = new ToolCallConstraint(tools, _client.Tokenizer);

            _client.Reset();
            _client.AddSystem("You are a router. Pick exactly one tool that best handles the user's request.");
            var reply = _client.Complete(prompt, constraint: constraint);

            return ToolCall.TryParse(reply, out var call)
                && string.Equals(call.Name, _toolName, StringComparison.Ordinal);
        }

        // Tool names feed an identifier-style enum DFA — map the skill name to [A-Za-z0-9_].
        private static string SanitizeToolName(string name)
        {
            var chars = new char[name.Length];
            for (var i = 0; i < name.Length; i++)
            {
                var c = name[i];
                chars[i] = char.IsLetterOrDigit(c) || c == '_' ? c : '_';
            }
            var sanitized = new string(chars);
            return sanitized.Length > 0 ? sanitized : "skill";
        }
    }
}
