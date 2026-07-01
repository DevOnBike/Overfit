// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text.Json;
using DevOnBike.Overfit.LanguageModels.Constraints;

namespace DevOnBike.Overfit.LanguageModels.Skills.Evaluation
{
    /// <summary>
    /// The subjective ("did it follow conventions / is it good?") half of the two-tier scheme — a local
    /// judge model scores a run's output against a free-text <c>rubric</c>. The judge's reply is forced to a
    /// fixed JSON shape (<c>{overall_pass, score, reason}</c>) via <see cref="JsonSchemaConstraint"/>, so it
    /// ALWAYS parses (unlike a cloud <c>--output-schema</c>, which can still emit invalid JSON). Register it in
    /// a <see cref="CheckRegistry"/> like any deterministic grader and name it from the case's
    /// <see cref="SkillEvalCase.ExpectedChecks"/>.
    ///
    /// <para>Prefer a judge stronger than the model under test; on a tiny judge, treat the rubric as advisory
    /// and lean on the deterministic graders. The judge runs through <see cref="OverfitClient.Complete"/>
    /// (stateless) so grading one case never bleeds into the next.</para>
    /// </summary>
    public sealed class RubricGrader : ISkillGrader
    {
        // additionalProperties:false + all-required keeps the small-model judge on-rails.
        private const string RubricSchema =
            "{\"type\":\"object\",\"additionalProperties\":false,"
            + "\"properties\":{\"overall_pass\":{\"type\":\"boolean\"},\"score\":{\"type\":\"integer\"},"
            + "\"reason\":{\"type\":\"string\"}},\"required\":[\"overall_pass\",\"score\",\"reason\"]}";

        private readonly OverfitClient _judge;
        private readonly string _rubric;

        public RubricGrader(OverfitClient judge, string rubric, string id = "rubric")
        {
            ArgumentNullException.ThrowIfNull(judge);
            ArgumentException.ThrowIfNullOrEmpty(id);
            _judge = judge;
            _rubric = rubric ?? string.Empty;
            Id = id;
        }

        public string Id { get; }

        public GradeCheck Grade(SkillEvalCase testCase, SkillRunResult result)
        {
            ArgumentNullException.ThrowIfNull(testCase);
            ArgumentNullException.ThrowIfNull(result);

            var prompt =
                "You are grading a model's answer against a rubric. Reply ONLY with JSON: "
                + "{\"overall_pass\": bool, \"score\": 0-100, \"reason\": string}.\n\n"
                + "RUBRIC:\n" + _rubric + "\n\n"
                + "USER REQUEST:\n" + testCase.Prompt + "\n\n"
                + "ANSWER TO GRADE:\n" + result.Output + "\n";

            _judge.Reset();
            var json = _judge.Complete(prompt, constraint: new JsonSchemaConstraint(_judge.Tokenizer, RubricSchema));

            try
            {
                using var doc = JsonDocument.Parse(json);
                var root = doc.RootElement;
                var pass = root.GetProperty("overall_pass").GetBoolean();
                var score = root.TryGetProperty("score", out var s) && s.TryGetInt32(out var sv) ? sv : 0;
                var reason = root.TryGetProperty("reason", out var r) ? r.GetString() ?? string.Empty : string.Empty;
                return new GradeCheck(Id, pass, $"score={score}; {reason}");
            }
            catch (JsonException)
            {
                return new GradeCheck(Id, false, "rubric judge returned unparseable JSON: " + json);
            }
        }
    }
}
