// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

// Turnkey showcase: evaluate an agent skill against a LOCAL model, offline and reproducibly.
// Runs a "be concise" skill ON vs OFF over a small dataset, grades each with a deterministic length check
// plus a schema-locked rubric judge, and prints pass-rate ON/OFF + the lift (what the skill actually adds).
//
//   dotnet run --project Demo/SkillEvalConsole -- <path-to-gguf>
//   (or set OVERFIT_MODEL for the path)

using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Skills.Evaluation;
using DevOnBike.Overfit.LanguageModels.Skills.Optimization;

var modelPath = args.Length > 0 ? args[0] : Environment.GetEnvironmentVariable("OVERFIT_MODEL");
if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
{
    Console.Error.WriteLine("Usage: SkillEvalConsole <path-to-gguf>   (or set OVERFIT_MODEL)");
    return 1;
}

Console.WriteLine($"Loading {Path.GetFileName(modelPath)} (greedy → reproducible) ...");
using var client = OverfitClient.LoadGguf(modelPath, maxContextLength: 2048, maxNewTokens: 128);

// ── SkillOpt mode: self-improve a weak instruction, keeping only edits that raise the held-out score ──
if (args.Contains("optimize", StringComparer.OrdinalIgnoreCase))
{
    var optRegistry = new CheckRegistry().Register(new ShortAnswerGrader());
    var train = new List<SkillEvalCase>
    {
        new("t1", "What is the capital of France?", ShouldTrigger: true, new[] { "short_answer" }),
        new("t2", "What is the capital of Japan?", ShouldTrigger: true, new[] { "short_answer" }),
    };
    var val = new List<SkillEvalCase>
    {
        new("v1", "What is the capital of Italy?", ShouldTrigger: true, new[] { "short_answer" }),
        new("v2", "What is the capital of Spain?", ShouldTrigger: true, new[] { "short_answer" }),
    };

    Console.WriteLine("SkillOpt: improving a weak instruction from failures (held-out gate) ...\n");
    var opt = SkillOptimizer.Optimize(
        initialInstructions: "Answer the question.",
        runnerFactory: i => new OverfitSkillRunner(client, "concise_answer", "Answer with just the answer.", i, measureTrigger: false),
        editor: new OverfitSkillEditor(client),
        registry: optRegistry,
        trainCases: train,
        validationCases: val,
        rounds: 3);

    foreach (var s in opt.Steps)
    {
        Console.WriteLine($"  round {s.Round}: val={s.ValidationScore:P0}  {(s.Accepted ? "ACCEPT" : "reject")}  \"{Trim(s.Instructions)}\"");
    }
    Console.WriteLine($"\n  best val score  : {opt.BestValidationScore:P0}");
    Console.WriteLine($"  best instructions: \"{Trim(opt.BestInstructions)}\"");
    return 0;
}

// ── the skill under test ──
var runner = new OverfitSkillRunner(
    client,
    skillName: "concise_answer",
    skillDescription: "Answer a short factual question with just the answer and nothing else.",
    skillInstructions: "Reply with ONLY the direct answer in as few words as possible. No sentence, no explanation.");

// ── graders: a deterministic length check gates the report; the schema-locked rubric judge is advisory ──
// (on a small local judge the rubric is noisy — treat it as a second signal, not the gate; see docs/skill-eval.md).
var rubric = new RubricGrader(client,
    rubric: "Pass only if the answer is correct AND at most a few words (no full sentence).",
    id: "concise_and_correct");

var registry = new CheckRegistry().Register(new ShortAnswerGrader());

// ── the dataset (real usage patterns; add negative controls with ShouldTrigger:false in a real set) ──
var cases = new List<SkillEvalCase>
{
    new("fr", "What is the capital of France?", ShouldTrigger: true, new[] { "short_answer" }),
    new("jp", "What is the capital of Japan?", ShouldTrigger: true, new[] { "short_answer" }),
    new("de", "What is the capital of Germany?", ShouldTrigger: true, new[] { "short_answer" }),
};

Console.WriteLine($"Evaluating {cases.Count} cases (skill ON vs OFF) ...\n");
var report = SkillEvaluator.Evaluate(cases, runner, registry);

foreach (var c in report.Cases)
{
    Console.WriteLine($"[{c.Case.Id}] trigger={(c.TriggerCorrect ? "ok" : "MISS")}");
    Console.WriteLine($"   ON  (pass={c.OnPass}) : {Trim(c.OnResult.Output)}");
    Console.WriteLine($"   OFF (pass={c.OffPass}): {Trim(c.OffResult.Output)}");
}

Console.WriteLine();
Console.WriteLine("  ── deterministic gate (short_answer) ──");
Console.WriteLine($"  pass ON  : {report.PassRateOn:P0}");
Console.WriteLine($"  pass OFF : {report.PassRateOff:P0}   (unaided baseline)");
Console.WriteLine($"  LIFT     : {report.Lift:P0}   (what the skill actually adds)");
Console.WriteLine($"  trigger  : {report.TriggerAccuracy:P0}");

// Second tier, advisory: the schema-locked rubric judge on each skill-ON answer.
Console.WriteLine();
Console.WriteLine("  ── rubric judge (advisory, schema-locked) ──");
foreach (var c in report.Cases)
{
    var verdict = rubric.Grade(c.Case, c.OnResult);
    Console.WriteLine($"  [{c.Case.Id}] pass={verdict.Pass}  ({verdict.Notes})");
}
return 0;

static string Trim(string s)
{
    var t = s.Replace('\n', ' ').Trim();
    return t.Length <= 80 ? t : t[..80] + "…";
}

// Deterministic check: the answer must be short (the skill's whole point).
file sealed class ShortAnswerGrader : ISkillGrader
{
    public string Id => "short_answer";
    public GradeCheck Grade(SkillEvalCase testCase, SkillRunResult result)
    {
        var words = result.Output.Split(
            new[] { ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries).Length;
        return new GradeCheck(Id, words > 0 && words <= 3, $"{words} words");
    }
}
