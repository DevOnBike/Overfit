// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

// Microsoft.Extensions.AI.Evaluation — running 100% LOCALLY, with an Overfit in-process model as the
// LLM judge. No Azure, no OpenAI key, no data egress: the official Microsoft evaluation framework
// (Coherence / Fluency / Groundedness quality metrics) scored entirely on your own hardware.
//
//   dotnet run -c Release --project Demo/EvaluationDemo -- [path-to-judge.gguf]
//
// The judge model defaults to %OVERFIT_JUDGE% or C:\qwen3b\qwen.q4km.gguf. Bigger judge = better
// judgments; the evaluator prompts are tuned against GPT-4o-class models, so treat small local
// judges (<7B) as a demo of the PLUMBING, not a calibrated quality gate.

using DevOnBike.Overfit.Demo.Evaluation;
using DevOnBike.Overfit.Extensions.AI;
using DevOnBike.Overfit.LanguageModels;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI.Evaluation;
using Microsoft.Extensions.AI.Evaluation.Quality;

var modelPath = args.Length > 0
    ? args[0]
    : Environment.GetEnvironmentVariable("OVERFIT_JUDGE") ?? @"C:\qwen3b\qwen.q4km.gguf";

if (!File.Exists(modelPath))
{
    Console.Error.WriteLine($"Judge model not found: {modelPath}");
    Console.Error.WriteLine("Usage: EvaluationDemo [path-to-judge.gguf]  (or set OVERFIT_JUDGE)");
    return 1;
}

Console.WriteLine($"Loading local judge: {modelPath}");
using var overfit = OverfitClient.LoadGguf(modelPath);
// The evaluator rubrics make the judge reason step-by-step BEFORE emitting the rating; the
// adapter's default 512-token cap can cut the verdict off (→ "Inconclusive"), so raise it.
IChatClient judge = new MaxTokensChatClient(overfit.AsChatClient(), 2048);
var chatConfig = new ChatConfiguration(judge);

// ── The thing being evaluated: a question, retrieved grounding context, and a candidate answer ──
// (in a real pipeline these come from your RAG/agent under test; here they're fixed so the demo
// is reproducible and runs in seconds)
const string question = "What is the capital of France and roughly how many people live there?";
const string groundingContext =
    "Paris is the capital and largest city of France. The city proper has an estimated " +
    "population of about 2.1 million people, while the metropolitan area is home to over 12 million.";
const string goodAnswer =
    "The capital of France is Paris. The city itself has around 2 million inhabitants, " +
    "and its wider metropolitan area exceeds 12 million.";
const string badAnswer =
    "The capital of France is Marseille, a small alpine village with about 900 residents.";

var userMessage = new ChatMessage(ChatRole.User, question);

foreach (var (label, answer) in new[] { ("GOOD answer", goodAnswer), ("BAD answer", badAnswer) })
{
    Console.WriteLine();
    Console.WriteLine($"=== {label}: \"{answer}\" ===");

    var response = new ChatResponse(new ChatMessage(ChatRole.Assistant, answer));

    // Coherence + Fluency judge the response text alone; Groundedness checks it against the
    // retrieved context (the RAG hallucination gate).
    IEvaluator[] evaluators =
    [
        new CoherenceEvaluator(),
        new FluencyEvaluator(),
        new GroundednessEvaluator(),
    ];

    var groundedness = new GroundednessEvaluatorContext(groundingContext);

    foreach (var evaluator in evaluators)
    {
        var result = await evaluator.EvaluateAsync(
            [userMessage], response, chatConfig, [groundedness]);

        foreach (var metric in result.Metrics.Values)
        {
            var value = metric is NumericMetric numeric ? numeric.Value?.ToString("0.#") ?? "n/a" : "n/a";
            var interpretation = metric.Interpretation is null
                ? ""
                : $"  [{metric.Interpretation.Rating}{(metric.Interpretation.Failed ? " / FAILED" : "")}]";
            Console.WriteLine($"  {metric.Name,-14} {value}/5{interpretation}");

            var reason = metric.Reason;
            if (!string.IsNullOrWhiteSpace(reason))
            {
                Console.WriteLine($"    reason: {Truncate(reason, 220)}");
            }
        }
    }
}

Console.WriteLine();
Console.WriteLine("All scoring above ran in-process on the local CPU — no cloud, no key, no egress.");
return 0;

static string Truncate(string s, int max)
    => s.Length <= max ? s : s[..max] + "…";

namespace DevOnBike.Overfit.Demo.Evaluation
{
    /// <summary>Ensures every judge call gets a generous output budget unless the caller set one.</summary>
    internal sealed class MaxTokensChatClient(IChatClient inner, int maxOutputTokens) : DelegatingChatClient(inner)
    {
        public override Task<ChatResponse> GetResponseAsync(
            IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            options = options?.Clone() ?? new ChatOptions();
            options.MaxOutputTokens ??= maxOutputTokens;
            return base.GetResponseAsync(messages, options, cancellationToken);
        }
    }
}
