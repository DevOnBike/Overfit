# Microsoft.Extensions.AI.Evaluation — fully local, with Overfit as the judge

[`Microsoft.Extensions.AI.Evaluation`](https://learn.microsoft.com/en-us/dotnet/ai/evaluation/libraries) is
Microsoft's framework for scoring AI responses (LLM-as-judge: Coherence, Fluency, Groundedness, Relevance, …,
plus reporting + response caching for CI). Its evaluators talk to the judge through the standard `IChatClient`
abstraction — and Overfit's [`DevOnBike.Overfit.Extensions.AI`](../Sources/Extensions.AI) adapter **is** an
`IChatClient`. That means the whole evaluation loop can run **in-process, on your own hardware: no Azure, no
OpenAI key, no data egress** — your prompts, contexts and candidate answers never leave the machine.

```csharp
using var overfit = OverfitClient.LoadGguf(@"C:\models\judge.gguf");   // the local judge
var chatConfig   = new ChatConfiguration(overfit.AsChatClient());

var result = await new GroundednessEvaluator().EvaluateAsync(
    [new ChatMessage(ChatRole.User, question)],
    new ChatResponse(new ChatMessage(ChatRole.Assistant, candidateAnswer)),
    chatConfig,
    [new GroundednessEvaluatorContext(retrievedContext)]);
// result.Metrics → score 1-5 + Interpretation (incl. Failed flag) + the judge's reasoning
```

## Run the demo

```bash
dotnet run -c Release --project Demo/EvaluationDemo -- C:\path\to\judge.gguf
# default judge: %OVERFIT_JUDGE% or C:\qwen3b\qwen.q4km.gguf
```

It evaluates a good and a bad RAG answer with `CoherenceEvaluator`, `FluencyEvaluator` and
`GroundednessEvaluator`, printing each metric's score, pass/fail interpretation and the judge's reasoning.

## Honest calibration note (measured)

The evaluator rubrics are tuned against GPT-4o-class judges. With a **small local judge** (we tested
Qwen2.5-3B and Phi-3.5-mini) the observed behaviour is:

- **Clearly bad answers are reliably caught** — scored 1/5 with `Failed` interpretations (the regression-gate
  use case works).
- **Good answers may come back "Inconclusive"** — the small judge reasons correctly but doesn't always emit
  the rating in the exact format the parser expects.

For calibrated absolute scores use a **≥7B instruct judge** (e.g. Qwen2.5-7B, Bielik-11B) — the plumbing is
identical, just point the demo at a bigger GGUF. The `Microsoft.Extensions.AI.Evaluation.Safety` package is
**not** coverable locally — it hard-requires the Azure AI Foundry service.

## How this relates to Overfit's own evaluation story

Overfit ships a **deterministic** RAG evaluation harness (`RagAssert`: expected-source recall@K, MRR,
paraphrase stability, false-premise traps, corpus lint — see [`rag-testing.md`](rag-testing.md)). The two are
complementary: M.E.AI.Evaluation judges **response text** with an LLM (subjective, model-dependent), Overfit's
harness gates **retrieval quality** with reproducible IR metrics (bit-stable in CI — what regulated teams
audit). Use both: deterministic retrieval gates + local LLM-judged response quality, all without leaving the box.
