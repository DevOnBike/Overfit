
# Test your RAG like you test your code

Most RAG failures are **retrieval** failures, not generation failures: if the wrong (or no) document is fetched,
the LLM either hallucinates or says "I don't know" — no prompt tweak fixes that. Yet RAG is usually shipped as a
black box you test by eyeballing answers. Edit a document, change the chunk size, swap the embedder, change
`topK` — and retrieval silently shifts. You find out when a user gets a wrong answer.

Overfit ships a **RAG Stability Harness**: the retrieval stage becomes a set of plain assertions you gate in CI,
exactly like unit tests. Pure .NET, in-process, deterministic, **no LLM call** — so it's fast and reproducible.

Namespace: `DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation`.

## What it guarantees — and what it doesn't

RAG is a two-stage function: `answer = LLM(question, retrieve(question))`. The harness tests `retrieve(...)` —
the deterministic stage where most bugs live — and lints the corpus it searches. It deliberately does **not**
call the LLM.

| Guaranteed / tested | Not covered |
|---|---|
| The right document reaches the model (expected-source recall) | The LLM's exact wording (separate stage; see "Reproducibility" below) |
| Rephrasings retrieve the same documents (paraphrase stability) | Whether a document's content is factually correct |
| Un-grounded questions find no confident source (false-premise traps) | Contradictions between documents (needs an NLI model — not yet) |
| Corpus hygiene: near-duplicates, orphans, too-short chunks | |

## The four checks

```csharp
using DevOnBike.Overfit.LanguageModels.Embeddings;   // SentenceEmbedder
using DevOnBike.Overfit.LanguageModels.Retrieval;    // VectorStore
using DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation;

// You already have these from your RAG pipeline:
//   var embedder = SentenceEmbedder.ForMiniLm(@"C:\minilm");
//   var store    = /* your indexed VectorStore */;

var eval = RagEvaluator.ForEmbedder(store, embedder);
```

**1. Expected-source recall** — "question X must retrieve document Y in the top-K."

```csharp
var report = eval.EvaluateRetrieval(
[
    new RetrievalCase("How do I reset my password?", "faq#password"),
    new RetrievalCase("What plans are available?",   "policy#plans"),
], topK: 5);

// report.RecallAtK, report.MeanReciprocalRank, report.Cases[i].Hit / Rank
```

**2. Paraphrase stability** — "these rephrasings must retrieve the same documents." Measured as the mean
pairwise Jaccard overlap of the retrieved id sets.

```csharp
var stability = eval.EvaluateParaphraseStability(
[
    new ParaphraseGroup("reset-pw",
        "How do I reset my password?",
        "I forgot my password, what now?",
        "How can I change my password?"),
], topK: 5, minJaccard: 0.6);
```

**3. False-premise traps** — a question whose premise isn't grounded should find **no** confident match;
otherwise the LLM is handed plausible-but-irrelevant evidence and grounds a hallucination.

```csharp
var traps = eval.EvaluateFalsePremise(
[
    new FalsePremiseCase("What is our refund policy on Mars?"),
], groundedThreshold: 0.5);   // top cosine >= 0.5 => a sprung trap
```

**4. Corpus linter** — static hygiene over the indexed store.

```csharp
var linter = new CorpusLinter(store);
var dups   = linter.FindNearDuplicates(threshold: 0.97);   // redundant passages crowding the top-K
var shorts = linter.FindShortDocuments(minChars: 40);      // headings / fragments with no answer
var orphans = linter.FindOrphans(queryVectors, topK: 5);   // chunks no query can reach
```

## Gate it in CI

`RagAssert` turns any report into a pass/fail gate. It throws `RagAssertionException` (a plain exception, so
xUnit / NUnit / MSTest all see a failed test) with the offending cases in the message.

```csharp
[Fact]
public void Rag_AnswersTheQuestionsItMust()
{
    var eval = RagEvaluator.ForEmbedder(_store, _embedder);

    RagAssert.RecallAtLeast(eval.EvaluateRetrieval(_cases), 0.95);          // every key Q finds its doc
    RagAssert.Stable(eval.EvaluateParaphraseStability(_groups));            // rephrasings agree
    RagAssert.NoGroundedFalsePremises(eval.EvaluateFalsePremise(_traps));   // no spurious grounding

    var linter = new CorpusLinter(_store);
    RagAssert.NoNearDuplicates(linter.FindNearDuplicates());
    RagAssert.NoShortDocuments(linter.FindShortDocuments());
}
```

Now editing a document, re-chunking, or swapping the embedder **can't silently break retrieval** — CI catches it,
and the failure message tells you exactly which query missed and what it retrieved instead.

The harness itself is testable without a model: a deterministic fake embedder over a tiny one-hot corpus makes
every retrieval predictable. See `Tests/LanguageModels/Retrieval/` for 14 model-free examples.

## Reproducibility: what "the same answer" really means

- **Retrieval is deterministic by construction**: a fixed embedder + corpus means the same question always
  produces the same vector and the same top-K. The harness doesn't *create* that — it **guards that it stays
  correct** across your changes.
- **For the final answer to be reproducible end-to-end**, also decode greedily (`temperature = 0`), which Overfit
  does bit-identically.

So: **harness-gated retrieval + greedy decode = a reproducible, auditable RAG answer** — the same input yields the
same output, with a regression test on the evidence that fed it. That is the claim for regulated / enterprise use.
The harness does **not** claim "the LLM always says the same thing regardless of settings" — with sampling on,
wording varies; that's a generation-stage choice, separate from retrieval correctness.

## Try it live — the demo `/rag/eval` endpoint

The ASP.NET starter (`Demo/LocalAgentAspNetDemo`) exposes the harness over its indexed corpus:

```powershell
$env:OVERFIT_MODEL_DIR     = "$env:USERPROFILE\.overfit\models"   # the host loads a chat model to start; /rag/eval doesn't call it
$env:OVERFIT_EMBEDDING_DIR = "C:\minilm"                          # a MiniLM directory ('overfit pull minilm')
$env:ASPNETCORE_URLS       = "http://localhost:5234"
dotnet run -c Release --project Demo/LocalAgentAspNetDemo
```

```powershell
Invoke-RestMethod http://localhost:5234/documents/index -Method Post     # index Data/*.md

$body = @{
  retrieval    = @(@{ query = "How do I reset my password?"; expectedSource = "support-faq.md" })
  paraphrase   = @(@{ name  = "reset-pw"; variants = @("reset password", "I forgot my password", "change my password") })
  falsePremise = @("What is the capital of Mars?")
  topK         = 3
} | ConvertTo-Json -Depth 6
Invoke-RestMethod http://localhost:5234/rag/eval -Method Post -ContentType application/json -Body $body
```

`expectedSource` is matched as a substring of the chunk id (typically the source file name). The response carries
the recall / MRR / per-case ranks, the paraphrase Jaccard, the false-premise top scores, and the lint findings —
all retrieval-side, no LLM call. (Swagger UI at `/swagger` has it too.)

## Not yet covered

- **Contradiction detection** ("same topic, conflicting facts") needs a natural-language-inference model; the
  near-duplicate lint flags same-topic redundancy, but not disagreement. On the roadmap.
