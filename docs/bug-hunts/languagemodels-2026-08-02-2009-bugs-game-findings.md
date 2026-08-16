# Bug hunt — `Sources/Main/LanguageModels` (excluding `Runtime/`)

**Scope reviewed:** `Sources/Main/LanguageModels/{Loading,Tokenizers,Rope,Quantization,Chat,Agents,Tools,
Constraints,Contracts,Memory}` (partial — see Coverage below). `Runtime/` explicitly excluded per instructions
(already hunted twice, 3 defects tracked in `ROADMAP.md`).

**Timestamp (UTC):** 2026-08-02 20:09
**Commit:** `996a161` (branch `gimli`)
**Score: 8 points / 4 defects**
**How the run ended:** neither by exhausting the scope nor by the 10-minute cap — stopped voluntarily at
~5.5 minutes elapsed after the yield of new high-confidence findings dropped, well short of both boundaries.
Treat the low count as **inconclusive about the unopened parts**, same as a time-capped run — see Coverage.
**README:** `Sources/Main/LanguageModels/README.md` exists and was read first (map of subdirectories, "fails
fluently" warning, scope-of-engine note). No defect found that the README's stated limitations already cover.

---

## Findings, ranked by damage

### 1. `RepackedWeightsFile.Open` trusts an unbounded name-length field from an untrusted sidecar file, and the one caller's "corrupt sidecar must never block loading" guarantee doesn't hold

**What breaks:** `Open` reads a raw `int32 nameLen` from the `.gguf.repack` sidecar and passes it straight
into `reader.ReadBytes(nameLen)` with no check against the remaining file size. `GgufReader.ReadString()`
guards the identical shape explicitly (`RequireDeclaredCountFitsInFile`, with a comment explaining exactly
why: "a declared string length is an allocation request from the file... checked against what the file
actually holds") — that defence was never carried over when this sidecar format was added. A crafted or
truncated `.gguf.repack` with `nameLen` set to a large positive value asks `BinaryReader.ReadBytes` to
allocate that many bytes up front, before any EOF check fires; a negative `nameLen` throws
`ArgumentOutOfRangeException`.

**Where:** `Sources/Main/LanguageModels/Loading/RepackedWeightsFile.cs`, `RepackedWeightsFile.Open`
(the entry-parsing loop, ~lines 155–164).

**How anyone would notice today:** they would not notice the load hazard directly — but the failure mode
contradicts a stated guarantee. `GgufLlamaLoader.TryOpenSidecar` (`GgufLlamaLoader.cs`, ~lines 78–98) opens
this exact sidecar next to every model file, with a comment reading "a corrupt/incompatible sidecar must
never block loading", and only catches `OverfitFormatException` / `IOException`. `ArgumentOutOfRangeException`
(negative `nameLen`) and `OutOfMemoryException` (huge `nameLen`) are not caught, so a malformed sidecar sitting
next to a downloaded model — the same threat model GGUF itself is defended against — crashes
`GgufLlamaLoader.Load` outright instead of falling back to the runtime repack path the design promises. This
is a contract violation as much as a missing bound: the comment states a guarantee the catch clause doesn't
provide.

**What test would have caught it:** a unit test that writes a `.repack` sidecar with `nameLen` set to a
negative value (or `int.MaxValue`) next to a real GGUF, and asserts `GgufLlamaLoader.Load` still succeeds
(falls back to no-sidecar loading) — mirroring the existing bound tests for `GgufReader.ReadString`.

---

### 2. `GgufTokenizer`'s constructor indexes `token_type` / `scores` by vocabulary id without checking those arrays are the same length as `tokens`

**What breaks:** `FromGguf` reads three independent GGUF metadata arrays — `tokenizer.ggml.tokens`,
`tokenizer.ggml.token_type`, `tokenizer.ggml.scores` — each with its own declared length in the file, with no
cross-validation that the lengths agree. The private constructor then does:
```csharp
for (var id = 0; id < tokens.Length; id++)
{
    _tokenToId.TryAdd(tokens[id], id);
    var byteValue = ParseByteToken(tokens[id], tokenTypes[id]);   // indexes tokenTypes[id]
    ...
}
```
If `tokenizer.ggml.token_type` (or `.scores`) declares fewer entries than `tokenizer.ggml.tokens` — trivial
for a truncated or hostile GGUF, since GGUF places no constraint linking the two array lengths — this throws
an unhandled `IndexOutOfRangeException` instead of the clean `OverfitFormatException` every other malformed-file
path in this loader produces. `SpmMerge`'s `_scores[id]` has the same exposure.

**Where:** `Sources/Main/LanguageModels/Tokenizers/GgufTokenizer.cs`, the private constructor (~lines 104–119)
and `FromGguf` (~lines 132–166).

**How anyone would notice today:** an unhandled crash on load — noisy, not silent, but it is the wrong
exception type for an input-validation problem this file otherwise handles carefully (every other GGUF loader
in `Loading/` converts a length mismatch into `OverfitFormatException` before it can reach an index). A caller
catching `OverfitFormatException` around tokenizer construction (a reasonable thing to do, mirroring
`TryOpenSidecar`'s pattern) will not catch this.

**What test would have caught it:** a unit test building a `GgufTokenizer` via `CreateForTest`/`FromGguf` with
`tokenTypes.Length < tokens.Length` and asserting `OverfitFormatException` (not `IndexOutOfRangeException`).

---

### 3. `GgufTensorInfo.ElementCount` computes the dimension product unchecked, with no cross-check against the file's actual size — unlike the equivalent safetensors path

**What breaks:** `GgufTensorInfo.ElementCount` (`n *= (long)Dims[i]` in a plain loop, no `checked`) can
silently overflow/wrap for attacker-chosen `Dims` values — `GgufReader`'s own `RequireDeclaredCountFitsInFile`
bounds *how many* dimensions a tensor can declare, but never bounds the *value* of each dimension read via
`_reader.ReadUInt64()`, nor validates that `Offset + computed-byte-size` actually fits inside the file.
`SafetensorsReader` was built with exactly this failure mode in mind — `RequireTensorsFitInTheDataBlock`'s doc
comment states outright: "This also catches a shape whose product overflowed Int64: a wrapped count will not
match the range" — because safetensors carries an explicit per-tensor `[begin, end)` byte range to check the
element-count product against. GGUF carries no equivalent field, and no substitute check was added, so the
one documented defence against this exact class of bug in the sibling loader does not exist here. Downstream,
`GgufLlamaLoader` cross-checks `info.ElementCount` against an *expected* count derived from config metadata
(`info.ElementCount != (long)vocab * dModel`, etc.) — but since `ElementCount` itself can be forced by crafted
`Dims` to equal any attacker-chosen value (via overflow), a crafted file can pass that check while the tensor's
real on-disk layout disagrees with its declared shape, and the resulting weights are read from the wrong
byte range/count with no exception at all — silently wrong output, not a crash.

**Where:** `Sources/Main/LanguageModels/Loading/GgufTensorInfo.cs`, `ElementCount` getter (lines 52–63);
consumed unchecked by `GgufReader.LoadTensorAsF32`/`LoadTensorQ4_KRaw`/`LoadTensorQ6_KRaw` and by
`GgufLlamaLoader`'s shape-equality checks (`GgufLlamaLoader.cs` ~lines 602, 629).

**How anyone would notice today:** they would not — this is exactly the "fails fluently" failure mode the
module's own README calls out (wrong-looking-but-plausible text, never an exception). It requires a
deliberately crafted file to trigger (not a random truncation, which the existing `RequireDeclaredCountFitsInFile`
already handles), so the practical exposure is narrower than finding 1/2, but the asymmetry with
`SafetensorsReader` — which explicitly defends the identical class of bug with a comment naming it — is a
concrete, fixable gap.

**What test would have caught it:** a unit test constructing a `GgufTensorInfo` with `Dims` values whose
product overflows `long` (e.g. two `ulong` values near `2^32` each) and asserting `ElementCount` throws
(`checked`) rather than returning a wrapped value; separately, a `GgufReader` test with a tensor whose `Dims`
product is inconsistent with the bytes actually available at `Offset` in the file.

---

### 4. `ReActAgent.Run` re-adds the tool-menu system prompt on every call, growing history unboundedly across repeated calls on the same agent — unlike the sibling `SummarizingChatSession`, which rebuilds history from scratch specifically to avoid this

**What breaks:** `ReActAgent.Run(...)` unconditionally calls `_chat.AddSystem(BuildSystemPrompt(_allTools))`
at the top of every invocation, with no check for whether that system message was already added by a prior
call on the same `ChatSession`. `ChatSession` is explicitly the stateful, reusable multi-turn primitive
(`History` is a public, ever-growing `List<ChatMessage>`; `ChatSession.Send` appends and never prunes), and
`ReActAgent` exposes `MaxSteps`/`Tools` as instance state suggesting the same agent instance is meant to be
reused for more than one query in a session. Call `Run` twice on the same agent and the "You can call
tools: ..." system message appears twice in history verbatim; call it N times and it appears N times,
consuming context budget and re-stating (potentially stale, if tools ever varied) instructions on every turn
forever. Contrast with `SummarizingChatSession.ApplyPlan`/`SummariseOldTurns`/`RestoreHistory`, which always
call `_chat.ResetConversation()` before re-adding any system message specifically to avoid duplication — the
sibling class in the same subsystem (`Memory/`) already encodes the fix for this exact class of bug that
`Agents/ReActAgent.cs` doesn't apply.

**Where:** `Sources/Main/LanguageModels/Agents/ReActAgent.cs`, `Run` (line 104:
`_chat.AddSystem(BuildSystemPrompt(_allTools));`).

**How anyone would notice today:** they would not, until someone reads the transcript or notices output
quality degrading turn over turn — this is unbounded accumulation with no counter, no cap and no error; it
silently eats into the model's context window every repeated call and (per the module's own "fails fluently"
observation) degrades generation quality rather than throwing.

**What test would have caught it:** a test that constructs one `ReActAgent`, calls `Run` twice, and asserts
`chat.History.Count(m => m.Role == "system")` stays at 1 (or that the tool-menu text appears only once in the
rendered prompt) — the shape `SummarizingChatSessionTests` presumably already exercises for its own class.

---

## Shared root cause

Findings 1–3 share one root cause: **the length/count-bound discipline that `GgufReader` built explicitly
for its own metadata parsing (`RequireDeclaredCountFitsInFile`, with the reasoning spelled out in comments)
was not carried forward to two places that read equally untrusted bytes** — the `.gguf.repack` sidecar format
(finding 1) and the GGUF tensor-dimension values themselves, as opposed to their count (finding 3). Finding 2
is the same class of bug (a length read from the file used to index another array without cross-validation)
one layer up, in the tokenizer metadata rather than the tensor-info metadata. All three are variations of "a
bound was built once and not re-applied everywhere the same shape of untrusted length appears."

Finding 4 is unrelated — a chat-history-management bug, not a loading bug — but its fix already exists in the
same subsystem (`Memory/SummarizingChatSession`), which is worth noting for whoever picks it up.

---

## Coverage

**Reviewed and found clean** (specific files/behaviours checked, no defect found):
- `GgufReader.cs` — full read (all metadata/tensor-count bounds, nesting-depth guard, streaming block readers).
- `SafetensorsReader.cs` — full read (`RequireTensorsFitInTheDataBlock`, the safetensors-side equivalent of
  finding 3, done correctly).
- `ShardedSafetensorsReader.cs`, `MemoryMappedModelFile.cs` — bounds-checked, no defect.
- `QwenTokenizer.cs`, `WordPieceTokenizer.cs` — decoder arrays sized from vocab, bound-checked on read/decode.
- `HuggingFaceChatTemplate.cs` — reads `tokenizer_config.json` defensively via `Utf8JsonReader`, no defect.
- `Rope/RopeScaling.cs`, `Rope/RopeTable.cs` — numerically reviewed, construction validated (positive/even
  `headDimension`, positive `theta`, matching `freqFactors` length); no obvious sign/off-by-one error found.
- `Quantization/QuantizationOptions.cs` — trivial DTO, nothing to find.
- `Chat/ChatSession.cs`, `Chat/StopSequenceDetector.cs` — stop-sequence partial-match holdback logic read in
  full, looks correct (longest-held-suffix, earliest-stop-wins); speculative/non-speculative decode paths.
- `Agents/CircuitBreaker.cs` — bounded, correct.
- `Constraints/JsonStateMachine.cs` — full RFC-8259 DFA read, `MaxDepth = 64` bit-stack bound is sound.
- `Tools/ToolCallConstraint.cs` — 64-tool cap enforced, envelope/schema DFA read in full, no defect found.
- `Memory/ChatHistoryCompactor.cs`, `Memory/SummarizingChatSession.cs` — read in full; this is what surfaced
  finding 4 by contrast.
- `Contracts/SamplingOptions.cs`, `Contracts/GenerationStats.cs` — read, no impossible-value defaults found
  beyond documented "0 = disabled" conventions.

**Not reached at all:**
- `Loading/`: `GgmlDequant.cs`, `GgmlQuant.cs`, `SafetensorsGpt2Loader.cs`, `SafetensorsLlamaLoader.cs`,
  `SequentialChunkReadStream.cs`, `LlamaConfigReader.cs`, `SafetensorsSource.cs`. Most of `GgufLlamaLoader.cs`
  itself (1400+ lines; only the config-parsing header and the embedding/expert-loading helpers were sampled).
- `Tokenizers/`: `GgufEmbeddedTokenizer.cs`, `ByteLevelAlphabet.cs`, `QwenChatTokenizer.cs`,
  `HuggingFaceBpeTokenizer.cs` (only the first ~100 lines opened — the array-length-mismatch pattern from
  finding 2 was not checked against this file's own vocab-building code).
- `Quantization/`: `IQuantizedModel.cs`, `IQuantizer.cs`, `QuantizationKind.cs`, `QuantizationScaleKind.cs`.
- `Chat/`: `ChatMessage.cs`, `ChatTemplate.cs`, `ChatTemplateFormat.cs`, `HuggingFaceChatModel.cs`,
  `QwenChatModel.cs`.
- `Agents/`: `CriticLoop.cs`, `ReActCompletion.cs`, `CriticIteration.cs`, `CriticVerdict.cs`,
  `ExtraMaskedTokensConstraint.cs`, `ReActStep.cs`, `ReActResult.cs`, `CircuitBreakerResult.cs`.
- `Tools/`: `ToolCall.cs`, `ToolDefinition.cs`, `ToolParameter.cs`, `ToolParameterKind.cs`.
- `Constraints/`: `JsonGrammarConstraint.cs`, `JsonSchemaConstraint.cs`, `RegexConstraint.cs` — not opened at
  all; these were flagged by the task brief as a likely-fruitful area ("a parse of model-produced JSON that
  assumes well-formedness") and were not checked.
- `Contracts/`: `GenerationOptions.cs`, `StreamingOptions.cs`, `KeyValueCacheShape.cs`, `IKeyValueCache.cs`,
  `ISlmInferenceEngine.cs`, `ISlmModel.cs`, `ISlmSession.cs`, `ITokenConstraint.cs`, `ITokenizer.cs`,
  `SamplingStrategy.cs`, `TokenGeneratedHandler.cs`, `EmbeddingPooling.cs`.
- Entire subtrees not opened: `Embeddings/`, `Retrieval/`, `LoRA/`, `Whisper/`, `Skills/`.

---

## What the short-of-21 score means

This run stopped at 4 defects (8 points), well short of 21, **before** exhausting the named scope and
**before** the 10-minute cap — a genuinely mixed ending, not either of the two clean cases the protocol
describes. Read it the same way you would read a time-capped run: **the score says nothing about the large
majority of files listed as "not reached" above**, most notably `Constraints/JsonGrammarConstraint.cs` and
`JsonSchemaConstraint.cs` (the two files the task brief specifically called out as likely to reward attention
and neither was opened), all of `Loading/GgufLlamaLoader.cs` past its header, and the four subtrees never
opened at all. The four findings above are each independently confirmed by reading the relevant code paths
end to end, not inferred — but they should not be read as "this is most of what's there."
