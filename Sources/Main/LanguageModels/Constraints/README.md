# `LanguageModels/Constraints` — making generation obey a grammar

Constrained decoding: at each step the sampler is told which tokens are legal, so the output is
guaranteed to parse rather than merely likely to. `ITokenConstraint` (in `../Contracts`) is the seam.

| Type | Guarantees |
|---|---|
| `JsonGrammarConstraint` | Syntactically valid JSON. |
| `JsonSchemaConstraint` | Valid JSON *and* conforming to a supplied schema. |
| `RegexConstraint` | Matches a regular expression, via the DFA in `Regex/`. |

`Schema/` compiles a JSON Schema once (`JsonSchemaCompiler` → `CompiledJsonSchema`) and then tracks
position with `JsonSchemaTracker`; `JsonStringTrie` handles enumerated string values so the legal-token
set for a field with fixed options is a trie walk rather than a scan.

## What this buys, and where it stops

A constraint removes whole classes of failure — unbalanced braces, missing required fields, an invalid
enum — and cannot make a small model *sensible*. Measured here: the `ReActAgent` loop with tool-call
constraints is unit-solid but unreliable end to end on a 3B model, because the failures that remain are
inside JSON *string* values, where the grammar has nothing to say. Structure is enforceable; content
needs a bigger model (7B+).

Constraints work per-token and run on every step, so they sit in the decode hot path: the legal-token
computation must not allocate.
