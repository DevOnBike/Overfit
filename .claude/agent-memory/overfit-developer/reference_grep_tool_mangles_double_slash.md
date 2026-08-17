---
name: grep-tool-mangles-double-slash
description: The Grep tool's content output can render `//` as `\`, making sound C# look corrupted; confirm with Read before reacting.
metadata:
  type: reference
---

Observed 2026-08-14. `Grep` with `output_mode: content` printed
`\ 300 steps: with the whole base …` for `Tests/LanguageModels/LoRA/Gpt1QLoRATests.cs:171`, and
`reused {reused}\{prompt.Length}` for a `PromptCacheReuseTests` interpolated string. Both files are
FINE — `Read` on the same lines shows `// 300 steps:` and `reused {reused} of {prompt.Length}`.

The `//` comment marker (and some other sequences) can come back through Grep as a backslash.

**Why it matters:** it looks exactly like a file corrupted by a bad script — which is a failure mode this
repository has actually had — so the reflex is to investigate or "fix" it. **Confirm with `Read` on the
specific line before believing Grep's rendering of any punctuation.** Grep is still right about WHICH
lines matched; it is the rendering that is untrustworthy.
