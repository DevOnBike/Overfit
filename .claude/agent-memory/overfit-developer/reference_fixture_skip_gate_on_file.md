---
name: reference-fixture-skip-gate-on-file
description: Gate a fixture skip on the FILE, not on Directory.Exists — an existing empty fixture dir passes the directory check and the test then goes red instead of skipping.
metadata:
  type: reference
---

Converting a vacuous `if (!Directory.Exists(dir)) { return; }` to `Assert.SkipWhen(!Directory.Exists(dir), ...)`
does **not** make the test CI-safe. Measured 2026-08-15 on
`BatchedPrefillParityTests.IncrementalDecode_PreservesSpaces_LikeChatSession`: with
`OVERFIT_QWEN3B_DIR` pointed at an existing but empty directory the guard passed and
`QwenTokenizer.Load` threw `FileNotFoundException` — the arm went **red**, not skipped. Gating on
`File.Exists(TestModelPaths.Qwen3B.TokenizerJsonPath)` gave `Skipped 1`, exit 0.

The CI-safety arm therefore has to point `OVERFIT_*_DIR` at an **existing empty** directory, not at a
non-existent path: the non-existent path exercises a weaker condition and passes either way.

`Assert.SkipWhen` (xunit.v3 3.2.2) does report as `Skipped` through the VSTest bridge — verified in the
same run — so it is the right tool for a fast-suite test whose fixture is only knowable at run time.
`ModelFact`/`FixtureFact` both derive from `LongFact` and would move such a test out of the fast suite.

Related: [[reference-coverage-runsettings-scope]]
