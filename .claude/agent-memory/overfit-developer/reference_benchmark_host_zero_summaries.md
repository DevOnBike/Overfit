---
name: benchmark-host-zero-summaries
description: BenchmarkDotNet returns zero summaries for a filter miss AND for a rejected command line; how to force an all-fail build arm cheaply.
metadata:
  type: reference
---

`BenchmarkSwitcher.Run(args)` returns an EMPTY summary sequence for at least four different situations,
and nothing in the return value distinguishes them:

- `--filter` matched nothing;
- the command line was REJECTED (`--cli <missing path>` prints "The provided CliPath … does NOT exist" and
  returns empty — measured 2026-08-14);
- `--list` / `--info` / `--help` / `--version`, which are legitimate questions, not failures;
- the interactive picker dismissed with no selection.

So any exit-code logic keyed on "zero summaries" must NOT assert *why*. `Sources/Benchmark/Program.cs`
returns 3 (`NothingRanExitCode`) for all of them but words the message as "if the filter matched
nothing… / if the log above reports a rejected option…".

**A build failure is a different shape**: summaries are non-empty, `Summary.BenchmarksCases.Length > 0`,
and every `BenchmarkReport.AllMeasurements.Count == 0`. Counting measurements (not reports) is what
separates "it ran" from "it was attempted".

**To force an all-fail build arm cheaply** — needed to exercise that branch — set `MSBuildSDKsPath` to a
missing directory in the benchmark host's environment. BDN's generated project is built by a child
`dotnet` process that inherits it, so the build fails in ~1.6 s. Things that do NOT work: `--buildTimeout 1`
(the cached build finishes first), `--runtimes net48` (the config's `[SimpleJob(Net10_0)]` stays, so you get
a PARTIAL failure — 4 of 8 measured, exit 0).

A Dry-job run of one benchmark still costs ~75-82 s wall and takes the global measurement mutex.
