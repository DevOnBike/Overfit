# Plan: Machine exclusion, made symmetric and complete

STATUS: ANALYSIS_READY

**Revision 3 (2026-08-14).** Five more inputs folded in since revision 2, all from the client via the
coordinator: (a) a clarification that inter-process mutex exclusion and intra-process thread-affinity are
different axes, sharpened by a newly-raised MSBuild **node-reuse** hazard; (b) a design ask for one reusable
`using`-scoped wrapper type, replacing hand-rolled acquire/release logic at every call site; (c) a placement
proposal for that wrapper (`internal`, linked source file); (d) the motivation behind (c) — avoid referencing
the whole engine from a project that has no other reason to; (e) a sharper constraint — `Sources/Analyzers`
should depend on no repo code at all, which on inspection **narrows the placement question from "five
projects" to "2-3 consumers that actually acquire the lock,"** since covering the other projects with the
guard needs no C# in them at all. **One of the two precedents offered for (b) does not exist in the current
tree** — corrected below rather than cited on trust, per this repository's own "verify before you answer"
rule. Revision 2's expanded-scope content (Registration/Subtraction, all five projects, no silent lease
fallback) is retained unchanged; this revision adds to Task 0, restructures the M1/M2 mechanism around the
wrapper, and corrects the placement section's scope down to its real size.

**Revision 4 (2026-08-14) — two corrections to revision 3, both verified independently by this analyst rather
than accepted on relay.** (1) The `PooledArray` finding stands, but its root cause was mischaracterised as
generic doc drift: `ROADMAP-COMPLETED.md:2007` correctly records the type's 2026-05-29 deletion into
`PooledBuffer<T>` — the record is right; three separate **index** entries summarising it
(`Runtime/README.md:12`, `CLAUDE.md`, `docs/code-patterns.md:27`) never picked up the change. (2) "Neither
lab-presence file has a producer anywhere in the tree" was too strong and is corrected: four producers exist —
`run_clean_measurement.py:21` writes `fp-run-clean-start.txt`; `run_overnight_measurement.py:34-39` opens and
holds `fp-run.lock` (it only *reads* the marker, at line 47, to recover its own start time); plus
`run_lab_events.py`/`run_lab_p5.py` (present, not individually read). All four live under `.claude/`, which
`.gitignore:276` excludes wholesale — a gitignore-respecting search (this analyst's first pass) finds only the
tracked consumers. The corrected finding is sharper, not weaker: the mechanism is half-built across a
durability boundary, the same shape `CLAUDE.md` already records for `Scripts/lab.py`'s own history, and it
explains rather than contradicts the A1 incident (whoever ran that measurement had a working producer
locally; the tracked repository never did). Both corrections are threaded through the affected sections below
rather than left as a note here only.

**Revision 5 (2026-08-14) — the client's answers to round 4's two questions, both expansive again, and a
third reading of the two `.claude/` scripts this time for their actual design rather than only their
existence.** (1) **Q1: the lab's presence signal is now a HARD refusal** (Decision 20), overriding
`ROADMAP.md:529-535`'s stated reasoning outright — not treated as acceptable-by-fiat here; the client
requires this plan to design for what that reasoning was protecting against, specifically: a separate,
more visible escape hatch for a day-long block (Requirement 30); a refusal message carrying holder, start
time, expected end and override instructions (Requirement 32); durability as part of the *definition of
done*, not a nice-to-have, with an explicit sequencing consequence (Constraint 33: the hard refusal must not
activate before the producer is durable, or it fails open while looking enforced); and a staleness answer
for a marker left behind by a dead 24-hour run. Reading the two `.claude/` scripts closely for this last
point turned up a real, useful asymmetry (Fact 27): they already embody two different staleness designs, one
crash-safe and one not, which turns "what should the staleness answer be" from an open design question into
a recommendation with working code behind it (Recommendation 28). (2) **Q2: Subtraction stays in this plan**,
together with Registration, sequencing unchanged (Decision 21) — a dependent, later task with its own spike
if the log's real shape does not support it, never designed against an imagined format. Both threaded through
the affected sections below.

GATES:
  verifier:           NOT_REQUIRED — no code written yet (analysis phase)
  reviewer:           NOT_REQUIRED — no code written yet (analysis phase)
  mutation-proof:     NOT_REQUIRED — no code written yet (analysis phase)
  performance:        NOT_REQUIRED — no perf claim in this plan; revisit if the build-hold spike (Task 0) lands
  security:           NOT_REQUIRED — no code written yet. Reasoned: Task 7 (lab-presence producer) will touch
                       a script that already parses Prometheus responses over a cluster connection — plausible
                       `overfit-ciso` trigger once that lands. Revisit at IMPLEMENT.
  leak-scan:          NOT_REQUIRED — no code written yet. Reasoned: expanded scope touches `Scripts/lab.py`,
                       `k8s/**` and hostnames already named in `aiops-backlog.md` — `overfit-leak-scan`'s own
                       trigger table names this class of path. WILL be required once Task 7 lands.
  AOT:                NOT_REQUIRED — build/CI/lab tooling, never reachable from Tests/AotSmokeTest
  API-compatibility:  NOT_REQUIRED — no public API of Sources/Main touched. **Watch this at DESIGN**: if the
                       wrapper (Task 1, below) ends up living inside `Main` rather than a neutral location,
                       confirm it stays `internal` — an `internal` type does not change the public surface, but
                       the decision of where it lives is not yet made (Open question 22).
  release-readiness:  NOT_REQUIRED — analysis phase

**This file is written by `overfit-analyst` only.** `overfit-architect` appends its sections before `STATUS`
may move to `APPROVED`. `overfit-developer` must not start from this document until that signature is present.

---

## What the client asked for

Original request (Polish, quoted):

> "wydaje mi sie ze te blokady powinny dzialac tak, jak idzie benchmark to nic innego na kompie nie moze byc
> uruchamiane - zadna kompilacja zadna publikacja testy itd"

Follow-up (fail-fast, named error):

> "ta blokada z mutexem to ma przy probie rezerwacji od razu rzucac bledem jakims sensownie nazwanym i cos
> mowiacym"

Round-1 answers, all expansive: (1) Registration/Subtraction in scope, (2) all five unguarded projects must be
covered, (3) a negative Task 0 result escalates rather than authorising the lease fallback silently.

**Round-2 inputs (this revision):**

4. Named-mutex clarification: a `Global\` mutex is **inter-process** exclusion; thread-affinity is a
   **separate axis** (`Mutex.ReleaseMutex()` must run on the acquiring thread within a process). Both are
   true and do not contradict each other.
5. Design ask: one reusable `using`-scoped wrapper type — acquire in the constructor, release in `Dispose` —
   instead of every participant hand-rolling acquire/check/release/`AbandonedMutexException` logic. Quoted:
   *"moze kurde napisz wrapper z usingiem na tego mutexa gdzie w dispose zwalnia mutexa aby go nie pisac
   wszedzie tak samo."*
6. Placement proposal: `internal`, and the **same source file linked** (`<Compile Include=".." Link=".." />`)
   into every project that needs it, rather than a project/assembly reference. Quoted: *"to ta strukturka
   moze byc jako internal i moznaby ja moze podpiac do kazdego potrzebnego miejsca w projekcie jako link."*

---

## A citation correction, made before anything is designed around it

The wrapper request cited `Sources/Main/Runtime/PooledArray.cs` as precedent ("`ref struct` wrapping
`ArrayPool` Rent/Return as a `using`"). **That file does not exist in the current tree.** Verified: `Glob` for
`**/PooledArray.cs` returns nothing; a repo-wide grep for the type name returns only *documentation* —
`Sources/Main/Runtime/README.md:12`, `CLAUDE.md`, `docs/code-patterns.md:27`, `ROADMAP-COMPLETED.md`, and one
analyzer doc — never a type definition. The pooled-buffer wrapper that does exist,
`Sources/Main/Tensors/PooledBuffer.cs`, is a **plain `struct`**, not a `ref struct`, and its own doc comment
explicitly lists "class field" and "closure-captured… written from a worker lambda" as supported lifetimes
(`PooledBuffer.cs:16-23`) — the opposite of what a mutex wrapper needs, and possible only *because*
`ArrayPool<T>.Return` does not care which thread calls it. Citing it as the shape to copy would be actively
wrong here.

**The precedent that actually fits, verified by reading both files, is `Sources/Main/Runtime/GcHandleScope.cs`
and `Sources/Main/Runtime/GcLatencyScope.cs`.** Both are `ref struct`, both wrap an acquire/release pair
(`GCHandle.Alloc`/`Free`; `GCSettings.LatencyMode` set/restore) in a `using`-shaped `Dispose()`, and both carry
the exact justification the mutex wrapper needs, stated in their own doc comments: *"a `ref struct`, so it
can't escape its scope or be boxed."* A `ref struct` cannot be a class field, cannot be captured by a lambda
or local function, and cannot be held across an `await` — the compiler rejects all three, which is precisely
the "impossible by construction, not by convention" property the wrapper request asked for.

**Root cause, verified rather than left as "plausibly renamed": the deletion WAS correctly recorded — the
index pointing at it is what went stale.** `ROADMAP-COMPLETED.md:2007` carries the exact history: `PooledArray<T>`
shipped 2026-05-26 as "`Runtime/PooledArray.cs` — zero-cost `using` wrapper over `ArrayPool`," and the same
line was amended three days later — *"Note 2026-05-29: `PooledArray<T>` was a duplicate of `PooledBuffer<T>`
and has been deleted — its callsites migrated to `PooledBuffer<T>(n, clearMemory: false)`."* That record is
accurate and dated. What never got updated is the **one-line index entries that summarise it elsewhere** —
`Sources/Main/Runtime/README.md:12`'s table still lists `PooledArray` as a present type, and `CLAUDE.md` and
`docs/code-patterns.md:27` both still name it as the `using`-scope option beside `PooledBuffer<T>`. So the
citation this plan received traced back to an index that **contradicts its own project's changelog** — the
historical record was right, the summary of it was not kept in sync. Worth a one-line fix to the three stale
index entries, independently of this plan; not fixed here (`Sources/Main` and `docs/` are read-only to this
analyst regardless). The redirect to `GcHandleScope`/`GcLatencyScope` stands unchanged, and the reason it
matters is now stated precisely: `PooledBuffer<T>` is a plain `struct`, **deliberately**, so that it can be a
class field or captured across an `await` — exactly the property the exclusion wrapper must not have.

---

## The wrapper — requirement, correctly-cited shape, and its two hazards

**Requirement (client, quoted above):** one type, used with `using`, that acquires
`Global\DevOnBike.Overfit.MachineMeasurement` on construction and releases it on `Dispose`, replacing the
hand-rolled sequence duplicated today in `MeasurementExclusion.cs` and `Program.cs`.

**Shape, modelled on `GcHandleScope`/`GcLatencyScope` rather than `PooledBuffer<T>`:** a `readonly ref struct`
(or `ref struct` if it needs a mutable field — `GcHandleScope` is not `readonly` for exactly this reason,
noted at `GcHandleScope.cs:24`) wrapping a `Mutex`. Two hazards the coordinator named, both confirmed correct
against the actual mechanics involved and both satisfied by the `ref struct` shape:

1. **`Dispose` inherits the mutex's thread affinity.** A `using` scope releases on whatever thread the scope
   ends on. For a synchronous scope that is the acquiring thread and nothing goes wrong; a scope spanning an
   `await` can resume elsewhere, and `ReleaseMutex()` then throws — "calling thread does not own the mutex."
   **A `ref struct` makes this impossible by construction**: it cannot be held across an `await` at all
   (`CS4013`/`CS8352`-class compiler errors on any attempt), matching exactly how `GcHandleScope` already
   protects its own OS handle.
2. **`Mutex.Dispose()` does not release the mutex — it closes the handle.** The wrapper's `Dispose()` must
   call `ReleaseMutex()` explicitly, and only when the mutex was actually acquired: a failed
   `WaitOne(TimeSpan.Zero)` must not attempt a release (it would throw, since this thread never owned it), and
   the `AbandonedMutexException` path (previous holder died; the lock is treated as ours) **is** an
   acquisition and therefore **does** need a release. Both existing hand-rolled implementations already get
   this right (`MeasurementExclusion.cs:88-99`, `Program.cs:53-61`) — the wrapper must not regress either
   nuance while collapsing the duplication.

**What the wrapper does not solve on its own.** It makes the two *managed* participants (test host, benchmark
host) uniform. It does not give the **build** side anywhere to live: the build's acquire and release would
need to happen in two different MSBuild targets, and a `ref struct` scoped to one C# method cannot span two
independent target invocations by construction — the same underlying question as Task 0, restated rather than
solved by this design.

---

## Placement — narrowed twice by the client since it was first raised, and smaller than round 3 first stated

**This section was overscoped in the previous pass and is corrected here, not merely extended.** It first
read as a placement question touching all six gated projects. It does not. Two clarifications from the
client narrow it to what actually matters:

**Clarification A — why linking, not referencing:** *"aby nie trzeba bylo referensowac maina calego"* — the
point of linking source rather than taking a `ProjectReference` is to avoid pulling in **the whole
`DevOnBike.Overfit` engine** just to obtain a small synchronisation scope. Stated as a constraint: **whatever
holds the shared type must be reachable without depending on the engine.**

**Clarification B — which is the sharper one, and dissolves most of what clarification A seemed to imply:**
*"Analyzers nie powinny zalezec od kodu w tym repo najlepiej"* — `Sources/Analyzers` should ideally depend on
**no code from this repository at all**, not a `ProjectReference` and not a linked file either, because an
analyzer loads into the **compiler process**: a dependency that is merely heavy in an application is a load
failure in an analyzer, and that failure surfaces as *"the analyzer silently did not run"* — a worse failure
shape than almost anything else this repository names, because nothing announces it.

**Working through both together resolves the apparent five-project debate down to one real question.**
Extending exclusion *coverage* (M5 — all five previously-unguarded projects) is a change to
`OverfitBuildExclusionCheck`'s **gate condition** in `Directory.Build.targets`, which is already auto-imported
into every project in the tree by directory-tree proximity (confirmed for `Templates/` specifically — its own
`.csproj` comment at line 26-27 states its `<Version>` is "inherited from the repo-wide `<Version>` in
`../Directory.Build.props` (auto-imported)," even though the project is deliberately kept out of `Overfit.sln`
and out of Central Package Management). **Widening the condition adds zero C# code and zero dependency to
`Sources/Analyzers`, `Tools/SemanticNavigator`, `Templates/`, `Demo/LabWorkload` or `Demo/LabLoadDriver`.**
None of the five ever needed the shared wrapper type — only a process that itself **acquires** the mutex does,
and that is exactly three participants: the test host, the benchmark host, and the build-side inline task.
Clarification B is therefore already satisfied by construction for `Sources/Analyzers`: it gets exclusion
coverage and gains no dependency, repo-sourced or otherwise.

**One thing worth Task 0 (or the architect) confirming rather than assuming, found while checking
`Templates.csproj` for this correction**: the project sets `<EnableDefaultCompileItems>false</EnableDefaultCompileItems>`
and `<Compile Remove="**\*" />` (it ships template content, not compiled output) — whether `CoreCompile`, and
therefore `BeforeTargets="CoreCompile"`, actually fires for a project with zero compile items is not verified
here. If it does not, `Templates/` needs a different hook (e.g. `BeforeTargets="Build"`) to be genuinely
covered rather than nominally so.

**So the real placement question is narrow: where does the shared type live for its three actual consumers?**

- **Test host and benchmark host** already carry a full `ProjectReference` to `Main` for real, load-bearing
  reasons (they test and benchmark the engine) — verified via the earlier `ProjectReference` grep
  (`Tests.csproj:58`, `Benchmarks.csproj:25`). Clarification A's "avoid referencing the whole engine" does not
  apply to them; they already depend on it fully. `InternalsVisibleTo` (`Main.csproj:45-46`) or a link both
  work at equally low cost — the architect's call, not a client constraint either way.
- **The build-side inline task** is the one participant that structurally **cannot** take a `ProjectReference`
  at all (`RoslynCodeTaskFactory` compiles it in isolation, with no project graph). This is the **only** place
  where "the same file, compiled twice" is a live design question, and the only place the third mutex-name
  literal currently lives. `Source=` (Task 0(e)) is the mechanism that would close it structurally; the
  always-available fallback, unchanged, is a kept literal plus a regression test asserting agreement.

**Trade-offs, narrowed to this one real case:**

1. A linked file compiles to a **distinct type per assembly** — immaterial here, since neither consumer ever
   needs to pass an instance across an assembly boundary.
2. Whatever the build task's `Source=` file contains is bound by `RoslynCodeTaskFactory`'s smaller compiler
   surface (no project references, uncertain namespace/using support) — a plain constant-holder is simple
   enough to satisfy that comfortably, which argues for keeping the shared file to just the constant (and
   possibly the wrapper) rather than more.
3. Does this measurement-only infrastructure belong inside `Main`, the published package, purely because
   `InternalsVisibleTo` happens to already reach there? Not obviously — the alternative (a small neutral file,
   linked into `Tests`/`Benchmarks` and, if `Source=` pans out, into the build task) avoids adding to the
   published DLL's compiled surface for zero engine value. **This is Open question 22, narrowed: a choice
   between `Main` and a neutral location, for exactly two-or-three consumers, not five.**

---

## Task 0 — the spike, now covering six linked questions rather than three

Not runnable now — the `[LongFact]` release gate holds the machine and `Directory.Build.targets` will itself
refuse a build with `error OVERFITMEASURING` while that holds. First task once the box is free. Five of the
six share one root cause: **what state is actually visible, and to whom, across MSBuild's process and thread
model** — grouping them is not padding, it is the same experiment answering five questions from one harness.
The sixth, (f), is unrelated but cheap to check in the same pass since it touches the same file
(`Directory.Build.targets`) and the same "does the gate actually fire" question, just for a different reason.

**(a) Mutex release across two targets, same build.** Can a `Mutex` acquired in one MSBuild inline task be
released in a later target of the *same* build without `ApplicationException` (thread-affinity, confirmed as
a real constraint by the client's own clarification — Fact 7), across `dotnet build`, `dotnet test`'s implicit
build, and `dotnet publish`?

**(b) `FileStream` persistence across targets.** Does a process-scoped `FileStream(FileShare.None)`, held via
static state in an inline-task-generated type, survive across multiple target invocations within one build?

**(c) Fires-once-per-invocation under parallel builds.** Under a parallel, multi-process solution build
(`dotnet build -c Release`, the default), does either mechanism in (a)/(b) fire once per invocation or once
per MSBuild worker process (Risk 8) — tested with all six to-be-gated projects present?

**(d) NEW — MSBuild node reuse, raised by the client directly and changing the severity of (a)/(b) if either
is attempted without answering this first.** MSBuild does **not** terminate its worker nodes when a build
finishes by default — they idle for reuse for roughly 15 minutes (`-nodeReuse:false` /
`MSBUILDDISABLENODEREUSE=1` disable it). If a mutex is acquired inside a node and not correctly released, it
does **not** die with the `dotnet build` command the operator ran — it stays held by a **live**, idle,
off-screen process until the node expires or is recycled. Two things make this worse than an ordinary bug:

   - The existing `AbandonedMutexException` recovery path — which every current participant already relies on
     to treat a dead holder's lock as free — **does not fire**, because the holder is alive, merely idle.
   - The operator sees **nothing**: no console attached to an idle node, no error, just every subsequent
     `dotnet build`/`test`/`benchmark` refusing for up to ~15 minutes with no visible cause.

   Task 0 must measure this directly, not reason about it: deliberately induce a leak (acquire without
   releasing, in the experimental harness) and observe (i) whether the node does keep the mutex live past the
   command's own exit, (ii) for how long, and (iii) whether anything at all is visible to an operator during
   that window. **If confirmed, this is exactly the kind of negative result Decision 14 already covers — it
   escalates to the client, it does not get quietly patched over.** A candidate mitigation exists
   (`-nodeReuse:false` forces node processes to exit after each build) but it is not pre-authorised here: it
   trades away node-reuse's warm-JIT/warm-assembly-load speed benefit for *every* ordinary build on the
   machine, which is a cost to the whole team's daily workflow, not only to this feature, and needs its own
   sign-off if proposed.

**(e) NEW — the linked-source-file mechanism, narrowed to its one real consumer.** Extending exclusion
*coverage* to the five previously-unguarded projects (M5) needs no C# and is not part of this question — see
"Placement," below, for why. The only participant that cannot take a `ProjectReference` at all is the
build-side inline task, so (e) is specifically: does `RoslynCodeTaskFactory`'s `<Code Type="Class"
Source="…">` (or `Type="Method"`) actually work against this repository's SDK/MSBuild version, and if so, can
the shared constant-holding file be structured so the build task compiles it without forcing `Tests`/
`Benchmarks` to also pull in MSBuild-task dependencies they have no other reason to carry? Unverified, flagged
as a claim to check rather than a fact, per the coordinator's own framing.

**(f) NEW — does `BeforeTargets="CoreCompile"` fire for a project with zero compile items?** Found while
checking `Templates.csproj` for the placement correction below: it sets `<Compile Remove="**\*" />` (ships
content, not code). If `CoreCompile` is skipped entirely for such a project, `Templates/` needs a different
hook (e.g. `BeforeTargets="Build"`) to be genuinely covered by M5 rather than nominally so.

**A minor, non-blocking footnote found while re-reading the two existing implementations for this revision:**
`MeasurementExclusion`'s release runs from an `AppDomain.CurrentDomain.ProcessExit` handler
(`MeasurementExclusion.cs:154-165`), not from a `finally` on the same call stack the way the benchmark host's
does. Whether a `ProcessExit` handler is guaranteed to run on the exact thread that acquired the mutex is not
something this analyst has verified against the CLR's own contract, and — unlike the build-side question —
there is no evidence of a problem: this code has shipped and run without a reported `ApplicationException`
here. Not escalated as a blocking risk; named so nobody re-derives it as new later.

**Oracle for the whole spike:** the experiment's own pass/fail log, written into this plan's `## Outcome` or a
follow-up note — not a confident restatement of the reasoning above without having run it.

**If (a)/(b)/(d) are negative, or (d) is confirmed as a real hazard: report to the client per Decision 14.
Do not proceed to Task 3/M4/M5's "hold" design on the strength of a workaround invented under time
pressure.**

---

## Inventory

### Already exists — Exclusion level

| Participant | File | Mechanism | Direction |
|---|---|---|---|
| Test suite | `Tests/MeasurementExclusion.cs` (registered via `[assembly: TestFramework]`) | `Global\DevOnBike.Overfit.MachineMeasurement` mutex, `WaitOne(TimeSpan.Zero)`, held for the process lifetime, released via `ProcessExit` | Holds. Refuses if benchmark or another test process holds it. `Environment.Exit(2)` |
| Benchmark host | `Sources/Benchmark/Program.cs:46-114` | Same mutex, same non-blocking acquire, held for `Main`'s lifetime via `finally { ReleaseMutex() }` | Holds. Refuses if test or another benchmark holds it. Exit code `BusyExitCode = 2` |
| Build (Main only) | `Directory.Build.targets:154-170`, `OverfitBuildExclusionCheck`, `BeforeTargets="CoreCompile"`, gated `MSBuildProjectName == 'Main'` | `Mutex.TryOpenExisting` — probes, never acquires | Checks only. MSBuild `error OVERFITMEASURING` if held |
| Refusal legibility (test side only) | `Tests/MeasurementRefusalMarker.cs` | Writes `Tests/bin/measurement-refusal.txt` | Compensates for `XC-17` (VSTest bridge swallows the console message) |
| Ownership handshake | `Program.cs:87-96`, read by `Directory.Build.targets:156-159` | `OVERFIT_MEASUREMENT_OWNER=<pid>` | Stops the build probe deadlocking the benchmark against its own generated per-job builds |
| Escape hatches | All three | `OVERFIT_ALLOW_CONCURRENT_MEASUREMENT=1`, `OVERFIT_MEASUREMENT_OWNER=<pid>`, `-p:OverfitBuildExclusionCheck=false` | Named, not silent |

**Inter-process vs. thread-affinity, stated once so nobody re-litigates it (client clarification, round 2).**
The `Global\` prefix is exactly what makes the mutex work across processes and even across user sessions —
that is settled and correct, and is why both existing holders can refuse a *different process* entirely.
Thread affinity is the separate, orthogonal constraint that `ReleaseMutex()` must run on the acquiring
thread within whichever process holds it. **Today's two participants are unaffected by it**, and it is worth
saying exactly why rather than leaving a reader to wonder why this was never a problem before: the benchmark
host acquires at the top of `Main` and releases in its `finally` on the same synchronous call stack — no
`await` in between — and the test host holds the mutex for its entire process lifetime, released once at
`ProcessExit`. **The hazard is specific to what is now being asked for**: a build that acquires in one MSBuild
target and releases in a *different* target, where nothing guarantees the same thread executes both.

### Already exists — a different, informal, one-directional attempt at Registration (lab → others), and its producer is real but lives across a durability boundary

This is not the mechanism `ROADMAP.md:517-558` describes (build/test/benchmark register **for the lab** to
read afterward). It is the reverse: the lab tries to make its own presence known so a long test run does not
start on top of it. **Corrected from an earlier pass of this plan, which said neither file had a producer
anywhere in the tree — too strong, and the accurate version is a sharper finding, not a weaker one:**

| Signal | File | Written by | Checked by |
|---|---|---|---|
| `Tests/bin/fp-run.lock` | OS-held exclusive `FileStream(FileShare.None)` | `.claude/run_overnight_measurement.py:34` (and, for the marker below, `:23`) | `Program.cs`'s `MeasurementInProgress` (benchmark host only) — tracked |
| `Tests/bin/fp-run-clean-start.txt` | Timestamp string, age checked against 24h | `.claude/run_clean_measurement.py:17`, `run_overnight_measurement.py:23`, and (unread but present) `run_lab_events.py`, `run_lab_p5.py` | `Scripts/longfact_gate.py`'s `refuse_if_the_box_is_an_instrument()` — tracked, but not the test process itself |

**Every producer lives under `.claude/`, which `.gitignore:276` excludes wholesale.** Three tracked consumers
(`Program.cs`, `longfact_gate.py`, `MeasurementRefusalMarker.cs`'s doc comment) wait for a file that only
untracked scratch tooling can create. A fresh clone gets the readers and not the writers; cleaning the
directory (`.\cleanup.cmd`'s neighbourhood, or a CI runner that never had `.claude/` populated in the first
place) silently disables the whole signal with no error anywhere. **This is the same failure shape
`CLAUDE.md` already records for `Scripts/lab.py`'s own history** — a helper first written into `.claude/` was
gone by the next morning, which is precisely why that file now lives under `Scripts/` instead. The mechanism
here has not yet made that move.

`docs/aiops/aiops-backlog.md:257-259` calls the original watcher "removed the same day" — accurate for the
*specific* script used in that run, but the pattern repeated: four more scratch producers were written after
that note and still live only in `.claude/`. **The incident**: `aiops-backlog.md:275-300`, run A1,
2026-08-06/07 — a bare `dotnet test` bypassed `longfact_gate.py` entirely and evicted Prometheus
mid-measurement; 2 of 11 recorded incidents fall after the eviction, enough to move the pass/fail verdict if
excluded, which they deliberately were not. **The durability gap explains why this survived rather than
being caught immediately**: whoever ran the lab that day very likely had a working producer script locally
(the four in `.claude/` now are evidence such tooling exists and gets used), so it worked *for them*, and
nothing in the tracked repository ever did — the next person to run a 24-hour measurement on a clean clone
gets no producer at all unless they already know to look in an untracked, gitignored directory. The fix is
already named in the backlog and not yet built: *"move the measurement-in-progress check from
`Scripts/longfact_gate.py` to where the **test process** starts, so no entrance bypasses it"*
(`aiops-backlog.md:298-300, 353-355`) — adopted here as Task 5/M6, and now informed by real, working prior
art for M8 (below) rather than a design built from nothing.

### Partially exists — the two `PB-7` gaps (unchanged by scope or wrapper design)

1. Three independent mutex-name string literals; `find_references` on `MeasurementExclusion.MutexName`
   returns exactly one use (its own constructor). Now addressed by the wrapper + linked-file design above,
   pending Open questions 21/22/(e).
2. The build only probes, never excludes. Addressed by Task 0/Task 3 (M4/M5).

### Does not exist

- Any mechanism letting a refusing participant say who, specifically, holds the lock and what kind (build /
  test / benchmark / lab).
- A **tracked, durable** producer for either lab-presence signal — one exists (Fact 4, corrected), but only
  under `.claude/`, which is gitignored. Nothing a fresh clone or a CI runner receives writes either file.
- Subtraction — nothing computes raw-vs-contamination-adjusted incident counts automatically; the one time it
  was done (A1) was by hand and deliberately not substituted for the recorded number.
- A build guard for the five projects that do not reference `Main` — now confirmed in scope.
- **The wrapper itself** (this revision's central new artefact) — no reusable acquire/release scope type
  exists anywhere in the tree for this mutex; the closest shape in spirit (`GcHandleScope`,
  `GcLatencyScope`) wraps different resources entirely.

---

## Problem, user need, business goal, proposed solution

Unchanged from revision 2. **Problem**: four consumers of one shared budget (build, test, benchmark, the
day-long lab measurement) share guards that are individually incomplete and, in two proven cases (`PB-7`,
A1), entirely disconnected from the process they were meant to protect. **User need**: immediate, specific
refusal at the actual entry point, not only in a convenience script. **Business goal**: every number this
repository produces is trustworthy without auditing what else ran alongside it. **Proposed solution**: (a)
symmetric, wrapper-based Exclusion across all six projects; (b) Registration (every build/test/benchmark logs
its window); (c) Subtraction, plus the already-diagnosed reverse-direction fix (M6/M7/M8).

**Success metric**: unchanged — `value: not stated` for a rate/frequency target; the concrete, checkable proxy
is that the `PB-7` shape and the A1 shape both become structurally impossible, verified by mutation-style
regression tests reproducing each incident as a fixture.

---

## Everything not yet settled fact

| # | Type | Statement |
|---|---|---|
| 1 | Fact | `find_references` on `MeasurementExclusion.MutexName` → exactly one use, its own constructor. |
| 2 | Fact | `Main.csproj:45-46` grants `InternalsVisibleTo` to `Tests` and `Benchmarks` already — relevant if the wrapper ends up in `Main`, no longer strictly necessary if it is linked instead (Open question 22). |
| 3 | Fact | Five projects build/publish with zero exclusion coverage today — now confirmed in scope. |
| 4 | Fact (corrected, revision 4, sharpened revision 5) | `fp-run.lock` and `fp-run-clean-start.txt` **do** have producers, and the two files have **different** producers with different designs (see Fact 27) — `.claude/run_clean_measurement.py:21` writes the marker, `run_overnight_measurement.py:34-39` holds the lock (and only reads the marker, at line 47, for its own start time); `run_lab_events.py`/`run_lab_p5.py` present but not individually read. What is actually true, and sharper than "no producer": every one of them lives under `.claude/`, which `.gitignore:276` excludes wholesale. A search scoped to tracked files (this analyst's first pass, and the default for a gitignore-respecting grep) finds only the three tracked consumers and no writer — not because none exists, but because the writer lives across a durability boundary a fresh clone never gets and a cleaned directory loses. Same failure `CLAUDE.md` already records for `Scripts/lab.py`'s own origin (a helper first written into `.claude/` was gone by the next morning). |
| 5 | Fact | A1 (`aiops-backlog.md:275-300`, 2026-08-06/07): a bare `dotnet test` evicted the lab's Prometheus mid-measurement; 2 of 11 incidents fall after the disturbance, enough to flip the pass/fail verdict, deliberately not excluded. |
| 6 | Fact | The fix for gap 4/5 is already named in `aiops-backlog.md:298-300, 353-355` as future work — adopted as Task 5 (M6). |
| 7 | Constraint (confirmed by client) | `System.Threading.Mutex` release is thread-owned, not process-owned; the `Global\` prefix's inter-process scope is a separate, non-conflicting axis. Today's two participants are unaffected (see "Inventory" above for why); the hazard is specific to a hold spanning two different MSBuild targets. |
| 8 | Risk | MSBuild's default parallel, multi-process solution build means an in-memory/env-var dedup for "fires once per build" does not propagate across sibling worker processes — Task 0(c). |
| 9 | Risk | If Fact 7 holds, a full-duration hold for `dotnet build`/`publish` may not be achievable the way the test/benchmark hosts already do it, because the repository owns no entry point for those commands. Per Decision 14, a negative result here is reported, not silently substituted. |
| 10 | Risk (NEW, client-raised, round 2) | MSBuild node reuse (default ~15 min idle retention) can leave a mutex held by a live, off-screen process well after the `dotnet build` command that acquired it has returned — `AbandonedMutexException` does **not** fire for a live holder, so this presents as an unexplained, invisible ~15-minute wedge rather than a recoverable error. Task 0(d). Per Decision 14, if confirmed this is reported to the client, not patched over with `-nodeReuse:false` unilaterally (that trades away a machine-wide build-speed benefit and needs its own sign-off). |
| 11 | Fact (corrected, revision 4) | `Sources/Main/Runtime/PooledArray.cs`, cited as wrapper precedent, does not exist — confirmed. But the historical record of its removal is **correct and dated**: `ROADMAP-COMPLETED.md:2007`, amended 2026-05-29, states `PooledArray<T>` "was a duplicate of `PooledBuffer<T>` and has been deleted." What is stale is three **index** entries that never picked up that change — `Runtime/README.md:12`'s table, `CLAUDE.md`, `docs/code-patterns.md:27` — not the changelog itself. `PooledBuffer<T>` (`Sources/Main/Tensors/PooledBuffer.cs`) is a plain `struct`, **deliberately**, explicitly supporting class-field and closure-captured lifetimes — the wrong shape to copy for a mutex wrapper, which must not have that property. |
| 12 | Fact (NEW) | `Sources/Main/Runtime/GcHandleScope.cs` and `GcLatencyScope.cs` are real, existing `ref struct` acquire/release wrappers with exactly the "cannot escape its scope or be boxed" property the mutex wrapper needs — the correct precedent to model, verified by reading both files. |
| 13 | Open question (NEW, developer/architect) | Does `RoslynCodeTaskFactory`'s `<Code Source="…">` work against this repo's SDK/MSBuild version, for the **one** consumer that needs it (the build-side inline task — see "Placement," narrowed)? Coordinator's own confidence flagged as unverified; carried forward as such, not upgraded. Task 0(e). |
| 13a | Constraint (client, round 3) | `Sources/Analyzers` should depend on no code from this repository at all — a dependency loaded into the compiler process is a load failure there, surfacing as "the analyzer silently did not run." Already satisfied by construction: M5 (extending exclusion coverage) is a pure `Directory.Build.targets` gate-condition change, adding no C# to any of the five newly-covered projects. |
| 13b | Fact (NEW) | Only three participants ever need the shared wrapper/constant type: the test host, the benchmark host (both already carry a full `ProjectReference` to `Main`, verified), and the build-side inline task (which cannot take a `ProjectReference` at all). The other five projects named in M5 need nothing beyond the widened gate condition — verified by reading `Templates.csproj`, which auto-imports `Directory.Build.props`/`.targets` by directory proximity despite being deliberately outside `Overfit.sln` and CPM. |
| 13c | Open question (NEW, developer/architect) | Does `BeforeTargets="CoreCompile"` fire for a project with `<Compile Remove="**\*" />` (zero compile items, e.g. `Templates/`)? If not, that project needs a different hook to be genuinely covered. Task 0(f). |
| 14 | Assumption | "Publish" covers `dotnet publish` of any of the now six gated projects. |
| 15 | Decision (inherited) | Fail-fast, never queue. |
| 16 | Decision (client, round 1) | Registration and Subtraction in scope. |
| 17 | Decision (client, round 1) | All five previously-unguarded projects covered. |
| 18 | Decision (client, round 1) | A negative Task 0 result escalates; the lease is never a silent fallback. |
| 19 | Decision (client, round 2) | One reusable `using`-scoped wrapper type replaces hand-rolled acquire/release logic at every managed call site. |
| 20 | Decision (client, round 5) | The lab's presence signal is a **hard refusal** — blocks build, test and benchmark the same way the mutex does. This overrides `ROADMAP.md:529-535`'s stated reasoning ("a lock that forbids testing for 24 hours will be worked around"), on the strength of Fact 5 (A1's real, measured cost). **`ROADMAP.md`'s text is now stale and needs its own follow-up edit — out of this analyst's write scope (`docs/specs/` and this analyst's memory directory only); flagged for whoever implements this rather than silently left to contradict the shipped behaviour.** |
| 21 | Decision (client, round 5) | Subtraction stays **in** this plan, together with Registration — the client took the larger scope deliberately, consistent with round 1's answer. Sequencing is unchanged and still binds: Subtraction has nothing to correlate against until Registration (M7/M8) produces a real log, so it is a later, dependent task, never a parallel one. If Subtraction's design genuinely needs a log format that does not exist yet, that is a risk with its own spike (Task 8a, below), not something designed against an imagined format. |
| 22 | Open question (developer/architect, NEW) | Where does the shared wrapper/constant file live — inside `Sources/Main` (reachable today via `InternalsVisibleTo`, but adds to the published package's compiled surface for zero engine value), or a neutral location linked into `Tests`/`Benchmarks` directly (no `Main` involvement, cleaner per the coordinator's third trade-off, but a new location this repository's structure does not yet have a home for)? |
| 23 | Open question (developer/architect) | Given Risk 8, how does "fires exactly once per build invocation" survive MSBuild's default parallel, multi-process model across six gated projects? |
| 24 | Open question (developer/architect) | One shared refusal identifier across build/test/benchmark/lab, or is the test side permanently limited to the marker file (`XC-17`)? |
| 25 | Open question (developer/architect) | Should refusals publish holder identity (kind, pid, start time, activity) to a shared, readable lease, replacing both `OVERFIT_MEASUREMENT_OWNER`'s process-inheritance-only reach and today's two disconnected lab markers, whose only producers currently live outside version control (Fact 4)? |
| 26 | Open question (developer/architect), evidence added revision 4 | Where should the lab-presence producer (M8) live? The four existing `.claude/` scripts are all developer-run Python drivers, not in-cluster service code — evidence toward "extend `Scripts/lab.py`," but not a decision, since nobody has confirmed those four are the *intended* design rather than four independent one-off scratch attempts at the same problem. The in-cluster `AnomalyGuardService` reaching the dev box's filesystem remains structurally harder and is not ruled out here (W3). |
| 27 | Fact (NEW, round 5, verified by reading both scripts) | The two existing `.claude/` producer scripts embody two **different** designs with different staleness properties. `run_clean_measurement.py:17-21` writes `fp-run-clean-start.txt` once at start and holds **no lock at all** — if the script dies mid-run, the marker is left behind with nothing to signal it is stale, and a reader has only a guessed TTL (`longfact_gate.py`'s hardcoded 24h) to fall back on, wrong in both directions (too short for a legitimately longer run, too long for a run that died an hour in). `run_overnight_measurement.py:34-45` instead opens `fp-run.lock` and holds an OS byte-range lock (`msvcrt.locking(..., LK_NBLCK, 1)`) for its entire loop — released by the OS automatically however the process dies, with **no duration guess involved**: liveness is "can anything else open this exclusively," not "has N hours passed." `Sources/Benchmark/Program.cs`'s own `MeasurementInProgress` doc comment independently describes the identical intent ("a real OS-held file lock... it disappears when the watcher dies however it dies"), so both sides were built expecting to interoperate, though nothing exercises them together as an automated test today. |
| 28 | Recommendation (analyst, not a decision) | Given Fact 27 and the new hard-refusal stakes (a wrong staleness call now blocks the whole team, not just a warning), M7's unification should standardise on the **lock-based** mechanism rather than the timestamp+TTL one, because it inherits the same crash-safety property the `Global\` mutex already relies on elsewhere in this plan, with no duration to guess. Left for the architect to confirm (Open question 29), not decided here. |
| 29 | Open question (developer/architect, NEW round 5) | Does a Python `msvcrt.locking` byte-range lock actually block, and get blocked by, a C# `FileStream(..., FileShare.None)` open on the same file? Both sides' doc comments describe matching intent (Fact 27), but nothing in this repository tests the two together automatically. Lower uncertainty than Task 0's MSBuild questions — the design already assumes it works — but worth a cheap regression test once M7 lands, rather than continued assumption. |
| 30 | Requirement (client, round 5) | The escape hatch for a day-long block needs its own design, not automatic reuse of the existing three. `OVERFIT_ALLOW_CONCURRENT_MEASUREMENT=1` is shaped for a short override remembered for one command; left set in a persistent shell profile after a legitimate emergency, it silently disables protection for every subsequent build/test/benchmark that developer runs for however long they forget it — a far larger blast radius at 24-hour stakes than at benchmark-length ones. |
| 31 | Open question (developer/architect, NEW round 5) | What does a deliberate override of the lab hard-refusal look like, given Requirement 30? A per-invocation flag rather than a persistent env var; a required reason string, matching this repo's own `OverfitAcceptedPrerelease` pattern (`Directory.Build.targets`'s prerelease check, which requires `Reason=`/`Since=` on every accepted entry) — named as options to weigh, not decided here. |
| 32 | Requirement (client, round 5) | The refusal message, at this duration, must name the holder, when it started, how long it is expected to run, and how to override deliberately — "something is measuring" is not enough for a block that can cost a developer a full day. `run_overnight_measurement.py:51-52` already computes and prints exactly this (`gate at … ends …`), so the expected-end time is available at the producer; it must be carried into whatever marker/lease the check reads, which none of today's four scripts currently expose in a machine-readable way (each only prints it to its own console). This is a field the M2/M3 wrapper/refusal design had no reason to carry for build/test/benchmark refusals of normally-short, unpredictable duration. |
| 33 | Constraint (client, round 5) | Durability is part of the definition of done, not an enhancement: a hard refusal keyed on a `.claude/`-only marker fails **open** on a fresh clone (nothing writes it, so nothing ever refuses) while the shipped code looks like it enforces the block. **Sequencing consequence**: M8 (the durable, tracked producer) must exist and be the documented way to start a lab run before M6's hard-refusal checks can be treated as protecting anyone — shipping the check without the tracked producer does not wedge the machine, but it produces exactly the false confidence the client is warning against. |

---

## Gate answers

Unchanged from revision 2: execution path neither inference nor training (build/CI/lab tooling); oracle is
mutation-style, reproducing the `PB-7` and A1 incidents as fixtures, plus (if Subtraction lands) replay of
the A1 log against the hand-derived 8.85/day as a known-good check; AOT reach no; allocation policy not
applicable; moat side neither.

---

## Scope (MoSCoW)

### Must

- **M1 (restructured this revision).** Build the wrapper — a `ref struct`, modelled on `GcHandleScope`/
  `GcLatencyScope` (not `PooledBuffer<T>`), acquiring `WaitOne(TimeSpan.Zero)` (treating
  `AbandonedMutexException` as acquired, per existing behaviour), releasing exactly once and only if actually
  acquired. Both managed participants (test host, benchmark host) adopt it, collapsing two of the three
  duplicated mutex-name literals and the duplicated acquire/release sequence. Placement per Open question 22.
- **M2.** Every refusal fails immediately with a named, informative message — the wrapper is the natural home
  for composing it once rather than three times (Open question 24/25 decide the exact shape).
- **M3.** Resolve Open questions 22-25 with the architect before implementing the shared parts.
- **M4.** The build actually excludes, not only is excluded — contingent on Task 0. Negative result escalates
  (Decision 18), including the node-reuse hazard (Risk 10) if confirmed.
- **M5.** Extend exclusion coverage to all five previously-unguarded projects. **Confirmed a pure
  `Directory.Build.targets` gate-condition change — no C# code, no dependency, added to any of the five**
  (Fact 13b), which is what makes it safe for `Sources/Analyzers` under Constraint 13a. Subject to Open
  question 23 (fires-once-under-parallel-builds) and 13c (`CoreCompile` firing for zero-compile-item
  projects, `Templates/` specifically).
- **M6 (widened, round 5 — was test-only, now all three).** Add a **hard refusal** on the unified
  lab-presence signal to `MeasurementExclusion` (test), the build-side probe (`Directory.Build.targets`), and
  `Program.cs` (benchmark) — the client's Decision 20 makes this symmetric with the machine-exclusion mutex
  itself, not only a test-side fix for the A1 gap. **Highest value-per-cost item in the plan** — the only one
  with a measured incident behind it — but **must not activate before M8 ships** (Constraint 33): a hard
  refusal reading a signal only `.claude/` scripts can write fails open, silently, on every machine that does
  not happen to have that untracked tooling.
- **M7.** Unify `fp-run.lock` and `fp-run-clean-start.txt` into one signal, one producer contract. **Recommend
  standardising on the lock-based mechanism** (Recommendation 28), pending Open question 29 (cross-language
  lock interop, low uncertainty but unverified).
- **M8.** Move a producer for that signal from `.claude/` (untracked, gitignored — Fact 4) into a tracked,
  durable location, mirroring the precedent `CLAUDE.md` already records for `Scripts/lab.py`'s own history.
  Lower cost than "build from nothing" — four working scratch scripts already do this — but **now a blocking
  dependency of M6, not merely a nice-to-have that happens to be cheap** (Constraint 33).
- **M9 (NEW, round 5, from the client's hard-refusal requirements).** Design and implement: (a) a
  deliberately-separate escape hatch for the 24-hour case (Requirement 30, Open question 31), (b) a refusal
  message carrying holder identity, start time, expected end time and the override instructions (Requirement
  32) — which requires the producer (M8) to expose the expected-end time in a machine-readable way, not only
  print it to its own console the way all four `.claude/` scripts do today.

### Should

*(empty — Subtraction moved to Must per Decision 21, below)*

### Must (continued — Subtraction, moved here per the client's round-5 answer)

- **M10 (was S1).** Subtraction proper. **NOT READY**, and deliberately kept that way rather than designed
  against a format that does not exist yet (Decision 21): gated on M7/M8 producing at least one real
  Registration log. If, once that log exists, its shape does not support the correlation Subtraction needs,
  that is a finding for a spike (Task 8a), not a reason to guess the schema now.

### Could

- **C1.** Benchmark-side and build-side siblings of `MeasurementRefusalMarker` — lower value; neither
  MSBuild's nor the benchmark host's console is swallowed the way the VSTest bridge swallows the test host's.

### Won't (this time)

- **W1.** Fixing `XC-17` itself.
- **W2.** A general "machine is busy" concept beyond what this repository's own tooling starts.
- **W3.** Deciding, in this plan, whether the in-cluster `AnomalyGuardService` becomes the lab-presence
  producer (Open question 26, below) — a real candidate, not ruled out, but not decided here.

---

## Value against cost

| | Value | Structural cost | Uncertain | Recommendation |
|---|---|---|---|---|
| M1 (wrapper) | Not stated; collapses 2 of 3 literals AND the duplicated acquire/release logic in one move — a stronger fix than round 1's constant-only design | Low — one new `ref struct` type modelled on an existing, verified precedent (`GcHandleScope`); only 2-3 real consumers (13b), not 5; placement question (22) adds design cost, not implementation cost | Where it lives for its 2-3 consumers; whether `Source=` unifies the build task's copy too | **Do now in design; implement after Open q 22 is answered** |
| M4 (build holds the lock) | The client's central, stated ask | High and unknown until Task 0, now covering 6 linked questions including a newly-raised, potentially severe node-reuse hazard | The whole mechanism's feasibility, including an operator-visible ~15-minute wedge if Risk 10 is real | **Spike first; treat a positive node-reuse finding as a stop, not a detail** |
| M5 (5 more projects covered) | Client confirmed literal scope | **Low, revised down** — confirmed a pure MSBuild gate-condition change, no C# or dependency added anywhere (Fact 13b); the earlier "placement debate across 5 projects" did not apply to any of them | Whether `CoreCompile` fires for zero-compile-item projects (13c/Task 0(f)); whether the condition can fire once under parallel builds (Open q 23, shared with M4) | **Do alongside M4's spike — cheap, and answers (c)/(f) settle it either way** |
| M6 (hard refusal, all three) | Real, dated, measured cost already paid once (a lab-day's validity); now blocks a whole team for up to 24h when it fires, which raises the value AND the cost of getting it wrong | Low mechanically (the fix is already named in the backlog); **the activation gate (Constraint 33) is the real cost** — must not ship live before M8 | Whether M8 lands in time to avoid a "looks enforced, isn't" gap | **Build the check now; gate its activation on M8, explicitly, not by coincidence of merge order** |
| M7 (unify onto the lock-based design) | Not stated; retiring a real staleness risk that a hard refusal makes much more expensive to get wrong | Low-medium — the lock-based mechanism already exists and works (`run_overnight_measurement.py`); the work is porting and testing the cross-language interop (Open q 29) | Whether `msvcrt.locking`/`FileShare.None` interop holds under test, not only by design intent | **Do — and prefer the lock design over the timestamp one (Recommendation 28), architect to confirm** |
| M8 (durable producer) | Not stated; without it M6 checks a file a tracked checkout never produces, same as today, now with the added risk of *appearing* enforced | **Lower than previously stated** — closer to "port and consolidate a working script" than "design from nothing" (Fact 4, corrected) | Which of the two designs to port (M7 decides this); Open question 26 (location) | **Do before M6 activates — Constraint 33 makes this a hard dependency, not a sequencing preference** |
| M9 (escape hatch + message content for the hard refusal) | Client-stated requirement, no number attached | Low-medium — mostly design and message-composition work, contingent on M8 exposing expected-end time machine-readably | Shape of the override mechanism (Open q 31); none of today's scripts expose expected-end time outside their own console | **Design alongside M6/M7/M8 — cannot be bolted on after the hard refusal ships, since the message format is part of what "done" means here (Requirement 32)** |
| M10 (Subtraction, was S1) | Client says in scope, deliberately, twice now | Unknown until M7/M8 exist and produce a real log — explicitly NOT sized further than that | Open question 21's sequencing is now a Decision; the remaining uncertainty is the log's actual shape, unknown until it exists | **Spike (Task 8a) against the real log once M7/M8 land; do not design the schema now** |

---

## Ordering — highest uncertainty first, correctness before anything resembling performance

1. **Task 0** — the five-question spike, above. Blocked on the machine.
2. **Task 1 — M1 (wrapper) + M2/M3**, blocked on Open questions 22/24/25, but the *design* (shape, hazards) is
   settled in this plan and does not need Task 0 to proceed conceptually — only the third participant's
   participation (build) does.
3. **Task 3 — M4/M5**, blocked on Task 0. Branches exactly as revision 2 specified: positive → symmetric
   build exclusion across all six projects; negative → report to client, describe the lease as an option,
   implement nothing further without sign-off.
4. **Task 6 — M7 (unify onto the lock-based design, Recommendation 28)**, before Task 7.
5. **Task 7 — M8 (durable producer)**, before Task 5 can be activated. Needs Open question 26 answered
   (location) and inherits M7's choice of mechanism.
6. **Task 5 — M6 (hard refusal, all three participants)**, code written any time after Task 1, but its
   **activation gated on Task 7 landing** (Constraint 33) — the check may exist in a disabled or
   not-yet-wired state before then; it must not go live first. Highest value-per-cost item in the plan once
   activated correctly.
7. **Task 5a — M9 (escape hatch + message content)**, alongside Task 5, not after it — Requirement 32 makes
   the message format part of what "done" means for the hard refusal, not a follow-up polish pass.
8. **Task 8 — M10/Subtraction**, NOT READY, gated on Tasks 6/7 producing a real log.
9. **Task 8a — Subtraction's own spike**, if Task 8's premise (a usable log exists) turns out false or the
   log's actual shape does not support the correlation Subtraction needs — named explicitly per Decision 21,
   rather than left implicit the way "NOT READY" alone would leave it.

---

## Traceability

| Goal | User need | Task | Acceptance criterion | Verified by |
|---|---|---|---|---|
| No further `PB-7`-shape incidents | Operator knows immediately the machine is busy | Task 1, Task 3 | Wrapper adopted by both managed hosts; build excludes symmetrically | Mutation tests; seven-arm manual verification extended |
| No further A1-shape incidents | Operator knows immediately a lab measurement is live, and is blocked, not merely warned | Task 5, 5a, 6, 7 | `dotnet test`/`build`/benchmark all hard-refuse while the unified, durable lab signal is live; message names holder, start, expected end, override | A1 reproduced as a fixture; message content asserted field-by-field |
| A hard refusal never fails open while looking enforced | A developer trusts the shipped code, not tribal knowledge of an untracked script | Task 7 before Task 5 activates | M8 ships and is the documented way to start a lab run before M6's check goes live | Constraint 33; activation order checked, not merely stated |
| A 24-hour block does not become a forgotten, silent bypass | An operator can override deliberately without disabling protection for days afterward | Task 5a | Escape hatch is per-invocation or reason-carrying, not a persistent env var left set | Design review against Requirement 30 |
| Feasibility known before cost is spent, including the node-reuse hazard | Nobody builds on an unverified assumption, and nobody is left with a silently wedged machine | Task 0 | Answers (a)-(f) together, with (d) treated as a stop condition if positive | The experiment's own log |
| Trustworthy lab false-positive numbers | The rate a client is shown is not silently inflated | Task 8/8a (NOT READY) | Automated figure matches the A1 hand-calculation (8.85/day) on replay | Compare against the recorded hand-derived number |

---

## BLOCKING QUESTIONS

**Round 1's client questions 1 and 2 are answered (Decisions 20, 21) and removed from this list.** Two new
ones replace them, both narrower and both about *how*, not *whether* — the *whether* is settled.

**For the client:**

1. `ROADMAP.md:517-558`'s "why the lab does not join the mutex" section is now factually wrong (Decision 20
   overrides it) and this analyst cannot edit it (out of write scope — `docs/specs/` and this analyst's memory
   directory only). Who updates it, and on what timeline relative to this plan shipping — before, so the
   ROADMAP never contradicts shipped behaviour, or after, as part of the same change?
2. Open question 31's override mechanism (a per-invocation flag, a required reason string, or something else)
   changes daily workflow for anyone who legitimately needs to build during a lab run. Is there a preference,
   or is this fully delegated to the architect?

**For the developer/architect:**

3. Task 0(a)-(e) — run the spike and report, including the node-reuse hazard as a first-class measured
   question, not a reasoned-about one.
4. If Task 0 is negative on any of (a)/(b)/(d): is a lease-based approximation acceptable **for the client to
   approve** — not a decision the architect or developer may make alone (Decision 18)?
5. Open question 22 — where does the shared wrapper/constant file live for its **2-3 actual consumers**
   (test host, benchmark host, and — if `Source=` pans out — the build task; **not** the five projects being
   newly covered by M5, which need no C# at all, Fact 13b): inside `Main` (simplest given existing
   `InternalsVisibleTo`, but adds to the published package for zero engine value) or a neutral linked
   location (cleaner separation, needs a new home)?
6. Open question 13/(e) — does `RoslynCodeTaskFactory`'s `Source=` attribute actually work here, for the one
   consumer that needs it (the build-side task)? And 13c/Task 0(f) — does `BeforeTargets="CoreCompile"` fire
   for a zero-compile-item project like `Templates/`?
7. Open questions 23-25 — once-per-build-invocation mechanism under parallel builds; one shared refusal
   identifier or not; shared holder-identity lease or not.
8. Where should the lab-presence producer (M8) live? Four working scripts already do this job from
   `.claude/` — untracked, gitignored, not durable (Fact 4). Should M8 promote and consolidate those four
   into `Scripts/`, matching the precedent already set for `Scripts/lab.py`'s own history, or does the
   in-cluster `AnomalyGuardService` need a way to reach the dev box's filesystem instead? Neither is decided
   here (W3).
9. Confirm Recommendation 28 (unify M7 onto the lock-based `fp-run.lock` design rather than the timestamp-
   based `fp-run-clean-start.txt` one) or say why the timestamp design should be kept instead — this analyst's
   reasoning favours the lock design given the new hard-refusal stakes, but the choice is the architect's.
10. Open question 29 — verify, once M7 lands, that a Python `msvcrt.locking` byte-range lock and a C#
    `FileStream(..., FileShare.None)` open actually interoperate on the same file; both sides' code already
    assumes it (matching doc comments in `run_overnight_measurement.py` and `Program.cs`), but nothing tests
    it together today.
11. Open question 31 — the escape-hatch design for the hard refusal (Requirement 30): per-invocation flag,
    required reason string matching the `OverfitAcceptedPrerelease` pattern, or another shape entirely?
12. Confirm the activation-ordering constraint (Constraint 33, M6 must not go live before M8) is actually
    enforceable in however this ships — e.g., a single PR landing both together, or an explicit flag that
    keeps the hard-refusal check inert until the producer's location is confirmed present.

---

## SUGGESTED IMPROVEMENTS TO MY ROLE

**Two real catches this run, both worth recording because both are the "verify before you answer" rule
working, not a gap in it — flagged because each one changed a design decision, not just a citation.**

1. The wrapper request arrived with a specific, named precedent (`Sources/Main/Runtime/PooledArray.cs`) that
   does not exist in the current tree — only stale documentation does. Had I cited it back without checking,
   the plan would have pointed the architect at a nonexistent file and, worse, at the wrong *shape* (the real
   type, `PooledBuffer<T>`, is explicitly designed for closures and class fields — the opposite property the
   wrapper needs).
2. The placement question, as first relayed, read as a five-project debate (does every newly-covered project
   need to reference or link the shared type). Checking `Templates.csproj` directly showed `Directory.Build.
   targets` already reaches it by directory-tree proximity regardless of solution/CPM membership, and
   re-reading the coordinator's own two messages together showed the real consumer count is 2-3, not 5-6 —
   the M5 coverage change and the M1 wrapper-placement question are two different mechanisms that had been
   discussed as if they shared a cost, and they do not.

Nothing to change in my own instructions from either. Both resolved inside the normal round-trip; recorded so
neither gets re-litigated as "the plan said five projects need this" later.

**Revision 4 adds a third, and it is on me rather than on a source I trusted.** Both round-3 findings above
were directionally right but imprecise in exactly the way a gitignore-respecting default search produces:
"only stale documentation remains" was true of the *symptom* (three index entries) but skipped the *record*
that got it right (`ROADMAP-COMPLETED.md:2007`, correctly dated), and "no producer anywhere in the tree" was
true only of the tracked tree, which is what `Grep` searches by default — I did not think to widen the search
past `.gitignore` before asserting absence. Both were caught by the coordinator searching more broadly than I
did, not by reasoning I had access to and skipped. **Worth naming as a standing pattern rather than two
one-off misses**: "not found by a default search" and "does not exist" are different claims, and this
repository specifically keeps working, load-bearing tooling in a gitignored directory (`.claude/`) by
convention (`CLAUDE.md`'s own `do-<agent>.py` scratch-file rule lives there) — so for *this* repository,
absence-from-tracked-files is a weaker claim than it would be elsewhere, and worth widening the search for
before asserting it, not only when a coordinator's correction prompts it.

**Revision 5, smaller correction, self-caught this time rather than relayed.** My own revision-4 fix said
`run_overnight_measurement.py:23,34` "write both paths." Reading the file properly for round 5's staleness
question (rather than only grepping for the filename, which is all the earlier pass did) showed line 23 only
*defines* the marker path — the script reads it at line 47 and never writes it; only the lock (line 34-39) is
actually produced there. A grep that confirms a string appears at a line is not the same claim as "this line
writes the file," and conflating them nearly went uncorrected a second time in the same document.
