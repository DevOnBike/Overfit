
---

## Idea: "Advanced .NET Performance" online course/workshop (Overfit as the case study)

**Premise:** monetize the perf expertise embodied in Overfit by teaching it — an online course / live cohort / corporate workshop on *real-world* high-performance .NET, using the Overfit engine as the running case study.

**Why it's strong:**
- **Credibility moat.** "I built a pure-C# LLM engine competitive with llama.cpp on CPU — zero-alloc, Native-AOT, no native deps" is a killer credential. Proof, not theory. Almost nobody can teach from that.
- **Content already exists.** The codebase + dev sessions ARE the curriculum: zero-allocation patterns (PooledBuffer, caller-owned buffers), SIMD/AVX2 intrinsics + TensorPrimitives, the `OverfitParallel` fn-pointer spin-dispatch (zero-alloc parallelism), quantized GEMV kernels (Q4_K/Q8), memory-bandwidth-bound analysis, cache/L1 reasoning, the custom Roslyn perf analyzer (OVERFIT001-020 + [OverfitHotPath]), benchmarking methodology (BenchmarkDotNet + MemoryDiagnoser + best-of-N rigor), AND **negative results** (register-blocking LOST to TensorPrimitives; "measure don't assume") — negatives are gold pedagogically.
- **Underserved niche.** Advanced .NET CPU perf is scarce content (most is beginner/intermediate). Buyers: senior .NET devs, fintech/HFT/trading, gamedev, anyone doing latency-sensitive .NET.
- **Top-of-funnel for the product.** Course → leads for the commercial Overfit services ("Zero-GC inference audit", "Private .NET RAG PoC", consulting). Course markets product, product proves course.

**Caveats (honest):**
- Niche audience → price PREMIUM (high price, fewer students), not mass-market.
- Production is real work (scripting/recording/editing) — but the material is mostly generated.
- Differentiation vs existing (Kokosa "Pro .NET Memory", Toub's blog, Dometrain perf): the **real LLM-engine case study + competitive-with-native + the analyzer/methodology**, not toy examples.

**Low-risk validation path (recommended before a big course build):**
1. A few YouTube deep-dives or a conference talk/workshop proposal (NDC, .NET Conf, DotNext) — Overfit as case study. Gauge demand cheaply.
2. If interest: a paid live cohort/workshop (high-touch, high-price) before a polished recorded course.
3. Channels: own platform / Dometrain / Udemy / corporate on-site. Live cohort = highest margin per student.

**Verdict:** strong, well-aligned, leverages the moat, doubles as product marketing. Validate demand cheaply first (talk/YouTube), then cohort, then recorded course.

### Differentiation deep-dive (vs Kokosa / Toub / Dometrain)

**The market gap:** nobody teaches perf through ONE real, ambitious system end-to-end with failures shown. Existing = (a) theory/internals (Kokosa "Pro .NET Memory" — book, "how the runtime works"), (b) fragmented reference (Toub's per-release posts), (c) toy examples (Dometrain — string/LINQ/Span tricks, intro→intermediate). Senior perf-engineering (system-level tradeoffs, measuring, dead-ends) is taught nowhere.

**5 axes of differentiation:**
1. **Case study IS the edge.** Not "speed up a loop" — "a pure-C# LLM engine that MUST be competitive with llama.cpp (C++/AVX-512), zero-alloc, AOT." One system getting faster across modules; micro-decisions compound. How seniors actually think.
2. **Competitive-with-native as forcing function.** Bar = "within 1.13-1.21× of llama.cpp," not "faster than naive." Forces real technique (bandwidth analysis, Q4_K/Q8 GEMV, intrinsics, cache, parallel dispatch). Teaches where managed .NET CAN match native and where it structurally can't (CNN ~11-30× behind MLAS — and why). Plus moat thinking: where to compete vs concede.
3. **Methodology + the analyzer = teach the PROCESS.** Empirical rigor (best-of-N both sides, A/B isolation, MemoryDiagnoser, "measure don't assume," documenting unflattering results) — the transferable meta-skill. The custom Roslyn analyzer (OVERFIT001-020 + [OverfitHotPath]) = unique artifact: enforce perf invariants at build time. Nobody teaches "build your own perf guardrails."
4. **Negative results = rarest, most valuable.** register-blocking REGRESSED (lost to TensorPrimitives); "parity 17.12" retracted as single-run noise; AVX-512 port disproven by floor math; whole-matrix-Q4_K loses single-thread / wins only parallel. "Here's the optimization I was SURE about, here's the measurement it failed, here's why" = gold. Antidote to cargo-cult perf. Nobody else shows the failures.
5. **Real subsystems as modules:** zero-alloc hot path → SIMD/intrinsics (when hand-SIMD beats/loses TensorPrimitives) → alloc-free parallelism (OverfitParallel fn-pointer dispatch vs Parallel.For's 925KB closures) → quantization + bandwidth-bound kernels → build-time enforcement → benchmarking methodology → AOT discipline.

**One-liner:** *"Not perf tricks. The full engineering of a .NET system that has to disprove the assumption you need C++. One real engine, every decision measured — including the ones that failed."* Kokosa=theory · Toub=fragments · Dometrain=toys · Overfit=senior perf-engineering end-to-end on a real ambitious system with the failures shown.

**Caveat:** depth is also the risk — NOT for beginners, narrower (senior) audience → premium price/positioning. The brand stands on continued honesty about where .NET loses (CNN vs MLAS, 1.13× vs llama.cpp) — that honesty IS the differentiator vs hype content.

---

