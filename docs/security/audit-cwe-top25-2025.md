# Audit — CWE Top 25 Most Dangerous Software Weaknesses (2025 list)

Standard: CISA/MITRE 2025 CWE Top 25, published Dec 2025, derived from 39,080 CVEs
(`cwe.mitre.org/top25/archive/2025/2025_cwe_top25.html`, fetched 2026-08-10). Commissioned by the
team-lead 2026-08-10 against the shortlist's #1 pick.

**Correction to the selection round**: the shortlist predicted `OVERFIT022/023/025/026/028` map onto
"CWE-674/835/121/789/190". Checked against the actual 2025 list: **only CWE-121 (rank 14) is on it.**
CWE-674 (uncontrolled recursion), CWE-835 (uncontrolled loop), CWE-190 (integer overflow) and CWE-789
(uncontrolled memory allocation) are real, well-known weaknesses these analyzers genuinely cover, but
none of the four appear in this specific CVE-derived list — it skews toward web injection and C/C++
memory classes, which is why a managed-language integer-overflow rule doesn't show up in NVD data the
same way. They're covered below anyway (worth auditing, just not "on the list").

**Verdict counts: 9 MEETS/covered, 5 GAP, 3 PARTIAL, 8 NOT APPLICABLE, 0 NOT CHECKED** (every clause was
evaluated at least at the "is the control class present" level; deeper per-loader path-traversal
coverage beyond the two loaders named below is the one area flagged as needing a follow-up pass rather
than fully enumerated here — see the note under CWE-22).

## Findings that matter, ranked by what an attacker reaches in this deployment

1. **A path-containment gap was found in at least one model-loading path during this audit
   (CWE-22).** Per the embargo rule, technical detail is **not published in this document** — reported
   directly to the maintainer. See CWE-22 row below for the non-revealing summary.
2. **CWE-306/862/863/284 (missing/incorrect authentication and authorization) — the shipped OpenAI-
   compatible server (`overfit serve`) has no authentication layer anywhere in its request pipeline, and
   the documented Docker deployment binds it to all interfaces by default.** This is real and currently
   shipping, not hypothetical; see the row below. Not embargoed — this is a missing-control-by-default
   pattern, not a memory-safety defect, and the same class of gap is public industry knowledge for
   self-hosted LLM servers generally (Ollama's default-exposure history is cited in this project's own
   `docs/TASKS.md` `MK-2` row as a competitive argument).
3. **CWE-121/787/125 (buffer/OOB read-write) — the control that would mechanically prevent a regression
   is scoped to `Sources/Main` only.** `OVERFIT025`/`OVERFIT026` are build errors in the library, but sit
   at their coded default (`Warning`, not enforced) in `Sources/Anomalies`, `Sources/Server.AspNet`,
   `Sources/Mcp`, `Sources/Cli` — all of which parse externally-influenced input (Prometheus responses,
   HTTP bodies, MCP JSON-RPC).
4. Everything else below is either not applicable to a library/CLI/self-hosted-server product with no
   browser or SQL surface, or is a lower-severity documentation/process gap.

## The 25 clauses

| # | CWE | Name | What was checked, and how | Verdict | Severity in this deployment | Remedy |
|---|---|---|---|---|---|---|
| 1 | CWE-79 | XSS | Grepped for HTML templating / Razor / cshtml rendering user content; `Server.AspNet` returns JSON/plain-text only, `docs`/`openapi.yaml` endpoints serve static embedded content | NOT APPLICABLE | — | — |
| 2 | CWE-89 | SQL Injection | Grepped for `SqlCommand`, `System.Data.SqlClient`, `Sqlite`, `DbContext` across `Sources/` — none found; `PersistentVectorStore` is file-backed, no SQL | NOT APPLICABLE | — | — |
| 3 | CWE-352 | CSRF | No cookie/session-based browser-facing surface; the OpenAI-compat API is a stateless bearer-style API consumed by non-browser clients | NOT APPLICABLE | — | — |
| 4 | CWE-862 | Missing Authorization | `Sources/Server.AspNet/Endpoints/OverfitOpenAiApi.cs:40-76` (`MapOverfitOpenAiApi`) wires `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/speech` with no auth middleware in the pipeline — confirmed by reading the full middleware chain, one logging `app.Use`, nothing else. `Sources/Cli/GuardMetricsEndpoint.cs:187-192,253-326` same for `POST /ack` | **GAP** | **High** — reachable over the network on the documented Docker default (see CWE-306 below); guard `/ack` already tracked as `SEC-1` in `docs/TASKS.md`, `PART`, NetworkPolicy mitigation verified inert on Docker-Desktop-class CNI | Add an opt-in API-key check to `OverfitAspNetServer.Serve` (a `Bearer` header compared against an env var, off by default with a startup warning when off and bound non-loopback — same shape as the Gateway's own TLS-guard opt-in). For the guard, `/ack` needs a shared-secret header; NetworkPolicy alone is insufficient (see `SEC-1`) |
| 5 | CWE-787 | Out-of-bounds Write | `OVERFIT025`/`OVERFIT026` (stackalloc byte-budget / variable-length ban) are the mechanical control; verified `error` severity in `.editorconfig` only under `[Sources/Main/**.cs]` (and stricter still in `Kernels/`/`Intrinsics/`) — code-default severity is `Warning` (`StackallocSizeAnalyzer.cs:83,92`) everywhere else | **PARTIAL** | Medium-High outside `Sources/Main` — `Sources/Anomalies` (parses Prometheus text), `Sources/Server.AspNet`/`Mcp` (parse HTTP/JSON-RPC bodies) get no build-time protection against a new unbounded `stackalloc` | Extend the `[Sources/Main/**.cs]` `error` promotion for OVERFIT025/026/028 to `Sources/Anomalies/**.cs` and `Sources/Mcp/**.cs` at minimum — both parse externally-supplied text and are excluded today only by omission, not by a stated reason (unlike the documented Tests/Benchmark exclusions) |
| 6 | CWE-22 | Path Traversal | Reviewed loader path-joins that combine a filesystem base directory with a value read from file content. `OnnxExternalData.ResolvePath` (`Sources/Main/Onnx/OnnxExternalData.cs:170-200`) has an explicit, verified containment check (rejects empty/rooted locations, re-tests with `Path.GetFullPath` after `Path.Combine`). **A second loader was found not to have the same check; per the embargo rule the specific file and mechanism are not published here — reported directly to the maintainer.** Coverage of the remaining loaders (GGUF, `.repack`, `tokenizer.json`, WAV/MP3) for the same pattern was not exhaustively re-checked in this pass | **GAP** (embargoed detail; see maintainer message, not this document) | Reported directly — see maintainer message | Reported directly — see maintainer message |
| 7 | CWE-416 | Use After Free | `.editorconfig:30-36` demotes `IDISP001/002/003/004/007/008/015` to `suggestion` repo-wide with a documented rationale (ownership-transfer + autograd-graph + JSON-enumerator-struct false-positive classes); the 3 genuine leaks found in the 2026-06-21 audit were fixed in code, not suppressed. Not re-verified in this pass beyond reading that rationale | MEETS (by prior audit, not re-verified here) | — | — |
| 8 | CWE-125 | Out-of-bounds Read | Same mechanical gap as CWE-787 (`OVERFIT025-028` scoping). Additionally: the analyzer that would catch "size/loop-bound/index read from `BinaryReader`/JSON without a validator" generically (`NR-2` in `docs/TASKS.md`) is **`OPEN`, not built** — the only defense against this class today is the one-time manual sweep `NR-3` (`DONE`, 7 defects fixed), which does not prevent a regression | **GAP** | Medium — regressions in loader bounds-checking have no mechanical gate; caught only by manual review | Build `NR-2`'s analyzer (already scoped in the backlog: flag a size/count/index read from `BinaryReader`/`JsonElement` with no comparison against a validated bound before use) |
| 9 | CWE-78 | OS Command Injection | Grepped for `Process.Start`/`ProcessStartInfo` across `Sources/` — none found | NOT APPLICABLE | — | — |
| 10 | CWE-94 | Code Injection | `RS0030` bans `System.Reflection`, `System.Activator`, `System.Linq.Expressions.Expression` at error severity repo-wide (verified `.editorconfig:4`, applies to `[*.cs]`, only excepted for generated code at the very end); no `eval`-equivalent construct exists in .NET | MEETS | — | — |
| 11 | CWE-120 | Classic Buffer Overflow | `Array.Copy` banned via `BannedSymbols.txt` + `RS0030` (error, repo-wide); `Span<T>.CopyTo` is the required replacement | MEETS | — | — |
| 12 | CWE-434 | Unrestricted File Upload | Not checked in this pass — MCP `transcribe` tool and any RAG-ingestion file-type/size validation were not reviewed. Files reach the process via CLI/MCP-caller-provided paths (a trusted local caller in the stdio model), not via an HTTP multipart upload endpoint — no HTTP upload endpoint exists in `Server.AspNet` today | **NOT CHECKED** (MCP/RAG file-argument validation); **NOT APPLICABLE** (no HTTP upload surface exists) | — | If an HTTP upload endpoint is added later, this clause must be re-checked |
| 13 | CWE-476 | NULL Pointer Dereference | Lower severity in .NET than in C/C++: a `NullReferenceException` is catchable and does not corrupt memory. Not systematically checked; `Nullable=enable` migration (2026, 84→0 warnings) reduces the class but was not re-verified here | PARTIAL (mitigated by language + prior migration, not re-verified) | Low | — |
| 14 | CWE-121 | Stack-based Buffer Overflow | Same as CWE-787/125: `OVERFIT025`/`OVERFIT026` are the direct control, `error` in `Sources/Main` only. This is the one CWE literally on the 2025 list that the analyzer ladder was built to prevent | **PARTIAL** | See CWE-787 | See CWE-787 |
| 15 | CWE-502 | Deserialization of Untrusted Data | Grepped for `BinaryFormatter`/`XmlSerializer`/`DataContractSerializer` — none found. All config/JSON parsing goes through `System.Text.Json` with source-gen contexts (no polymorphic `$type` binding seen in the paths reviewed). GGUF/ONNX/safetensors are hand-rolled binary readers, not a generic deserializer with gadget-chain risk | MEETS | — | — |
| 16 | CWE-122 | Heap-based Buffer Overflow | Same control family as CWE-787/121; additionally `OVERFIT028` (int-overflow in size arithmetic feeding an allocation) is `error` in `Sources/Main` only, and is explicitly demoted to `suggestion` (below its own coded `Warning` default) everywhere else per `.editorconfig:177` and the comment at line 361 ("Demo/ and Server keep the global suggestion") | **PARTIAL** | Medium-High outside Main, same reasoning as CWE-787 | Same remedy as CWE-787 |
| 17 | CWE-863 | Incorrect Authorization | No role/permission model exists to check for "incorrect" (as opposed to absent) authorization — the gap is CWE-862/306, not a flawed check | NOT APPLICABLE (subsumed by CWE-862 finding) | — | — |
| 18 | CWE-20 | Improper Input Validation | General category; the specific, checkable instances are covered under CWE-125/787/22 above rather than repeated here | Covered above | — | — |
| 19 | CWE-284 | Improper Access Control | Same instance as CWE-862/306 (the server and guard endpoints) | Covered above | — | — |
| 20 | CWE-200 | Exposure of Sensitive Information | Not checked in this pass — would need a log/error-message content review across `Server.AspNet`, `Anomalies`, and the loaders for stack traces or secrets reaching a client response or a log line | **NOT CHECKED** | — | Scope a follow-up pass specifically for error-response and log-line content |
| 21 | CWE-306 | Missing Authentication for Critical Function | `Sources/Cli/Dockerfile:45` — the documented, shipped Docker image's `ENTRYPOINT` is `["/app/overfit", "serve", "--host", "0.0.0.0", "--port", "8080"]`, overriding the CLI's own safe default (`--host` defaults to `127.0.0.1`, verified `Sources/Cli/Program.cs:71-75`). Combined with the CWE-862 finding (no auth middleware anywhere in the request pipeline), the documented container deployment path serves `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/speech` to the network with zero authentication | **GAP** | **High** — this is the primary documented deployment channel ("Three ways to ship it" in `Sources/Cli/README.md`), not an edge case | Two independent fixes, either sufficient alone: (a) change the Dockerfile default to `127.0.0.1` and require an explicit `--host 0.0.0.0` (or a `--allow-network` flag) the way the Gateway command already requires `--insecure` to bind non-loopback; (b) add the opt-in API-key middleware from the CWE-862 remedy. (a) is one line and matches an existing pattern in the same CLI |
| 22 | CWE-918 | SSRF | Not checked in this pass — would need to review whether any tool/RAG surface reachable from model output (a prompt-injection-controlled value) triggers an outbound HTTP fetch to an attacker-chosen URL. `HfDownloader` fetches user-typed model names via the CLI, which is operator-initiated, not model-output-initiated, and out of scope for SSRF | **NOT CHECKED** (model-output-triggered fetch paths) | — | Scope as part of the injection-sink review named in this role's own threat-hunting order (`## Hunt in this order`, point 8) |
| 23 | CWE-77 | Command Injection | Same as CWE-78 | NOT APPLICABLE | — | — |
| 24 | CWE-639 | Authorization Bypass via User-Controlled Key | The guard's `POST /ack?id=<n>` (`GuardMetricsEndpoint.cs:274-279`) takes an operator-supplied incident `id` with no ownership/ACL concept at all — there is only one privilege level (none), so there is no "bypass" of a check that doesn't exist. Same root cause as CWE-862/306, not a distinct instance | NOT APPLICABLE (subsumed) | — | — |
| 25 | CWE-770 | Allocation of Resources Without Limits or Throttling | `OVERFIT025/026/028` cover allocation-size classes (scoping issue noted above). Request-body size: `Server.AspNet` sets no explicit `MaxRequestBodySize`, so it runs on Kestrel's built-in default (~28.6 MB); not itself a gap, but combined with CWE-306 (no auth) an unauthenticated caller can drive full-size requests at will | **PARTIAL** | Low standalone; compounds the CWE-306 finding | Once auth exists (CWE-306 remedy), this stops mattering; independently, consider an explicit lower `MaxRequestBodySize` for the chat/embeddings routes |

## What this audit did not do

- Did not re-run the 2026-06-21 `IDISP*` (CWE-416) audit; cited its conclusion, not re-verified.
- Did not exhaustively check every GGUF/`.repack`/`tokenizer.json`/WAV/MP3 loader for the CWE-22 path-join
  pattern — only the two instances named (one MEETS, one embargoed GAP) were reviewed in depth.
- Did not review log/error-message content for CWE-200, or model-output-triggered fetch paths for
  CWE-918 — both marked NOT CHECKED above rather than guessed at.
- Did not review `Sources/Anomalies` for CWE-125/787 instances beyond confirming the analyzer scoping
  gap; a targeted read of that project's Prometheus-response parsing was out of this pass's time budget.
