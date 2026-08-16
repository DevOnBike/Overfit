# Applicable public security standards — shortlist

Selection only, not an audit. Written 2026-08-10 to answer "which standards apply, so we can audit
against them later." Versions verified via web search on the same date; re-check before citing, this
space moves fast (three of the items below were revised in 2026 alone).

## What was scoped against (verified in the tree, not assumed)

- **Loaders for untrusted third-party files**: GGUF, ONNX with a hand-rolled protobuf parser
  (`Sources/Main/Onnx/OnnxProtoParser.cs`, no `Google.Protobuf`) plus external `.data` sidecars
  (`OnnxExternalData.cs`), safetensors, `.bin`, `tokenizer.json`. Largest surface, least covered by any
  standard below.
- **Serving surfaces**: `Sources/Server.AspNet` (OpenAI-compatible HTTP API, `OverfitOpenAiApi.cs`),
  `Sources/Server` (shared OpenAI types), `Sources/Mcp` (stdio JSON-RPC, `McpServer.cs`). Also the
  anomaly-guard's Prometheus-facing HTTP endpoint (`Sources/Cli/GuardMetricsEndpoint.cs`).
- **Build/release**: 6 GitHub Actions workflows, 1 of 27 `uses:` refs SHA-pinned, no SBOM/signing/
  provenance — see `supply_chain_snapshot.md` in agent memory for the full breakdown.
- **Existing guards**: `NuGetAudit`, the `OVERFIT0xx` Roslyn analyzer ladder (confirmed:
  `RecursionAnalyzer.cs` = OVERFIT022, `UnboundedLoopAnalyzer.cs` = OVERFIT023,
  `StackallocSizeAnalyzer.cs` = OVERFIT025/026), Checkmarx One + DevSkim in CI, SourceLink.
- **Commercial on-prem components**: out of scope for this document; they are audited separately in
  material that does not become public.

## Shortlist

| Standard | Version / date | Governs | Does NOT cover |
|---|---|---|---|
| **CWE Top 25** | 2025 list, published Dec 2025 by CISA/MITRE from 39,080 CVEs | Weakness taxonomy — CWE-190 (int overflow), CWE-789 (uncontrolled memory allocation), CWE-674 (uncontrolled recursion), CWE-400 (uncontrolled resource consumption) map directly onto what OVERFIT022/023/025-030 already enforce | Not AI-specific; says nothing about model-format semantics or prompt/output risk |
| **MITRE ATLAS** | v5.4.0, Feb 2026 (16 tactics, 84 techniques, agent-focused additions through late 2025/early 2026) | Threat *catalog*, not a checklist — `AML.T0010` (ML Supply Chain Compromise via a poisoned model artefact) is the exact shape of the loader risk | Not auditable pass/fail; used to write the threat model's attacker language, not to certify anything |
| **NIST SP 800-218A** | Published 26 Jul 2024, still current | SSDF profile specific to generative-AI/foundation-model software producers — closest federal document to "how do you secure a foundation-model-loading library's dev lifecycle" | Process-level (data provenance, eval, training-time practices); says nothing about parser-level memory safety |
| **SLSA** | v1.1 stable (v1.2 in draft as of this check) | Build integrity levels L0-L3 (hosted build, signed provenance, isolated build platform) | Says nothing about application logic; purely supply-chain, and Overfit is at L0 today (no signed provenance) |
| **OpenSSF Scorecard** | v5.5.0, 23 Apr 2026 | Automated, runnable checker — branch protection, pinned dependencies (the exact 1/27 gap already measured), SAST presence, Dangerous-Workflow — produces a public numeric score | Not a manual-review framework; only as good as what it can statically detect |
| **OWASP Top 10 for LLM Applications** | 2026 edition, published 4 Aug 2026 at Black Hat USA (first edition weighting real incident data at 25%) | Model-as-component risk: prompt injection, sensitive-info disclosure, output handling, RAG-source poisoning — relevant to the chat/RAG/MCP surfaces | Explicitly does not cover the model-*file* attack surface (a malicious GGUF/ONNX is not "prompt injection"); also does not cover agentic/tool-use risk, which is a separate OWASP document |
| **OWASP Top 10 for Agentic Applications** | Published Dec 2025 | Goal hijacking, tool misuse, cascading multi-agent failures — relevant if/when `AL-8` (MCP host role) ships | Not yet load-bearing: Overfit today is MCP *server*, not *host*; this becomes relevant only once `AL-8` ships |
| **OWASP Machine Learning Security Top 10** | Still a working draft, no finalized numbered version (confirmed via GitHub project page) | Conceptually the closest fit for the loader surface (its "Model Deserialization Attack" category is exactly the GGUF/ONNX/safetensors risk) | Cite with an explicit caveat — this is not a ratified standard and citing a version number would be fabricating one |
| **CycloneDX ML-BOM** | Spec v1.7 (current), standardized as ECMA-424 | Machine-readable model/dataset bill-of-materials; the EU AI Act Art. 11 technical-documentation requirement (in force 2 Aug 2026 for high-risk systems) maps onto its fields | Not a security control by itself — it's a disclosure format; needs something to populate it (model provenance, hash) which Overfit does not currently emit |
| **OWASP ASVS** | v5.0.0, 30 May 2025 (~350 requirements, 17 chapters — first major revision in 6 years) | HTTP-surface only: `Server.AspNet`, the guard's `:9469` endpoint. Its authentication chapter (V4/V7 in the 5.0 numbering) is exactly what flags an unauthenticated write endpoint | Zero relevance to the library or the parsers — do not apply it there |
| **NIST AI RMF 1.0 + AI 600-1 (GenAI Profile)** | AI RMF: Jan 2023; AI 600-1: 26 Jul 2024, still current | Org-level GOVERN/MAP/MEASURE/MANAGE structure; useful for the "published position on untrusted models" narrative in `SECURITY.md` | Voluntary, no pass/fail; not code-auditable |
| **ISO/IEC 42001:2023** | Published Dec 2023; EN adoption approved 13 Mar 2026 | Certifiable AI management system (policies, governance, org structure) — covers ~70% of EU AI Act high-risk documentation by one industry estimate | Heavy: needs a management system and an accredited external audit, not something this review can assess. Business decision (enterprise sales credibility), not an engineering one |

## Fit: engine/parser vs. HTTP-surface-only

**Fits the parser/engine (the actual largest surface)**: CWE Top 25, MITRE ATLAS, SLSA, OpenSSF
Scorecard, NIST SP 800-218A, OWASP ML Top 10 (informally — it's a draft), CycloneDX ML-BOM (as a
target artefact, not a control).

**Fits the HTTP/MCP surface only, do not extend to the library**: OWASP ASVS, OWASP LLM Top 10 2026
(covers prompt/RAG/output risk on the chat surfaces, not the loaders), OWASP Agentic Top 10 (not yet
load-bearing).

**Org-level, not code-level**: NIST AI RMF/600-1, ISO/IEC 42001.

## Ranked recommendation

1. **CWE Top 25, mapped against the existing `OVERFIT0xx` ladder and the loader code.** Free to start —
   it's a taxonomy, not a process; the mapping is mostly already implicit in the analyzer rules and just
   needs writing down. No new artefact required.
2. **OpenSSF Scorecard, run via the official Action.** Cheap (one workflow file, no secrets needed for
   the read-only checks), produces a public score, and directly measures the one concrete gap already
   found (`1/27` actions pinned). This is a **workflow-file proposal**, not something I can add myself —
   draft goes to the user per the git/GitHub boundary.
3. **SLSA, targeted at L1 first** (build provenance via `actions/attest-build-provenance`, no isolated
   builder required). Natural follow-on once Scorecard is in and gives a numeric "before" score.

**Not recommended to start with**: ISO/IEC 42001 (needs an accredited audit and a management system,
business decision not an engineering one) and full NIST AI RMF (voluntary, no artefact to fail or pass —
better mined for prose in `SECURITY.md`'s untrusted-model section than "audited against").

## What is already satisfied, incidentally (credit at audit time)

| Guard already in the repo | Standard clause it lands on |
|---|---|
| `NuGetAudit=true`, `NU1901-1904` as errors (`Directory.Build.props`) | SSDF PW.4 (verify third-party components); Scorecard's "Vulnerabilities" check |
| `OVERFIT022` (unbounded recursion), `OVERFIT023` (unbounded loop) | CWE-674, CWE-835 |
| `OVERFIT025`/`OVERFIT026` (`stackalloc` >512B, variable-length) | CWE-121/CWE-789 (stack/uncontrolled allocation) |
| `OVERFIT028`-class (int overflow in size arithmetic) | CWE-190 |
| Checkmarx One + DevSkim in CI, SARIF upload | SSDF PW.7/PW.8 (static analysis); a partial Scorecard "SAST" check |
| `Microsoft.SourceLink.GitHub` + `ContinuousIntegrationBuild`/`SourceRevisionId` | SLSA provenance building block (source mapping), though not full provenance |
| `checkmarx/ast-github-action` pinned to a full commit SHA (the one of 27) | Scorecard's "Pinned-Dependencies" check, partially |

## One open item a standard would flag (internal note, not for publication)

`SEC-1` in `docs/TASKS.md` (unauthenticated `POST /ack` on the guard's metrics port) is exactly what
OWASP ASVS's authentication chapter checks for, and is already documented — with its mitigation and
residual risk — in `docs/security/aiops-anomaly-guard-trust.md`. No new disclosure here; noted only so
a future ASVS-mapped audit does not "rediscover" it as new.
