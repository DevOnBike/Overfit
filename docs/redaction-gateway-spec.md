# Redaction Gateway — Spec & Decisions

**One-liner:** an OpenAI-compatible proxy that sits in front of an upstream LLM, redacts outbound PII/secrets
from each request before it leaves the box, optionally restores them on the response, and audits every action —
*"change one base URL and your prompts stop leaking secrets."* On-prem, pure-managed, Native-AOT.

## Status

- **Phase 1 — redaction engine — DONE** (`Sources/Main/Redaction/`): `Redactor` (Redact/Restore, deterministic
  overlap resolution, indexed placeholders), `RedactionRule` / `RedactionMatch` / `RedactionResult`,
  `DefaultRedactionRules` (8 starter rules via `[GeneratedRegex]`), `RedactionAuditRecord` (category counts only —
  never sensitive values) + `IRedactionAuditSink`. 5 unit tests green, build clean, AOT-friendly.
- **Phase 2 — proxy + production policy — PENDING the decisions below.** The engine is structurally stable; the
  spec configures *rules + policy*, it does not change the engine.

## Architecture (data flow)

```
client (OpenAI SDK, base_url → gateway)
   │  POST /v1/chat/completions
   ▼
[ gateway ]  parse → REDACT messages (Redactor) → audit(counts, action)
   │            │ policy: redact / BLOCK / alert / allow (per category)
   │            ▼ (request-scoped placeholder↔original map, in RAM only)
   │  forward (gateway injects upstream API key) ──► upstream LLM (OpenAI / Anthropic / local overfit serve)
   │                                                      │
   ◄──────────────── RESTORE placeholders (optional) ◄────┘  response
```

---

## Decisions

Each item: the question, the realistic options, a **PROPOSED DEFAULT (MVP)**, and a blank for your call.
Confirm the defaults you like; override the rest.

### 1. Detection scope — what to detect
- **Structured PII** (regex + validator, cheap/deterministic): email, phone (locale!), cards (+ Luhn), IBAN,
  **PL: PESEL (+ checksum), NIP, REGON, ID-card no.**, US SSN.
- **Free-text PII** (names, addresses, DOB): regex can't do this — needs NER (a model). In scope now or Phase 3?
- **Secrets**: vendor keys (OpenAI `sk-`, AWS `AKIA`, GCP, Azure, Stripe, GitHub `ghp_`, Slack `xox*`),
  **generic high-entropy** tokens (Shannon entropy > threshold), JWT, PEM private keys, passwords in
  connection-strings/URLs, bearer tokens.
- **Infra/internal**: internal hostnames/domains, private IP (RFC1918) vs all IPv4, internal URLs, file paths,
  employee IDs, project codenames.
- **Allowlist**: values to never redact (own public domain, known-safe) — guards against over-redaction.

> **PROPOSED DEFAULT:** structured PII + secrets (vendor + entropy) + private IPs, all with validators to cut
> false positives; allowlist supported; free-text NER deferred to Phase 3.
> **Your call (must-have categories, esp. PL PII + your secret shapes): ______**

### 2. Policy per category — what to do on a hit
- **Actions:** `REDACT` (replace + forward) / `BLOCK` (refuse whole request, 4xx) / `ALERT` (forward unchanged,
  log/notify) / `ALLOW` (ignore).
- **Direction:** apply on the **request** (outbound) and/or the **response** (model may echo a secret / leak).
- **Fail mode:** if the redactor errors/times out → forward anyway (**fail-open**, availability) or refuse
  (**fail-closed**, safety)?

> **PROPOSED DEFAULT:** category→action map; unknown→REDACT; hard categories (private key, password)→BLOCK;
> redact both directions; **fail-closed** (this is a security gateway).
> **Your call (action per category + fail mode): ______**

### 3. Upstream + auth — where it forwards and who may call
- **Upstream:** single fixed upstream vs per-model routing (model name → OpenAI / Anthropic / local).
- **Upstream auth — high-value:** gateway holds the real API key; clients use a gateway key → **clients never see
  the real secret** (gateway = key vault + egress firewall in one). Or pass-through client auth.
- **Client auth:** per-tenant API keys / mTLS / none (trusted network).
- **Tenancy:** multi-tenant (per-tenant policy + audit + keys) vs single.
- **Streaming (SSE):** redacting a token stream is hard (a hit can straddle a chunk boundary → must buffer).

> **PROPOSED DEFAULT:** gateway-holds-upstream-key (vault); one configurable upstream first, per-model routing
> later; client API-key auth; single-tenant first; **non-streaming first** (streaming is a flagged follow-up).
> **Your call (vault yes/no, streaming required now?, multi-tenant?): ______**

### 4. Response un-redaction — restore originals or not
- `restore`: client gets a coherent answer naming the real entities (placeholders mapped back). Risk: the model's
  claims *about* a redacted entity may be wrong (it never saw it). Great for summarize/translate/rewrite.
- `keep-redacted`: client sees placeholders (safest, less useful).
- **Security:** restore means holding the original↔placeholder map for the request lifetime — must be
  **request-scoped, in RAM, never logged/persisted**.

> **PROPOSED DEFAULT:** `restore` (we have `Redactor.Restore`), request-scoped map, never persisted; configurable
> per route/category.
> **Your call (restore vs keep-redacted): ______**

### 5. Audit sink — where and what to log
- **Sink:** append-only JSONL file / syslog-SIEM (Splunk/Elastic) / DB / webhook / stdout.
- **Content:** requestId, timestamp, tenant, **per-category counts, action taken**, upstream, latency — **never
  the values**. Optional salted hash of values (correlation without exposure).
- **Integrity:** retention/rotation, tamper-evidence (append-only, hash chain?).

> **PROPOSED DEFAULT:** JSONL append-only (simple, AOT, zero deps) via the existing `IRedactionAuditSink`; record
> action + counts + correlation id; hash-chain optional later.
> **Your call (JSONL vs SIEM, salted-hash needed?): ______**

### 6. Surface / wiring — how it runs
- **Form:** `overfit gateway` standalone command vs a flag on `overfit serve` vs a library middleware.
- **Reuse:** `OverfitOpenAiServer` (HttpListener + OpenAI DTOs + AOT JSON ctx) in a forward/proxy mode.
- **Endpoints:** `/v1/chat/completions` passthrough (the pitch); also `/v1/embeddings`, `/v1/completions`?
- **Field scope:** redact only `message.content`, or also system prompt, tool-call args, function definitions?
- **Config:** JSON/YAML file for rules+policy+upstream, env vars, or both.

> **PROPOSED DEFAULT:** `overfit gateway --upstream <url> --config policy.json`, standalone, reusing the server;
> OpenAI-compatible passthrough; redact content + tool-call args + system prompt; JSON config file.
> **Your call (standalone command vs serve flag, which endpoints, field scope): ______**

---

## Cross-cutting (also worth pinning)
- **Streaming SSE** — hard (redaction across chunk boundaries); call it out as a separate milestone.
- **Non-chat endpoints** (embeddings/completions) — in scope?
- **Latency** — regex is microseconds; entropy detection is cheap. Not a concern.
- **Over-redaction** — allowlist + validators are the key to keeping responses useful.

## Phased plan
1. **Phase 2a — proxy skeleton:** `overfit gateway` reusing `OverfitOpenAiServer`; forward `/v1/chat/completions`
   to a configured upstream (gateway-injected key); redact request content via `Redactor`; restore on response;
   JSONL audit. Non-streaming. Defaults above.
2. **Phase 2b — policy engine:** category→action map (redact/block/alert/allow), fail-closed, config file, response
   scanning.
3. **Phase 2c — production rules:** the real pattern set + validators (PL PII, vendor secrets, entropy), allowlist.
4. **Phase 3 (follow-ups):** streaming SSE redaction, free-text NER, multi-tenant, SIEM sink, per-model routing.
