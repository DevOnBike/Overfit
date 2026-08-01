# `Redaction` — detecting and replacing sensitive values in text

Rule-driven detection of personal data and secrets, with reversible replacement. `Redactor` applies a
`RedactionPolicy` built from `RedactionRule` sets — `DefaultRedactionRules`, `PolishRedactionRules`
(PESEL, NIP, REGON, Polish phone and address shapes), `SecretRedactionRules` (keys, tokens,
connection strings) — and returns a `RedactionResult` carrying the matches and the mapping needed to
restore them.

`StreamingResponseScanner` and `StreamingRestorer` do the same over a token stream, which is what an
SSE response needs: the decision has to be made on partial text without buffering the whole answer,
and a value must not be split across a chunk boundary in a way that hides it.

`RedactionValidators` implements checksum validation (a candidate that fails its check digit is not a
PESEL), which is the difference between a rule and a false-positive generator. `IRedactionAuditSink`
and `RedactionAuditRecord` record what was replaced and why, since "we redact" is a claim someone will
eventually have to evidence.

## Scope

This directory is the detection and replacement engine. The proxy that uses it, its configuration and
its operational documentation live with that feature and its CLI help. Keep discussion of the
deployment story there rather than in general-purpose documentation.
