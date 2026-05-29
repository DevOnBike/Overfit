# Getting Support

Overfit has two support paths — free open-source and paid commercial. Pick the one that matches the cost of you being stuck.

## Free / open-source

| Channel | Use it for |
|---|---|
| [GitHub Issues](https://github.com/DevOnBike/Overfit/issues) | Bug reports, feature requests, regression reports |
| [GitHub Discussions](https://github.com/DevOnBike/Overfit/discussions) | Usage questions, design questions, "how do I…?" |
| Source code | Authoritative answer to "what does this actually do" |

Free-tier triage is best-effort. The maintainer responds when there's time — typically within a week for clear bug reports, sometimes longer for design discussions. **There is no response-time guarantee on the free path.** If you need one, use the commercial path.

When filing an issue, please include:

- Overfit version (NuGet version or commit SHA if building from source).
- .NET runtime version (`dotnet --version`).
- OS / CPU architecture.
- Minimal reproducer or failing test.
- For LLM issues: which model, which quantisation, which loader.

## Paid / commercial

When you need response-time guarantees, fixed-scope engagements, or a named owner accountable for delivery, see [`COMMERCIAL.md`](COMMERCIAL.md). Three paths:

### Service packages — fixed scope, fixed price

- **Package 1 — Private .NET RAG/Agent PoC** (2–3 weeks) — local LLM + RAG + tool calling, deployed in your infrastructure, no data egress.
- **Package 2 — Python / ONNX-sidecar replacement** (2 weeks) — collapse your model-server tier into the .NET process.
- **Package 3 — Zero-GC inference audit** (1–2 weeks) — profile your .NET inference hot path, identify allocation/GC sources, AOT-clean verification.

### Standalone commercial license

Per-product, perpetual. Use this when you don't need integration help — just the legal clearance to embed in a closed-source product.

### Support retainer

Monthly tiers with response-time SLAs:

| Tier | Response SLA (business hours, CET) | Hours/month included |
|---|---|---|
| Standard | Next business day | 4 h |
| Business | Same business day | 12 h |
| Priority | Within 4 business hours | 32 h |

Full tier scope, what's covered / not covered, and how to buy are in [`COMMERCIAL.md`](COMMERCIAL.md).

**Contact:** `devonbike@gmail.com` — subject line `Overfit Support` (operational) or `Overfit Commercial Inquiry` (pre-sales).

## Security issues

Do not file security vulnerabilities as public issues. See [`SECURITY.md`](SECURITY.md) for the private reporting channel and response timeline.

## What's not supported, even commercially

To be precise upfront, the maintainer does not:

- Build your end-user product features (Overfit is the runtime; your product is your product).
- Provide generic .NET / ASP.NET / DevOps consulting unrelated to Overfit.
- Run a 24/7 NOC or pager rotation. Response SLAs are business-hours, CET.
- Sub-contract to a multi-engineer team. The author does the work personally.

If you need any of the above, you're looking for a different vendor — and that's fine to know early.
