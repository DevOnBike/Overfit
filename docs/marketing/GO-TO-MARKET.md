# Overfit — Go-to-Market Plan

From "public repo" to first paid revenue. Horizon: one quarter (≈12 weeks).

---

## The model — two revenue streams

| Stream | What it is | Conversion | Reality for a solo operator |
|--------|-----------|------------|------------------------------|
| **Product** | Open-source engine, AGPLv3 → commercial license. Tiers $4.8k / $18k / $48k. | **Self-identifying** — a closed-source .NET shop *must* buy the commercial license. | Low volume early; the long tail. Its real early job is **credibility + lead funnel**. |
| **Services** | Implementation & consulting around Overfit — deployment, fine-tuning, architecture. | Inbound from launches + LinkedIn Services + partners. | **Where the money is early.** One engagement = $5k–50k. This funds the project. |

**The strategic point:** the product earns trust and generates leads; the *services* monetize them. Services revenue lands first; product-license revenue grows later with adoption. Plan accordingly — do not wait for license sales to "prove" the market.

---

## Positioning — one wedge

> **Run and fine-tune LLMs in pure C# — no Python, no data egress.**

Lead with the **regulated / no-egress** wedge (healthcare, finance, public sector, compliance-bound .NET shops). It names a buyer with a budget and a real pain. "Pure-.NET AI" is the broad, secondary message — not the lead.

---

## The motion

Content + coordinated launches → awareness → GitHub / landing → inbound. Inbound splits two ways: **developers** (→ some convert to a product license) and **enterprises / regulated teams** (→ a services engagement). One person, so: no high-touch sales, **productized** fixed-scope services, leveraged channels (launches, self-identifying license conversion).

---

## Phase 0 — Foundations (Week 1)

The "is it launchable AND can it monetize" checklist.

**Product side**
- [ ] Deploy the fixed `index.html` → overfit-ml.com live with the new positioning. *(file ready)*
- [ ] README: add "Why not X?" comparison + a clear **Licensing** section (the AGPL → commercial trigger, stated plainly).
- [ ] Sanity: CI green, `dotnet run` demo works, NuGet package current.

**Services side**
- [ ] LinkedIn profile aligned — headline tells one story with the Services section; Services section filled (see `landing-copy-fixes.md` / LinkedIn services copy).
- [ ] `COMMERCIAL.md` is a *real offer* — what each tier includes, how to buy/contact.
- [ ] Three productized service packages defined (see below).
- [ ] Contact intake works (landing form → inbox).

## Phase 1 — Anchor content (Week 1–2)

- [ ] **Technical blog post** — the deep-dive ("zero-allocation LLM inference in pure C#"). The durable asset every launch post links to.
- [ ] LinkedIn business article published (3 drafts already exist: `linkedin-*.md`).

## Phase 2 — Launch (Week 2–3, one coordinated window)

- [ ] **Show HN** + **r/dotnet** in a tight window; posts link to the blog post.
- [ ] LinkedIn business article same week.
- [ ] Submit to "This Week in .NET" newsletter / .NET Discord.
- [ ] Then 1–2 weeks: respond to **every** comment and GitHub issue, fast. For a new project, responsiveness *is* the marketing.

## Phase 3 — Convert the attention (Week 3–6)

- [ ] Triage inbound: product-license inquiry vs services inquiry.
- [ ] Make license-buying frictionless (no dead ends from the landing).
- [ ] **Land the first paid services engagement** — even a small one. Start with the entry offer (Architecture Review).
- [ ] First engagement → **first case study** (anonymized if needed). This is the single most valuable asset you do not have yet.

## Phase 4 — Second beat + channel (Week 6–12)

- [ ] Build **embedding-model support** (ROADMAP priority #1) → Overfit does RAG end-to-end → **a second launch** ("Overfit now does RAG"). A second shot at attention.
- [ ] **Consultancy / partner outreach** — now from strength: a public launch and a case study behind you. One consultancy = a channel to many regulated clients.

---

## The services — productized (entry → core → retainer)

A solo operator cannot sell open-ended hourly consulting. Fixed scope, fixed price.

| Package | Scope | Indicative price | Role |
|---------|-------|------------------|------|
| **AI Architecture Review** | 1–2 weeks. Review the client's private/self-hosted AI options; deliver a concrete architecture + plan. | ~$3–5k | **Entry offer / tripwire.** Low-commitment "yes". Foot in the door. |
| **Private LLM Proof-of-Concept** | 3–4 weeks. Stand up Overfit on the client's data + infrastructure; deliver a working pilot. | ~$15–25k | **Core offer.** Proves the use case → becomes the case study. |
| **Deployment & Support** | Production deployment + retainer / SLA. Maps to the Production ($18k) and Enterprise ($48k) license tiers. | $18k–48k / yr | **Recurring revenue.** |

Sell the ladder: Architecture Review is the easy first cheque; it leads to the POC; the POC leads to Deployment & Support. Prices above are starting points — set your own.

---

## Always-on (through every phase)

- **Content cadence** — one post / article every 1–2 weeks keeps the project alive between launches (dev.to, LinkedIn, blog).
- **Fast response** — to issues, comments, inquiries. Non-negotiable for a new project.

## What "in the market" means — the metric that matters

Not GitHub stars. The real signals, in order: first inbound services inquiry → **first paid engagement** → first case study → first commercial license sold. The first 12 weeks succeed if you have **one paid engagement and one case study**. Everything else is a leading indicator.

## Honest constraints

- One person. The plan is leveraged on purpose — launches and self-identifying conversion do the work that a sales team would.
- Services must stay productized / fixed-scope, or they eat all your time.
- Services money arrives before license money. Do not measure early success by license sales.
- 12 weeks gets you *into* the market with first revenue — not to "scale". Scaling is the next quarter.

## Who does what

- **You:** deploy the landing, run the launch posts, sell and close engagements, deliver them.
- **Claude Code can produce:** README "Why not X" + Licensing section, the technical blog post, the launch copy (Show HN / r/dotnet), the three service-package one-pagers, and the embedding-model feature (code).

---

## Immediate next action

Phase 0 → README "Why not X?" + Licensing section. It is the one remaining launch-blocker that is closed with code, and it is where a Hacker News / r/dotnet visitor actually lands.
