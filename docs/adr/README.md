# `docs/adr` — architecture decision records

`NNNN-<slug>.md`, numbered sequentially, never renumbered. An ADR records **why** a decision was taken and
what it costs to live with — the code already says what the system does.

## Only for decisions that are hard to reverse

Most decisions do not need one. In this repository the list that does is short and specific:

- what becomes **public API** in the shipped package — it cannot be withdrawn;
- **which assembly** a capability lives in, once its types are public;
- whether something is **reachable from the AOT smoketest**, because the no-reflection constraint then
  propagates to everything it calls;
- the **on-disk or on-wire format** of anything persisted or exchanged;
- **which side of the open/commercial boundary** a capability falls on;
- a **dependency added to `Main`**, since every consumer inherits it.

Naming, file layout, whether to extract a helper — those belong to whoever writes the code.

## Format

**Context** (what forced a decision) · **Forces** (requirements, constraints, risks) · **Options considered**
(real ones, not strawmen) · **Decision** · **Rationale** · **Consequences** (what it costs, what it forecloses)
· **Status** — proposed, accepted, superseded by `NNNN`, or withdrawn.

Superseded records are **kept, not deleted**. A decision that was reversed is more informative than one that
was never written down, and this repository treats a discarded experiment's reasoning as evidence worth
preserving.

## Two rules that keep these useful

**An ADR documents a decision; it does not advertise a technology.** If it reads as a case for a tool rather
than an account of a trade-off, it is not finished.

**Check whether the decision already has a home before writing one.** Design reasoning here lives in code
comments, `CLAUDE.md` and `docs/` — a duplicate that drifts is worse than a pointer. Prefer linking to the
existing explanation over restating it.

Change-specific plans live in [`../specs`](../specs).
