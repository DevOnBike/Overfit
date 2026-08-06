# `docs/specs` — plan files

One file per change, named `<slug>-plan.md`. It is written before implementation and is **the single place
that describes what is being built and why**.

## Who writes what

A plan file has two authors and they do not overwrite each other.

| | writes | owns |
|---|---|---|
| `overfit-analyst` | first | problem, goal, users, success metric, business rules, scenarios, acceptance criteria, functional scope, value against cost |
| `overfit-architect` | appends after | system context, boundaries and responsibilities, quality requirements as parameters, technical risk and its spikes, operability, deployment |

**Neither rewrites the other's sections.** A disagreement is recorded as a numbered finding inside the
author's own section, so that it gets resolved rather than erased. Both end with `BLOCKING QUESTIONS`, split
into questions for the client and questions for the technical reader.

## The handshake — a plan is not ready until it is signed

Implementation is **gated** on the architecture half being present. `overfit-developer` refuses to write
source from a plan whose architecture sections are missing, and treats their absence as a blocker rather than
as permission.

The reason is specific: **an unmade decision does not stay unmade.** It gets made by whoever writes the code
first, for whatever reason was convenient at the time — and by the time anyone notices, it is load-bearing.
Asking costs one message; unpicking a shipped coin-toss costs far more.

A plan is closed in one of two ways, both written into the file:

1. **The architecture sections**, answering execution path, allocation policy, AOT reach, ownership,
   assembly and dependency direction, public API surface, quality parameters and threading; or
2. **A short sign-off**, when the change genuinely carries nothing beyond the standing rules — but still
   stating execution path, AOT reachability and allocation policy, because a developer cannot infer those and
   must not guess them.

**Leaving the sections out is not the second option.** Silence and "nothing to add" are different statements
and only one of them is a decision.

Small changes are exempt by their nature — a comment fix, a rename, a test-only change. Anything touching
`Sources/Main`, crossing an assembly boundary, changing public API, touching a hot path or a file parser, or
adding a dependency is **not** small, however few lines it takes.

## Why one file and not two

A plan and a separate architecture note that disagree are worse than either alone, and nothing makes them
agree once they have separate authors and separate lifecycles. This repository has already paid for that with
a roadmap and a changelog that contradicted each other.

The same rule applies over time: **when a request revisits an existing plan, update that file** rather than
starting a second one.

## What a plan is not

Not a design document that fixes every detail — it sets boundaries, names the verification oracle and the
success metric, and leaves the rest to whoever implements it. Not a status tracker either. And not a place
for a performance target without the benchmark that would settle it.

## Lifecycle

A plan stays after the work ships. It is the record of *why* the change had the shape it did, which is
information the diff does not carry. If a decision in it turns out to be wrong, say so in the file — deleting
the reasoning is how the same mistake gets made twice.

Cross-cutting decisions that are hard to reverse do not live here; they go to [`../adr`](../adr).
