# Bug hunts

One file per run of the `overfit-find-bugs-game` agent, written by the agent itself.

**Naming:** `[module]-[yyyy-MM-dd-HHmm]-bugs-game-findings.md`, timestamp in UTC — for example
`anomalies-2026-08-01-2158-bugs-game-findings.md`.

## Why these are files rather than transcript

A hunt that exists only in a conversation is a hunt nobody acts on next week. The defects this agent looks
for — a failure that produces no signal, a claim the code does not back, a bound that discards work instead
of truncating it — are precisely the ones that get rediscovered months later, during the incident they
caused. A dated file per scope also makes it possible to ask the useful question: *did this subsystem get
better or worse since the last read-through?*

## How to read a score

The game is two points per defect, played to 21. **A score below 21 is a result, not a failure.** It says
the code was in better shape than the game assumed, and it is worth far more than a padded list — every
entry in these files is meant to be worth acting on, and one that is not devalues the rest.

Each file therefore ends with what was reviewed and found clean. That section is what stops the next
reviewer repeating the work, and it is the only honest way to interpret a short list.

## Related

- `docs/silence-review.md` — the method behind the silent-failure half of the checklist, and the evidence
  that produced it.
- `docs/aiops/aiops-repair-plan.md` — the shape a findings file takes once the caller decides to schedule the
  work: what breaks, where, how to fix it, and what test would have caught it.
