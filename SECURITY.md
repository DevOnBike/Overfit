# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| `10.x` (latest minor) | Yes |
| `< 10.x` | No |

Security fixes ship in the next patch release on the latest minor. Commercial-license customers under active annual maintenance get backports to the version they are pinned to, for the duration of that maintenance contract (see [`COMMERCIAL.md`](COMMERCIAL.md)).

## Reporting a Vulnerability

**Do not file a public GitHub issue for security vulnerabilities.** Public disclosure before a fix is available puts other users at risk.

Email security reports to **devonbike@gmail.com** with subject `Overfit Security`. Include:

- Affected version(s) and platform.
- Steps to reproduce, ideally as a minimal repro.
- Impact assessment if you have one.
- Whether you intend to disclose publicly and on what timeline.

PGP key on request before sending sensitive details.

## Response timeline

| Stage | Free / OSS | Business retainer | Priority retainer |
|---|---|---|---|
| First human acknowledgement | 2 business days | 1 business day | 4 business hours |
| Triage decision (severity + fix plan) | 5 business days | 3 business days | 1 business day |
| Backport to pinned version | n/a | yes, while maintenance is active | yes, while maintenance is active |

Business hours = Mon–Fri 09:00–17:00 Europe/Warsaw (CET / CEST). See [`COMMERCIAL.md`](COMMERCIAL.md) for retainer tier definitions.

## Coordinated disclosure

Coordinated disclosure is preferred. Default window: **90 days from acknowledgement to public disclosure**, shortened with mutual agreement once a fix has been released. Credit in the changelog and the GitHub Security Advisory is offered unless the reporter prefers anonymity.

## Out of scope

The following are not considered security vulnerabilities:

- Issues only reachable by deliberately misusing AOT-banned APIs (see `Sources/Main/BannedSymbols.txt`).
- Vulnerabilities in the model weights you load. Model files are data Overfit reads, not code Overfit owns; report those to the model author.
- Vulnerabilities in third-party NuGet dependencies — report to the upstream maintainer first, then to Overfit only if the exposure surface in Overfit is non-trivial.
- Performance / denial-of-service reports without proof of cross-tenant or cross-process impact (Overfit is an in-process library; the host application owns its trust boundary).
- Findings from static analysers without a working repro (we accept SARIF uploads but treat them as triage hints, not confirmed vulnerabilities).
