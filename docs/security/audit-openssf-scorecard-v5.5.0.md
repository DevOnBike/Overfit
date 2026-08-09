# Audit — OpenSSF Scorecard v5.5.0

Standard: OpenSSF Scorecard, checks as of v5.5.0 (23 Apr 2026). **Not run as a tool** — no `gh` auth in
this session (`gh api` refuses unauthenticated), so this is a manual walk of Scorecard's published check
list against the repository. Where a check needs the GitHub API (branch protection, review history,
contributor affiliation, webhooks), it is marked `NOT CHECKED` rather than guessed. Commissioned by the
team-lead 2026-08-10 against the shortlist's #2 pick.

**Verdict counts: 7 MEETS, 4 GAP, 2 PARTIAL, 1 NOT APPLICABLE, 5 NOT CHECKED.**

## Findings that matter, ranked by what an attacker reaches in this deployment

1. **Pinned-Dependencies — 1 of 28 `uses:` refs pinned to a full commit SHA** (recount, up from the
   28-total figure that was 27 at last measurement; the ratio is unchanged). Four workflows use secrets
   with real publishing power (`publish-nuget.yml` → `NUGET_API_KEY`; `docker-publish.yml` →
   `DOCKERHUB_TOKEN`; `overthink-playstore.yml` → four Android signing secrets + a Play service-account
   JSON; `checkmarx-one.yml` → Checkmarx credentials, already pinned). A floating tag on any action inside
   the first three can be repointed by whoever controls that action's repository and the next run executes
   their code holding real publishing credentials.
2. **`overthink-playstore.yml` is the highest-value single target**: it decodes an Android upload
   keystore, signs a release AAB, and hands the Play Console service-account JSON directly to a
   **third-party, unpinned action** (`r0adkll/upload-google-play@v1`) as plaintext input. This is the one
   workflow where an unpinned action receives a long-lived publishing credential as a literal argument
   rather than just an environment secret.
3. **Token-Permissions is inconsistently armed, not absent** — `docker-publish.yml`, `devskim.yml` and
   `checkmarx-one.yml` all declare explicit least-privilege `permissions:` blocks (verified by reading
   each); `ci.yml`, `publish-nuget.yml` and `overthink-playstore.yml` declare none, so they run under
   whatever the repository/organization default is (itself `NOT CHECKED` — needs Settings → Actions →
   General, unreadable without `gh` auth).
4. Signed-Releases and SBOM/provenance are absent for both shipped artefacts (NuGet package, Docker
   image) — lower urgency than #1/#2 because it doesn't hand an attacker a credential, but it's the gap
   that blocks any SLSA claim later.

## Checks

| Scorecard check | What was checked, and how | Verdict | Severity in this deployment | Remedy |
|---|---|---|---|---|
| Pinned-Dependencies | Counted every `uses:` line in all 6 workflow files (28 total) and its pin style | **GAP** | High — see ranked finding #1 | See the worked diff below; apply to all 27 floating refs, prioritizing the 4 secret-using workflows first |
| Token-Permissions | Read every workflow's `permissions:` block (top-level and job-level) | **PARTIAL** | Medium — see ranked finding #3 | Add `permissions: contents: read` at minimum to `ci.yml`, `publish-nuget.yml`, `overthink-playstore.yml` (none of the three need repo-write; `publish-nuget`/`overthink-playstore` only need the ability to push to their respective external registries, which goes through a secret, not the `GITHUB_TOKEN`) |
| Dangerous-Workflow | Grepped all workflows for `pull_request_target` (none found — `ci.yml` uses the safe `pull_request` trigger) and for `${{ github.event.* }}` / `${{ github.head_ref }}` interpolated into a `run:` shell block (none found — the only `${{ inputs.* }}` interpolation is in `workflow_dispatch`-only workflows, which only a repo collaborator with write access can trigger, not a PR author) | MEETS | — | — |
| Binary-Artifacts | `git ls-files` filtered for `.dll/.exe/.so/.dylib/.jar` — 0 tracked | MEETS | — | — |
| License | `LICENSE.md` present at repo root (AGPLv3 per the file-header template in `.editorconfig`) | MEETS | — | — |
| Security-Policy | `SECURITY.md` present with reporting contact, response SLA, disclosure window, scope | MEETS | — | — |
| Vulnerabilities | `Directory.Build.props`: `NuGetAudit=true`, `NuGetAuditMode=all`, `NU1901-1904` promoted to build errors (`WarningsAsErrors`, verified line 11) | MEETS | — | — |
| SAST | Checkmarx One (`checkmarx-one.yml`) and DevSkim (`devskim.yml`) both wired into CI and both upload SARIF via `github/codeql-action/upload-sarif@v3` | MEETS (present and wired; whether findings are triaged/gate a merge is a separate, unchecked question — see `guards-already-exist` memory note) | — | — |
| CI-Tests | `ci.yml` runs on `push`/`pull_request` to `main`, matrix `[ubuntu-latest, windows-latest]`, runs `dotnet test` | MEETS | — | — |
| Maintained | `git log` shows commits within the last day at time of check (branch `gimli`, 5 most recent commits all same-day) | MEETS | — | — |
| Dependency-Update-Tool | No `.github/dependabot.yml` or Renovate config found in the tree | **GAP** | Low — `NuGetAudit` already turns a vulnerable NuGet dependency into a build error, which is the outcome Dependabot's alerts aim at; the gap that remains is GitHub Actions and container base images, neither covered by `NuGetAudit` | Add `dependabot.yml` scoped to `github-actions` and `docker` ecosystems only; deliberately exclude `nuget` so it doesn't fight the intentional pins in `Directory.Packages.props` (e.g. `Microsoft.CodeAnalysis.CSharp` held at the SDK's Roslyn version) — leave NuGet bumps to the existing `overfit-packages-update` process |
| Fuzzing | No OSS-Fuzz integration, no in-repo fuzz harness (`dotnet-fuzz`/`SharpFuzz`) found | **GAP** | Medium — the parser surface (GGUF/ONNX/safetensors/tokenizer.json) is exactly what fuzzing is for, and it's this project's own stated largest attack surface | Out of scope to build in this pass; worth a dedicated follow-up given the parser-surface framing in this role's own definition |
| Packaging | `publish-nuget.yml` publishes via `dotnet nuget push` from a recognized CI provider (GitHub Actions), but only on `workflow_dispatch` (manual), not automatically on a release/tag | **PARTIAL** | — | Scorecard's exact heuristic for "automated enough" was not verified by running the tool — flagged so the team-lead knows this line is an estimate, not a measurement |
| Signed-Releases | No `cosign`/GPG signing step in `publish-nuget.yml` or `docker-publish.yml`; no provenance attestation (`actions/attest-build-provenance` or similar) in either | **GAP** | Medium — this is also the SLSA L1 gap named in the original shortlist | Add `actions/attest-build-provenance@<sha>` to both publish workflows; cheap, no isolated builder required |
| Branch-Protection | Needs `gh api repos/:owner/:repo/branches/main/protection` | **NOT CHECKED** — `gh` unauthenticated this session (same gap as `XC-5` in `docs/TASKS.md`) | — | User: `gh auth login`, then this check can be re-run in minutes |
| Code-Review | Needs the same branch-protection data (required-reviews setting) plus PR review history | **NOT CHECKED** | — | Same as above |
| Contributors | Needs GitHub API to check contributor company affiliation diversity | **NOT CHECKED** | — | Low priority for a single-maintainer project; Scorecard will score this low regardless |
| CII-Best-Practices | Checked for an OpenSSF Best Practices badge in `README.md` — none found; did not check bestpractices.dev directly | **NOT CHECKED** | — | Optional, self-service form at bestpractices.dev; not urgent |
| Webhooks | Needs repo-admin-scoped GitHub API access | **NOT CHECKED** | — | — |
| CI-Best-Practices / package-ecosystem-specific checks (npm, Go, etc.) | Not applicable — no npm/Go/etc. package manifests in this repo | NOT APPLICABLE | — | — |

## Worked pinning diff (one action, verified; the pattern to repeat)

`actions/checkout@v4` resolves to commit `11d5960a326750d5838078e36cf38b85af677262` as of 2026-08-10
(via `api.github.com/repos/actions/checkout/commits/v4` — **re-verify immediately before applying**,
since a tag can move between this check and the maintainer's action). This action alone appears 6 times
across the 6 workflow files.

```diff
-        uses: actions/checkout@v4
+        uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4
```

Apply the same substitution to every `uses:` line found in the earlier `grep -n "uses:"` sweep except the
one already pinned (`checkmarx/ast-github-action@8e887bb...`). **Do not hand-fetch the other 26 SHAs one
at a time in a follow-up chat** — that repeats the staleness risk 26 times over. The maintainer should run
one of:

- StepSecurity's `pin-github-actions` CLI (`npx pin-github-actions .github/workflows/*.yml`), which
  resolves and rewrites every ref in one pass and is the tool Scorecard itself points to in its own
  remediation docs, or
- `gh api repos/<owner>/<repo>/commits/<tag>` once per distinct `owner/repo@tag` pair still floating (13
  distinct actions across the 6 files, per the `uses:` sweep above), each followed by the same
  `# v<tag>` trailing-comment convention as the worked example.

I did not run either tool — this is a proposal, not an applied change, per the write boundary.

## What this audit did not do

- Did not install or run the Scorecard binary/Action itself; every verdict above is a manual read against
  Scorecard's published check descriptions, not the tool's own scoring logic. **Treat the PARTIAL/GAP
  verdicts as directionally right, not as the number Scorecard would print** — only running the tool
  produces that number.
- 5 checks are `NOT CHECKED` because they need authenticated GitHub API access this session does not
  have — listed above with the exact command that unblocks each.
