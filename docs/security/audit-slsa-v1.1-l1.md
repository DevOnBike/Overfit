# Audit — SLSA v1.1, Build Track Level 1

Standard: SLSA v1.1 (stable), Build Track Level 1 only, per `slsa.dev/spec/v1.1/levels` (fetched
2026-08-10). Commissioned by the team-lead 2026-08-10 as the third and final audit from the original
shortlist. Scope: **the published NuGet package `DevOnBike.Overfit`**, per the team-lead's explicit
framing — what a consumer can retrieve and check *today*, not what the CI could be made to emit. The
Docker image (`docker-publish.yml`) gets a shorter secondary note since it's a second build artifact this
project ships, but is not the primary subject.

**Verdict counts: 0 MEETS, 3 GAP, 1 PARTIAL, 0 NOT APPLICABLE, 2 NOT CHECKED.**

## What L1 requires, verbatim from the spec

Software producer: (1) follow a consistent build process, (2) build on a platform meeting L1, (3)
distribute provenance to consumers. Build platform: (4) automatically generate provenance identifying
what entity built the artifact, what build process was used, and the top-level inputs. **The spec states
provenance "may be incomplete and/or unsigned at L1"** — L1 is a documentation/mistake-catching bar, not
an anti-tampering one.

## Findings that matter, ranked

1. **No provenance artifact exists today, in any form, for either shipped artifact (NuGet or Docker).**
   Verified live: the published package page (`nuget.org/packages/DevOnBike.Overfit`, checked 2026-08-10)
   has no Provenance tab, no attestation, no signed-publisher badge — only the standard README/
   Frameworks/Dependencies/Versions tabs. This is requirement (4), the load-bearing one; without it,
   (3) has nothing to distribute.
2. **The build process is not exclusively scripted — nothing technically prevents a local, ad hoc
   publish.** `publish-nuget.yml` pushes with `dotnet nuget push --api-key ${{ secrets.NUGET_API_KEY }}`,
   a long-lived key. That same key, if held locally (it has to be, to configure the GitHub secret in the
   first place), can run the identical `dotnet nuget push` command from a laptop, producing a package
   nuget.org and any consumer would treat as identical to a workflow-built one. Requirement (1)
   ("consistent process") is a stated intention, not something enforced by any control this audit found.
3. **Git tags don't cover every release.** `git tag -l`: `10.0.15`, `10.0.22`, `10.0.29`, `10.0.30`,
   `demo` — current source version is `10.0.31` (`Directory.Build.props`), untagged, and the gaps between
   the tagged versions (e.g. nothing between `.22` and `.29`) mean a consumer trying to independently
   rebuild a specific published version and diff it against the package cannot always find the exact
   commit from a tag alone — they'd need the SourceLink-embedded commit SHA instead (see below), which
   does cover every build.
4. **NuGet.org does not support verifiable provenance today, even for a maximally diligent publisher.**
   Checked 2026-08-10: NuGet.org's Trusted Publishing (OIDC-based, no long-lived key) is live, but per
   Microsoft's own docs there are "no additional benefits... aside from ease of publishing... though
   additional benefits with... provenance attestations are foreseeable in the future" — i.e. the registry
   itself has no attestation-verification surface yet, unlike npm (which shipped provenance verification
   in 2025). `publish-nuget.yml` doesn't use Trusted Publishing either way (confirmed: `--api-key`, not
   OIDC) — a separate, smaller finding (credential-handling hygiene, not an SLSA L1 clause) worth noting
   alongside this one since fixing #2 naturally goes through the same workflow step.

## What DOES exist for a consumer today (partial credit, not enough for L1)

- **SourceLink**: `Main.csproj` sets `PublishRepositoryUrl=true`, `EmbedUntrackedSources=true`,
  `DebugType=embedded` (PDB embedded in the DLL, not a separate `.snupkg`). `publish-nuget.yml` passes
  `-p:SourceRevisionId=<short-sha>` at build and `-p:RepositoryCommit=${{ github.sha }}` at pack. **A
  consumer who downloads the `.nupkg` and inspects the embedded nuspec/PDB can recover the exact commit
  the assembly was built from and the repository URL** — this is real, retrievable, and *covers every
  build* (unlike the tags). But it identifies the **source commit**, not "what entity built it" or "what
  build process was used" — SLSA provenance requirements (4) asks for more than SourceLink was designed
  to answer, and it is not a machine-verifiable attestation (no signature, no in-toto/SLSA predicate
  format, no registry-side verification) — a consumer must trust their own DLL inspection, not check
  anything against an independent record of what CI produced.

## Table

| L1 requirement | What was checked, and how | Verdict | What it means for a NuGet consumer today | Remedy |
|---|---|---|---|---|
| (1) Consistent build process | Read `publish-nuget.yml` in full; checked whether any technical control (branch protection requiring the workflow, an environment gate, OIDC-only publishing) prevents a manual push | **GAP** | A consumer has no way to know whether any given version was built by the workflow or pushed by hand — nothing distinguishes the two in the artifact itself | Migrate to NuGet Trusted Publishing (OIDC) so publishing is *only possible* from the named workflow run — this closes the gap structurally rather than by policy |
| (2) Build on an L1-capable platform | GitHub Actions (`ubuntu-latest`) is used for the scripted path; not enforced as the *only* path (see requirement 1) | **PARTIAL** | Same caveat as (1) — the platform is fine when used, but nothing requires it | Same remedy as (1) |
| (3) Distribute provenance to consumers | Live-checked `nuget.org/packages/DevOnBike.Overfit` for a Provenance tab or attached attestation | **GAP** | Nothing to distribute because nothing is generated (requirement 4) | Depends on (4) |
| (4) Build platform auto-generates provenance (entity, process, top-level inputs) | Grepped `publish-nuget.yml` for `attest`, `provenance`, `in-toto`, `slsa` — none found; confirmed no `actions/attest-build-provenance` or equivalent step | **GAP** | Nothing exists beyond the SourceLink commit-SHA evidence described above, which answers "which commit" but not "which entity/process built it" in a verifiable form | Add `actions/attest-build-provenance@<sha>` (or the NuGet-specific pattern behind the "Creating provenance attestations for NuGet packages in GitHub Actions" approach — generate an in-toto SLSA predicate over the `.nupkg` hash, attach it to a GitHub Release since NuGet.org itself has no attestation storage yet) |
| Docker image (`docker-publish.yml`) — secondary note | `docker/build-push-action@v6` is used without an explicit `provenance:` input; Buildx has generated default-mode provenance attestations on registry pushes since v5/buildx 0.10 (2023), so the Docker Hub push **may already carry minimal attestation data for free** | **NOT CHECKED** — would need `docker buildx imagetools inspect <image>` against the live published image, which this session cannot run | Unknown until inspected — flagging so it isn't assumed absent OR assumed present | Run `docker buildx imagetools inspect devonbikeit/overfit:<version> --format '{{ json .Provenance }}'` once, to settle this rather than guess |
| Whether any release has, in practice, ever been published locally rather than via the workflow | Would need GitHub Actions run history (`gh run list`) correlated with nuget.org version timestamps | **NOT CHECKED** — `gh` unauthenticated this session, same gap as `XC-5` | This is a practice question, not a control question — even a clean history wouldn't fix requirement (1)'s structural gap | `gh auth login`, then `gh run list --workflow=publish-nuget.yml` |

## What L1 does not buy — say this plainly, because a pass would be easy to over-read

Even a full L1 pass **would not address**:

- **A compromised GitHub Actions runner or a repointed floating action tampering with the build.** L1
  provenance may be unsigned and the builder is not required to be isolated — tampering *during* the
  build is explicitly out of scope until L3. This project's own already-measured gap (**1 of 28 `uses:`
  refs SHA-pinned**, from the Scorecard audit) sits exactly in this uncovered space: a repointed
  `actions/checkout@v4` could alter what gets built, and L1 provenance — even if implemented — would
  faithfully attest to the tampered build as if it were legitimate, because L1 doesn't verify the builder
  itself.
- **A leaked or reused `NUGET_API_KEY` publishing a tampered package.** Nothing about L1 provenance
  prevents this; it only helps a consumer *notice* after the fact, and only if they check — and today
  there is nothing to check.
- **Tampering after the build** (a nupkg swapped en route or on the registry) — that's an L2 concern
  (signed provenance, non-forgeable by the registry operator) and is separately unaddressed regardless of
  this project's L1 status.

**A future "SLSA" line in any public document must say the level.** Even reaching L1 fully would still
leave the two problems above (tampering during/after build) completely open — stating "SLSA-compliant"
without "L1" would read as far more assurance than L1 actually provides, and would be exactly the kind of
overreach this role's own instructions warn against for the untrusted-model claim in `SECURITY.md`.

## What this audit did not do

- Did not inspect the live Docker Hub manifest for existing Buildx-default provenance — flagged as
  `NOT CHECKED` with the exact command rather than assumed either way.
- Did not check GitHub Actions run history against nuget.org version timestamps (`gh` unauthenticated).
- Did not evaluate L2 or L3 requirements — out of scope, per the team-lead's explicit "L1 only."
- Did not check the Android/Play artifact (`overthink-playstore.yml`) for provenance — out of scope per
  the NuGet-consumer framing this round; flag if a future pass should cover it too.
