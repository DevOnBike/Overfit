"""XC-34: does the built DevOnBike.Overfit break the last package on nuget.org?

WHY THIS EXISTS. On 2026-08-12/13 seven public types were removed from `DevOnBike.Overfit` and the version
was raised from 10.0.31 to 10.1.0 BY HAND, because nothing was watching. 10.0.31 was already published, so
a build-and-push without that manual step would have republished a breaking change under an existing
number. The comparator that answers the question had landed the day before, with 48 tests and no caller.

WHAT IT DOES. Resolves the latest published version (or takes --baseline-version), downloads that .nupkg,
extracts the library assembly into a gitignored cache, points OVERFIT_API_BASELINE at it and runs the
gate test in Tests/. Prints `breaking`, `additive` or `unchanged`.

WHY THE SCRIPT AND NOT ONLY THE TEST. The test SKIPS when OVERFIT_API_BASELINE is unset, because a test
that fails without a network is a test that gets filtered out of every run. A skip is the honest result
there and it is also useless as a gate, so absence has to be loud somewhere: this script EXITS NON-ZERO
when the baseline cannot be fetched or extracted, and never converts a missing baseline into a pass.

    python Scripts/api_compat_check.py
    python Scripts/api_compat_check.py --baseline-version 10.0.30
    python Scripts/api_compat_check.py --no-build          # trust the existing Release output

EXIT CODES
    0  unchanged or additive
    1  breaking, or the test failed for any other reason
    2  the baseline could not be fetched, extracted, or the candidate is missing
"""
import argparse
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]

PACKAGE = "devonbike.overfit"
ASSEMBLY = "DevOnBike.Overfit.dll"

# Under Tests/bin, which .gitignore already covers via `[Bb]in/`. Deliberately not a new ignore entry and
# deliberately not %TEMP%: it survives between runs on the dev box and is purged by cleanup.cmd like every
# other build artefact.
CACHE = ROOT / "Tests" / "bin" / "api-baseline"

INDEX = "https://api.nuget.org/v3-flatcontainer/{p}/index.json"
NUPKG = "https://api.nuget.org/v3-flatcontainer/{p}/{v}/{p}.{v}.nupkg"

CANDIDATE = ROOT / "Sources" / "Main" / "bin" / "Release" / "net10.0" / ASSEMBLY

# Written by the test, deleted here before every run. A passing test prints nothing a runner shows, so
# without this `unchanged` and `additive` are the same green from outside — and the gate is asked for three
# answers, not two. Kept in step with ApiCompatibilityGate.VerdictPath().
VERDICT_FILE = ROOT / "Tests" / "bin" / "api-compat-verdict.txt"

TEST = "PublishedApiCompatibilityTests.TheCandidateDoesNotBreakTheLastPublishedPackage"


class Failed(Exception):
    """Anything that must exit 2: the baseline or the candidate is not there to compare."""


def fetch(url, what):
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "overfit-api-compat"})

        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise Failed("could not fetch %s from %s: %s" % (what, url, error))

    # An empty body and a negative answer look identical downstream, and this repository has been bitten by
    # exactly that with an empty stdout and a zero exit code.
    if not body:
        raise Failed("fetched %s from %s and it was EMPTY" % (what, url))

    return body


def version_key(version):
    """Semver-ish sort. nuget returns the list sorted, but relying on that puts the answer in someone
    else's hands, and a string sort would put 10.0.9 after 10.0.31."""
    parts = re.split(r"[.\-+]", version)
    key = []

    for part in parts:
        key.append((0, int(part), "") if part.isdigit() else (1, 0, part))

    return key


def latest_published():
    body = fetch(INDEX.format(p=PACKAGE), "the version index")

    try:
        versions = json.loads(body).get("versions") or []
    except json.JSONDecodeError as error:
        raise Failed("the version index is not JSON: %s" % error)

    # Prereleases carry a `-tag` and are not what a consumer gets by default.
    stable = [v for v in versions if "-" not in v]

    if not stable:
        raise Failed("no stable version in the index (%d entries total)" % len(versions))

    stable.sort(key=version_key)

    return stable[-1]


def download_baseline(version):
    """Returns the extracted assembly path. Reads the package's entry list rather than assuming
    `lib/net10.0/` — the folder name differs on older baselines, and a guessed path fails as 'not found',
    which reads like 'nothing published' instead of 'I looked in the wrong place'."""
    target = CACHE / version / ASSEMBLY

    if target.exists():
        print("baseline %s already cached: %s" % (version, target))

        return target

    url = NUPKG.format(p=PACKAGE, v=version)
    body = fetch(url, "package %s" % version)

    print("downloaded %s (%d bytes)" % (url, len(body)))

    try:
        archive = zipfile.ZipFile(io.BytesIO(body))
    except zipfile.BadZipFile as error:
        raise Failed("package %s is not a valid .nupkg: %s" % (version, error))

    entries = [n for n in archive.namelist()
               if n.lower().startswith("lib/") and n.endswith("/" + ASSEMBLY)]

    if not entries:
        libs = sorted({n.split("/")[1] for n in archive.namelist()
                       if n.lower().startswith("lib/") and "/" in n[4:]})
        raise Failed("no lib/**/%s in package %s. lib folders present: %s"
                     % (ASSEMBLY, version, libs or "<none>"))

    # Highest target framework wins, so a package that also ships net8.0 is compared on the same surface
    # the current build produces.
    entries.sort(key=lambda n: version_key(n.split("/")[1]))
    chosen = entries[-1]

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(archive.read(chosen))

    if target.stat().st_size == 0:
        raise Failed("extracted %s from %s and it was empty" % (chosen, version))

    print("extracted %s -> %s (%d bytes)" % (chosen, target, target.stat().st_size))

    return target


def build():
    print("building Sources/Main and Tests (Release) ...")
    result = subprocess.run(
        ["dotnet", "build", "-c", "Release", str(ROOT / "Tests" / "Tests.csproj")],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")

    for line in (result.stdout or "").splitlines():
        if ": error" in line or ": warning" in line:
            print("  " + line.strip())

    # NOT `returncode != 0`. The MCP navigator holds overfit-navigator.dll open, so a solution build exits 1
    # on MSB3021/3026/3027 with no compiler diagnostic. The verdict is whether the candidate assembly is
    # newer than this call started, which is checked by the caller.
    if result.returncode != 0:
        print("  (build exit %d — judged by compiler diagnostics above and by the candidate below)"
              % result.returncode)


def run_gate(baseline):
    environment = dict(os.environ)
    environment["OVERFIT_API_BASELINE"] = str(baseline)

    # A leftover from an earlier comparison reads exactly like this one's answer, and that is the failure
    # this repository finds in its own harnesses more often than any other.
    if VERDICT_FILE.exists():
        VERDICT_FILE.unlink()

    print("running %s with OVERFIT_API_BASELINE=%s" % (TEST, baseline))

    result = subprocess.run(
        ["dotnet", "test", str(ROOT / "Tests" / "Tests.csproj"), "-c", "Release", "--no-build",
         "--filter", "FullyQualifiedName~" + TEST],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=environment)

    output = (result.stdout or "") + (result.stderr or "")

    return result.returncode, output


def verdict_from(output):
    """The file the test wrote is the source of truth; the runner's output is the fallback, because a test
    that failed before writing it still has its verdict in the assertion message."""
    if VERDICT_FILE.exists():
        text = VERDICT_FILE.read_text(encoding="utf-8", errors="replace")
        found = re.search(r"API compatibility verdict: (\w+)", text)

        if found:
            return found.group(1).lower()

    found = re.search(r"API compatibility verdict: (\w+)", output)

    return found.group(1).lower() if found else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-version", help="compare against this published version instead of the latest")
    parser.add_argument("--no-build", action="store_true", help="use the existing Release output")
    arguments = parser.parse_args()

    try:
        version = arguments.baseline_version or latest_published()

        print("baseline version: %s" % version)

        baseline = download_baseline(version)

        if not arguments.no_build:
            build()

        if not CANDIDATE.exists():
            raise Failed("no candidate assembly at %s — build it with `dotnet build -c Release`" % CANDIDATE)

        print("candidate: %s (%d bytes)" % (CANDIDATE, CANDIDATE.stat().st_size))
    except Failed as error:
        # The loud half. The test can only skip; this is where absence stops the release.
        print("\nBASELINE UNAVAILABLE: %s" % error)
        print("No comparison was made. This is NOT a pass.")

        return 2

    code, output = run_gate(baseline)

    for line in output.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        # Prefixes are matched AFTER stripping, which the first version got wrong: the header lines are
        # indented in the report, so "  blocking findings" never matched and the count — the one number a
        # reader wants — was filtered out of a run that was otherwise correct.
        if (stripped.startswith(("Passed!", "Failed!", "API compatibility verdict:", "!!", "baseline ",
                                 "candidate:", "highest allowed", "blocking findings"))
                or "error" in stripped.lower()
                or re.match(r"^(Passed|Failed|Skipped)\s+\S+\.\w+", stripped)
                or "SKIPPED, not passed" in stripped):
            print("  " + stripped)

    named = verdict_from(output)

    if code == 0:
        if named is None:
            print("\nThe test passed but wrote NO verdict to %s, so the three-way answer is not available. "
                  "That is a defect in the harness, not a clean result." % VERDICT_FILE)

            return 2

        print("\nVERDICT: %s" % named)

        return 0

    if "SKIPPED, not passed" in output or re.search(r"\bSkipped[:!]\s*1", output):
        print("\nThe gate SKIPPED. The baseline was fetched, so this means the variable did not reach the "
              "test process — a comparison did NOT happen and this is not a pass.")

        return 2

    print("\nVERDICT: %s" % (named or "breaking (or the test failed for another reason — read above)"))

    return 1


if __name__ == "__main__":
    sys.exit(main())
