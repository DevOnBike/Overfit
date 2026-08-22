"""What state was this repository actually in when that result was produced?

    import sys
    sys.path.insert(0, r"D:\\Overfit\\Scripts")
    from provenance import snapshot

    record = snapshot("unexplained red in MiniInstructionCheckpointTests")
    record.report()

    if record.stale:
        print("rebuild before believing anything this run printed")

Standalone, which is how it is meant to be used the moment a surprising result appears::

    python D:/Overfit/Scripts/provenance.py
    python D:/Overfit/Scripts/provenance.py --json --out Tests/bin/prov-before.json
    python D:/Overfit/Scripts/provenance.py --compare Tests/bin/prov-before.json

Exit codes standalone: ``0`` clean, ``3`` at least one assembly is older than its own source, ``2`` the
record could not be built (git did not answer, so the record cannot later be read as evidence of anything).

**Why this exists.** `XC-106`: `Demo_LoadCheckpoint_AndShowMiniInstructionGeneration` was red on 2026-08-20
and 2026-08-21 with a fixed sampler seed, a tracked checkpoint and nobody having edited the test, and it is
green now. Five candidate commits were eliminated — three by direct revert with byte-identical output, one
by prior measurement, two by date — and no committed change on the reachable path accounts for it. The
leading account is that the red was never a property of any committed source state: it was read from a stale
binary or a dirty tree. **That can now be neither confirmed nor refuted, because the build artefacts and
timestamps from those two days are gone.** The record this module writes is the thing whose absence closed
that door, and it costs under a second.

**The unit of the staleness check is seconds of margin between an assembly's modification time and the
newest compile input that feeds it.** That unit was chosen, not assumed: MSBuild's own up-to-date decision
is made on exactly that comparison, so a check in any other unit would be answering a different question
from the one that governs whether the binary under test was rebuilt. A hash difference tells you two builds
differ; a file count tells you nothing about which is newer.

**Know precisely which direction of staleness this catches, because it is not the only one.**

  - **Caught: source newer than the assembly.** Somebody edited a `.cs` and ran the tests without a rebuild,
    or a harness re-stamped a restored file (`Scripts.mutate.touch` does this deliberately). The margin goes
    negative and the record says so.
  - **NOT caught: assembly newer than a source that moved backwards.** This is the 2026-08-19 incident that
    produced `Scripts/mutate.py` — a `shutil.copy2` restore preserved the modification time, so the restored
    correct source was *older* than the object built from the mutated source, MSBuild concluded there was
    nothing to rebuild, and a full suite reported **100 failures against source byte-identical to a green
    commit**, with a timestamp that had gone backwards by 113.5 s. In that shape the assembly is newer than
    the source and every margin here is positive.

The second shape is why **hash, size and mtime are all recorded, not just the hash**, and why ``--compare``
exists: two records taken around a suspicious run settle it, because a dll whose hash changed while `HEAD`
and the dirty list did not, or a source tree whose newest mtime went backwards, are both visible in the
difference and invisible in either record alone. `XC-99`'s developer run used exactly that signal — the dll
hash returning to its pre-mutation value — as its evidence that a restore had taken.

**No tolerance is applied to the margin, and that is deliberate.** NTFS stores 100 ns granularity, and
Python's `st_mtime` truncates against .NET's ticks at about the microsecond (measured on the same untouched
file: `.968373Z` from Python, `.9683732Z` from .NET). That is seven orders of magnitude below the 113.5 s
the recorded incident produced and below the seconds a real compile takes, so inventing a fudge factor would
only widen the window in which a genuine miss reads as clean.

**Empty and unknown are different answers and are never merged.** An empty dirty list means the tree is
clean; a dirty list of `None` means git did not answer and carries the reason it did not. A record that
silently recorded nothing would be worse than no record, because it would later be read as evidence.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Assemblies recorded by default, as ``(output path, owning project)``, both relative to :data:`ROOT`.
#:
#: **The test assembly and the library's copy under `Tests/bin` are both here on purpose.** The library
#: output under `Sources/Main/bin` is not the file a test run loads — the test host loads the copy beside
#: `DevOnBike.Overfit.Tests.dll`, and the two can differ. `XC-106` is a question about what a *test* printed,
#: so recording only the library output would record the wrong file's identity. Recording the test assembly
#: as well is what distinguishes "the library changed" from "the test changed" when a later reader compares
#: two records; it is the difference between a red that moved with the product and one that moved with its
#: oracle.
ASSEMBLIES = (
    ("Sources/Main/bin/Release/net10.0/DevOnBike.Overfit.dll", "Sources/Main/Main.csproj"),
    ("Tests/bin/Release/net10.0/DevOnBike.Overfit.dll", "Sources/Main/Main.csproj"),
    ("Tests/bin/Release/net10.0/DevOnBike.Overfit.Tests.dll", "Tests/Tests.csproj"),
)

#: File suffixes counted as compile inputs when looking for the newest source.
#:
#: `.txt` is here because `BannedSymbols.txt` is an MSBuild `AdditionalFiles` entry and changing it changes
#: what the build produces. **`.md` is deliberately absent**: the only build inputs with that suffix are the
#: analyzer release trackers, while every README under `Sources/` would produce a stale flag for a binary it
#: cannot affect — and a guard that cries wolf is how a team learns to ignore one.
SOURCE_SUFFIXES = {".cs", ".csproj", ".props", ".targets", ".resx", ".txt"}

#: Files counted as compile inputs by name, for those that have no suffix `pathlib` will report.
SOURCE_NAMES = {".editorconfig"}

#: Never walked when looking for the newest source. `bin` and `obj` hold the build's own output, and
#: including them would compare a build against itself.
SKIP_DIRECTORIES = {"bin", "obj", ".git", ".vs", ".idea", "node_modules", "TestResults", "packages"}

#: Repository-root files that feed every project here. `Directory.Build.props` carries the version and the
#: warning promotions, `Directory.Packages.props` the pinned package versions, `.editorconfig` the analyzer
#: severities — each of them changes what a build produces without living under any one project.
ROOT_BUILD_FILES = (
    "Directory.Build.props",
    "Directory.Build.targets",
    "Directory.Packages.props",
    ".editorconfig",
)

_COMMIT = re.compile(r"^[0-9a-f]{40}$")

_PROJECT_REFERENCE = re.compile(r'ProjectReference\s+Include\s*=\s*"([^"]+)"')

#: Ceiling on the project-reference walk. The graph is a DAG of a dozen projects, so this can only be
#: reached by a cycle in the csproj files, and a harness that hangs is worse than one that reports a bound.
#: BOUND: project graph depth, 64 projects visited.
MAX_PROJECTS = 64


def _git(arguments):
    """Runs one git command under :data:`ROOT`, returning ``(stdout, reason)`` with ``reason`` set on failure."""
    printable = "git " + " ".join(arguments)

    try:
        process = subprocess.run(
            ["git", "-C", str(ROOT)] + list(arguments),
            capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, "%s could not run: %s" % (printable, error)

    if process.returncode != 0:
        return None, "%s exited %d: %s" % (printable, process.returncode, (process.stderr or "").strip())

    return process.stdout, None


def head_commit():
    """``(commit id, reason)`` for `git rev-parse HEAD`, shape-checked rather than merely non-empty.

    A malformed answer and a real one look identical downstream once the record is written, so the value is
    matched against a 40-character hex id before it is accepted.
    """
    text, reason = _git(["rev-parse", "HEAD"])

    if reason:
        return None, reason

    value = (text or "").strip()

    if not _COMMIT.match(value):
        return None, "git rev-parse HEAD returned %r, which is not a commit id" % value

    return value, None


def dirty_paths():
    """``(list of porcelain lines, reason)``. An empty list means clean; ``None`` means git did not answer.

    The full list is kept rather than a count. A count cannot tell a later reader whether the one file that
    mattered was among them, which is the only question anybody asks of this field.
    """
    text, reason = _git(["status", "--porcelain"])

    if reason:
        return None, reason

    return [line for line in (text or "").splitlines() if line.strip()], None


def hash_file(path, chunk=1 << 20):
    """SHA-256 of a file, read in chunks so a multi-megabyte assembly does not land in memory whole."""
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)

    return digest.hexdigest()


def project_sources(csproj):
    """Directories whose source compiles into ``csproj``, following `ProjectReference` transitively.

    Resolved from the csproj files rather than from a hand-written list, because a hand-written list of
    project references is exactly the kind of thing that stops matching the build without anybody noticing.
    """
    start = pathlib.Path(csproj)
    pending = [start]
    seen = set()
    directories = []

    while pending and len(seen) < MAX_PROJECTS:
        current = pending.pop()
        resolved = current.resolve()

        if resolved in seen:
            continue

        seen.add(resolved)

        if not resolved.is_file():
            continue

        directories.append(resolved.parent)

        try:
            text = resolved.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for include in _PROJECT_REFERENCE.findall(text):
            pending.append(resolved.parent / include.replace("\\", "/"))

    return directories


def newest_source(directories):
    """``(path, mtime)`` of the most recently modified compile input under any of ``directories``.

    The repository-root build files are always considered, because they feed every project and live under
    none of them. Returns ``(None, None)`` when nothing was found, which the caller must not read as "fresh".
    """
    newest_path = None
    newest_mtime = None

    candidates = []

    for name in ROOT_BUILD_FILES:
        candidates.append(ROOT / name)

    for directory in directories:
        for folder, subfolders, files in os.walk(directory):
            subfolders[:] = [name for name in subfolders if name not in SKIP_DIRECTORIES]

            for name in files:
                if pathlib.PurePath(name).suffix in SOURCE_SUFFIXES or name in SOURCE_NAMES:
                    candidates.append(pathlib.Path(folder) / name)

    for candidate in candidates:
        try:
            mtime = candidate.stat().st_mtime
        except OSError:
            continue

        if newest_mtime is None or mtime > newest_mtime:
            newest_mtime = mtime
            newest_path = candidate

    return newest_path, newest_mtime


def _utc(epoch):
    """An epoch in UTC, to the second. Sub-second digits are noise here and differ between readers."""
    if epoch is None:
        return None

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _configuration(path):
    """``(configuration, framework)`` read back out of a `bin/<config>/<tfm>/` output path.

    Reproducing a result needs to know which configuration produced the binary, and this repository builds
    and tests in Release only — so a record showing `Debug` is itself the answer to some future question.
    """
    parts = list(pathlib.PurePath(path).parts)

    # Scanned from the END. Taking the first `bin` reports the wrong pair whenever anything sits underneath
    # one - `Tests/bin/scratch/Proj/bin/Release/net10.0/x.dll` gave `[scratch/Proj]` until this was fixed,
    # and a real output path has exactly one `bin`, so the defect was invisible on every real assembly.
    for index in range(len(parts) - 3, -1, -1):
        if parts[index].lower() == "bin":
            return parts[index + 1], parts[index + 2]

    return None, None


class Assembly:
    """One built assembly, its identity, and whether it is older than the source that supposedly produced it."""

    def __init__(self, output, csproj):
        self.path = str(ROOT / output) if not pathlib.Path(output).is_absolute() else str(output)
        self.project = str(ROOT / csproj) if not pathlib.Path(csproj).is_absolute() else str(csproj)
        self.name = pathlib.PurePath(self.path).name
        self.configuration, self.framework = _configuration(self.path)

        self.exists = False
        self.reason = None
        self.sha256 = None
        self.size_bytes = None
        self.mtime = None
        self.source_path = None
        self.source_mtime = None
        self.margin_seconds = None
        self.stale = False

        self._measure()

    def _measure(self):
        file = pathlib.Path(self.path)

        if not file.is_file():
            # The reason, not just the absence: "never built" and "built into a different configuration"
            # are different situations and the reader cannot tell them apart from a missing field.
            self.reason = "no file at this path — the project was not built into this configuration"

            return

        try:
            stat = file.stat()
            self.sha256 = hash_file(file)
        except OSError as error:
            self.reason = "could not read the assembly: %s" % error

            return

        self.exists = True
        self.size_bytes = stat.st_size
        self.mtime = stat.st_mtime

        project = pathlib.Path(self.project)

        if not project.is_file():
            self.reason = "no project file at %s — staleness not checked" % self.project

            return

        source_path, source_mtime = newest_source(project_sources(project))

        if source_mtime is None:
            self.reason = "found no compile inputs under %s — staleness not checked" % project.parent

            return

        self.source_path = str(source_path)
        self.source_mtime = source_mtime
        self.margin_seconds = self.mtime - source_mtime

        # Strictly older, with no tolerance. See the module docstring for why no fudge factor is applied
        # and for the direction of staleness this does NOT see.
        self.stale = self.margin_seconds < 0.0

    def to_dict(self):
        return {
            "name": self.name,
            "path": self.path,
            "project": self.project,
            "configuration": self.configuration,
            "framework": self.framework,
            "exists": self.exists,
            "reason": self.reason,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "mtime": self.mtime,
            "mtime_utc": _utc(self.mtime),
            "newest_source": self.source_path,
            "newest_source_mtime": self.source_mtime,
            "newest_source_mtime_utc": _utc(self.source_mtime),
            "margin_seconds": self.margin_seconds,
            "stale": self.stale,
        }

    def lines(self):
        if not self.exists:
            return ["%s: MISSING — %s" % (self.name, self.reason),
                    "    path %s" % self.path]

        rows = ["%s  %s  %d bytes  %s  [%s/%s]"
                % (self.name, (self.sha256 or "")[:16], self.size_bytes, _utc(self.mtime),
                   self.configuration, self.framework),
                "    path %s" % self.path]

        if self.margin_seconds is None:
            rows.append("    staleness NOT CHECKED — %s" % self.reason)

            return rows

        verdict = "*** STALE ***" if self.stale else "up to date"

        rows.append("    %s — assembly is %+.1f s against the newest compile input" % (verdict, self.margin_seconds))
        rows.append("    newest input %s (%s)" % (self.source_path, _utc(self.source_mtime)))

        return rows


class Provenance:
    """Everything worth knowing about the tree and the binaries at one instant."""

    def __init__(self, label, head, head_reason, dirty, dirty_reason, assemblies):
        self.label = label
        self.taken_at = time.time()
        self.head = head
        self.head_reason = head_reason
        self.dirty = dirty
        self.dirty_reason = dirty_reason
        self.assemblies = assemblies

    @property
    def stale(self):
        """True when any recorded assembly is older than its own source."""
        return any(assembly.stale for assembly in self.assemblies)

    @property
    def usable(self):
        """False when the identity of the tree could not be established, so the record proves nothing."""
        return self.head is not None and self.dirty is not None

    def to_dict(self):
        return {
            "label": self.label,
            "taken_at": self.taken_at,
            "taken_at_utc": _utc(self.taken_at),
            "root": str(ROOT),
            "head": self.head,
            "head_reason": self.head_reason,
            "dirty": self.dirty,
            "dirty_reason": self.dirty_reason,
            "assemblies": [assembly.to_dict() for assembly in self.assemblies],
            "stale": self.stale,
            "usable": self.usable,
        }

    def report(self):
        """Prints the record and returns True when nothing is stale."""
        print("[provenance] %s — taken %s" % (self.label, _utc(self.taken_at)))
        print("[provenance] root %s" % ROOT)

        if self.head:
            print("[provenance] HEAD %s" % self.head)

        if self.head_reason:
            print("[provenance] HEAD UNKNOWN — %s" % self.head_reason)

        if self.dirty_reason:
            print("[provenance] working tree UNKNOWN — %s" % self.dirty_reason)

        if self.dirty is not None:
            if not self.dirty:
                print("[provenance] working tree CLEAN")
            else:
                print("[provenance] working tree DIRTY — %d path(s):" % len(self.dirty))

                for line in self.dirty:
                    print("[provenance]     %s" % line)

        for assembly in self.assemblies:
            for line in assembly.lines():
                print("[provenance] %s" % line)

        if self.stale:
            print("[provenance] *** AT LEAST ONE ASSEMBLY IS OLDER THAN ITS OWN SOURCE ***")
            print("[provenance] a run against it did not execute the source in this tree — rebuild first")

        return not self.stale


def snapshot(label="provenance", assemblies=None):
    """Records the tree identity and every assembly that matters, right now.

    ``assemblies`` overrides :data:`ASSEMBLIES` and takes ``(output path, owning project)`` pairs.
    """
    head, head_reason = head_commit()
    dirty, dirty_reason = dirty_paths()

    # `is None`, not `or`: an explicitly empty list means "record no assemblies", and silently substituting
    # the default set for it would hand a caller three files it did not ask about.
    if assemblies is None:
        assemblies = ASSEMBLIES

    recorded = [Assembly(output, csproj) for output, csproj in assemblies]

    return Provenance(label, head, head_reason, dirty, dirty_reason, recorded)


def compare(earlier, later):
    """Differences between an earlier record (a dict, as written by ``--out``) and a later one.

    **This is the arm that sees the staleness the mtime check cannot.** An assembly whose hash moved while
    `HEAD` and the dirty list did not is a binary that changed without the source changing, which is the
    2026-08-19 shape; a source tree whose newest modification time went backwards is the restore that caused
    it. Neither is visible in a single record.
    """
    findings = []
    now = later.to_dict()

    if earlier.get("head") != now["head"]:
        findings.append("HEAD moved: %s -> %s" % (earlier.get("head"), now["head"]))

    if earlier.get("dirty") != now["dirty"]:
        findings.append("working tree changed: %d dirty path(s) -> %d"
                        % (len(earlier.get("dirty") or []), len(now["dirty"] or [])))

    same_tree = earlier.get("head") == now["head"] and earlier.get("dirty") == now["dirty"]
    previous = {entry["path"]: entry for entry in earlier.get("assemblies", [])}

    for entry in now["assemblies"]:
        before = previous.get(entry["path"])

        if before is None:
            findings.append("%s: not in the earlier record" % entry["name"])

            continue

        if before.get("sha256") != entry["sha256"]:
            note = " WHILE HEAD AND THE WORKING TREE DID NOT MOVE" if same_tree else ""
            findings.append("%s: hash changed %s -> %s%s"
                            % (entry["name"], (before.get("sha256") or "none")[:16],
                               (entry["sha256"] or "none")[:16], note))

        if before.get("mtime") and entry["mtime"] and entry["mtime"] < before["mtime"]:
            findings.append("%s: modification time went BACKWARDS by %.1f s"
                            % (entry["name"], before["mtime"] - entry["mtime"]))

        if (before.get("newest_source_mtime") and entry["newest_source_mtime"]
                and entry["newest_source_mtime"] < before["newest_source_mtime"]):
            findings.append(
                "%s: the newest compile input went BACKWARDS by %.1f s — a restore that preserved a "
                "timestamp will stop MSBuild rebuilding"
                % (entry["name"], before["newest_source_mtime"] - entry["newest_source_mtime"]))

    return findings


def main(argv=None):
    # Reconfigured here rather than at import, so importing this module does not change a caller's stdout.
    # MSBuild output is not cp1252 and this console is, so both ends need saying somewhere.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Record what state this repository was in.")
    parser.add_argument("--label", default="provenance", help="what this record is about")
    parser.add_argument("--json", action="store_true", help="print the record as JSON")
    parser.add_argument("--out", help="write the record as JSON to this path")
    parser.add_argument("--compare", help="an earlier JSON record to diff against")
    arguments = parser.parse_args(argv)

    record = snapshot(arguments.label)

    if arguments.json:
        print(json.dumps(record.to_dict(), indent=2))
    else:
        record.report()

    if arguments.out:
        destination = pathlib.Path(arguments.out)

        if not destination.is_absolute():
            destination = ROOT / destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
        print("[provenance] written to %s" % destination)

    if arguments.compare:
        source = pathlib.Path(arguments.compare)

        if not source.is_absolute():
            source = ROOT / source

        earlier = json.loads(source.read_text(encoding="utf-8"))
        findings = compare(earlier, record)

        print("[provenance] against %s:" % source)

        if not findings:
            print("[provenance]     nothing moved")

        for finding in findings:
            print("[provenance]     %s" % finding)

    if not record.usable:
        # The identity of the tree is the part that makes this a record rather than a note. Without it a
        # later reader would be building on something that says nothing, which is worse than nothing.
        return 2

    return 3 if record.stale else 0


if __name__ == "__main__":
    sys.exit(main())
