"""Every package version pinned inside `Templates/` must agree with what this repository decided.

WHY THIS EXISTS (`XC-24`, found 2026-08-12 during a package survey). `dotnet new` template content may
reference only PUBLISHED NuGet packages — a path into this repo would not exist in a scaffolded app, which
is the reason recorded in `Directory.Build.props:53-54`. So `Templates/content/OverfitChat` is deliberately
outside `Overfit.sln` and outside Central Package Management, and the consequence is that **nothing in this
repository can see its pins**: no restore, no build, no analyzer, no audit, and `dotnet list package` does
not report them. They drift silently, and the survey that moved the rest of the Extensions.AI train could
not touch them.

The fix is a check rather than a move, and this is it.

WHAT IT COMPARES AGAINST, in order, naming the anchor for every pin it reports:

  1. `DevOnBike.Overfit*`      `<LastPublishedVersion>` in `Directory.Build.props` — the last version
                               actually ON nuget.org, which is NOT `<Version>`. The two differ whenever
                               work is in flight, and template content may reference only PUBLISHED
                               packages, so anchoring on `<Version>` would demand a bump to a package that
                               does not exist and break `dotnet new overfit-chat` on first restore. There is
                               deliberately NO fallback to `<Version>`: a missing `<LastPublishedVersion>`
                               is reported as "cannot be checked", because silently anchoring on the wrong
                               property is worse than saying nothing. A self-test case pins that.
  2. the same package in       `Directory.Packages.props`. If the repository pins it centrally, the template
                               must not disagree.
  3. a release-train sibling   e.g. `Microsoft.Extensions.AI` is absent from CPM but
                               `Microsoft.Extensions.AI.Abstractions` is not, and they ship in lockstep.
                               Only used when every sibling agrees on one version.
  4. nothing                   reported as a failure in its own right — a pin with no anchor is the `XC-24`
                               defect itself, not a pin that happens to be fine.

WHAT IT CANNOT TELL YOU, stated so a clean run is not over-read: it is entirely OFFLINE and says only
whether the template agrees with THIS repository. It does not query nuget.org and therefore cannot tell you
that the world moved on while the repository stood still. That is deliberate — "latest on nuget.org" is the
wrong bar for a repo that holds packages back on purpose (see the prerelease accept-list in
`Directory.Build.targets`), and a checker that needs the network is a checker CI turns off.

    python Scripts/check_template_pins.py

Exit code 0 = every pin is anchored and agrees, 1 = at least one pin drifted or has no anchor, 2 = the
self-test failed, so a clean result would have meant nothing.
"""
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "Templates"
CPM_FILE = ROOT / "Directory.Packages.props"
BUILD_PROPS = ROOT / "Directory.Build.props"

# `Include` and `Update` both pin; `Version` may be an attribute or a child element, and the child form is
# what a hand-edit usually produces. Matching only the attribute form would make this checker silently miss
# exactly the pins most likely to be wrong.
PACKAGE_REF = re.compile(
    r"<PackageReference\s+(?:Include|Update)\s*=\s*\"(?P<name>[^\"]+)\"(?P<rest>.*?)(?:/>|</PackageReference>)",
    re.DOTALL)
VERSION_ATTR = re.compile(r"Version\s*=\s*\"(?P<version>[^\"]+)\"")
VERSION_ELEM = re.compile(r"<Version>\s*(?P<version>[^<]+?)\s*</Version>")
PACKAGE_VERSION = re.compile(
    r"<PackageVersion\s+Include\s*=\s*\"(?P<name>[^\"]+)\"\s+Version\s*=\s*\"(?P<version>[^\"]+)\"")
REPO_VERSION = re.compile(r"<LastPublishedVersion>\s*(?P<version>[^<]+?)\s*</LastPublishedVersion>")


def read_text(path):
    """Bytes then decode, so a BOM never becomes part of the first tag name."""
    return path.read_bytes().decode("utf-8-sig")


def parse_pins(text):
    """Every PackageReference in a project file, as (name, version-or-None)."""
    pins = []

    for match in PACKAGE_REF.finditer(text):
        rest = match.group("rest")
        attr = VERSION_ATTR.search(rest)
        elem = VERSION_ELEM.search(rest)
        version = attr.group("version") if attr else (elem.group("version") if elem else None)
        pins.append((match.group("name"), version))

    return pins


def parse_central_versions(text):
    """`Directory.Packages.props` as a name -> version map."""
    return {m.group("name"): m.group("version") for m in PACKAGE_VERSION.finditer(text)}


def parse_repo_version(text):
    """The LAST PUBLISHED version from `Directory.Build.props`.

    Deliberately NOT `<Version>`: that is what this repository BUILDS, and template content may reference
    only packages that exist on nuget.org. On 2026-08-14 the two read 10.1.0 and 10.0.31; anchoring on the
    former would have bumped the template to an unpublished package and broken `dotnet new` on restore.
    """
    match = REPO_VERSION.search(text)

    return match.group("version") if match else None


def version_key(version):
    """Comparable form. Numeric segments before any '-' suffix; a prerelease sorts below its release."""
    core = version.split("-", 1)[0]
    parts = []

    for segment in core.split("."):
        parts.append(int(segment) if segment.isdigit() else 0)

    while len(parts) < 4:
        parts.append(0)

    return tuple(parts), 0 if "-" in version else 1


def train_sibling_version(name, central):
    """The version every `<name>.*` sibling in CPM agrees on, or None when there is no usable answer.

    A single disagreement makes this anchor unusable rather than approximate: an anchor that is right most
    of the time produces a report nobody can act on, and this checker's whole value is that its findings
    are specific enough to fix.
    """
    if len(name.split(".")) < 2:
        return None, []

    siblings = {q: v for q, v in central.items() if q.startswith(name + ".")}

    if not siblings:
        return None, []

    versions = set(siblings.values())

    if len(versions) != 1:
        return None, sorted(siblings)

    return versions.pop(), sorted(siblings)


def audit(pins, central, repo_version):
    """Findings for one project's pins, worst first: no anchor, then disagreement."""
    findings = []

    for name, version in pins:
        if version is None:
            findings.append(f"{name}: no version anywhere on the PackageReference — cannot be checked")

            continue

        if name == "DevOnBike.Overfit" or name.startswith("DevOnBike.Overfit."):
            if repo_version is None:
                findings.append(f"{name} {version}: no <LastPublishedVersion> in Directory.Build.props to compare against")

                continue

            if version != repo_version:
                findings.append(
                    f"{name} {version} != {repo_version}, the last PUBLISHED version "
                    f"(<LastPublishedVersion> in Directory.Build.props) — a scaffolded app starts life "
                    f"on an older release than the newest one customers can restore")

            continue

        if name in central:
            if version != central[name]:
                findings.append(
                    f"{name} {version} != {central[name]} in Directory.Packages.props — the template "
                    f"disagrees with central package management")

            continue

        sibling_version, siblings = train_sibling_version(name, central)

        if sibling_version is not None:
            if version != sibling_version:
                direction = "behind" if version_key(version) < version_key(sibling_version) else "ahead of"
                findings.append(
                    f"{name} {version} is {direction} {sibling_version}, the version its release train is "
                    f"pinned to in Directory.Packages.props ({', '.join(siblings)})")

            continue

        if siblings:
            findings.append(
                f"{name} {version}: its release-train siblings disagree with each other "
                f"({', '.join(siblings)}), so there is no version to check it against")

            continue

        findings.append(
            f"{name} {version}: NOT in Directory.Packages.props, not a DevOnBike package, and no release-train "
            f"sibling is — nothing in this repository can see, check or update this pin")

    return findings


def self_test():
    """A checker that finds nothing looks exactly like a tree with nothing wrong.

    So before trusting a clean run, prove it can still fail: one seeded case per branch that can report,
    plus a pin of each anchored kind that must stay silent.
    """
    central = {
        "Microsoft.Extensions.AI.Abstractions": "10.9.0",
        "Microsoft.Extensions.AI.Evaluation": "10.9.0",
        "MathNet.Numerics": "5.0.0",
        "Split.Train.One": "1.0.0",
        "Split.Train.Two": "2.0.0",
    }
    repo_version = "10.1.0"

    cases = {
        "DevOnBike pin behind the repo": ([("DevOnBike.Overfit", "10.0.29")], True),
        "central pin disagrees": ([("MathNet.Numerics", "4.9.0")], True),
        "train sibling drift": ([("Microsoft.Extensions.AI", "10.8.0")], True),
        "no anchor at all": ([("Some.Unknown.Package", "1.2.3")], True),
        "no version attribute": ([("MathNet.Numerics", None)], True),
        "train siblings disagree": ([("Split.Train", "1.0.0")], True),
        "DevOnBike pin current (must be clean)": ([("DevOnBike.Overfit.Extensions.AI", "10.1.0")], False),
        "central pin matches (must be clean)": ([("MathNet.Numerics", "5.0.0")], False),
        "train sibling matches (must be clean)": ([("Microsoft.Extensions.AI", "10.9.0")], False),
    }

    ok = True

    for label, (pins, expect_finding) in cases.items():
        found = audit(pins, central, repo_version)

        if bool(found) != expect_finding:
            ok = False
            print(f"  SELF-TEST FAILED: {label} -> {found or 'silence'}")

    # The parser is half the checker, and a regex that quietly matches nothing reports a clean tree.
    parsed = parse_pins(
        "<PackageReference Include=\"A\" Version=\"1.0.0\" />\n"
        "<PackageReference Include=\"B\">\n  <Version>2.0.0</Version>\n</PackageReference>\n"
        "<PackageReference Include=\"C\" />\n")

    if parsed != [("A", "1.0.0"), ("B", "2.0.0"), ("C", None)]:
        ok = False
        print(f"  SELF-TEST FAILED: pin parser -> {parsed}")

    if parse_central_versions("<PackageVersion Include=\"X\" Version=\"9.9.9\" />") != {"X": "9.9.9"}:
        ok = False
        print("  SELF-TEST FAILED: central version parser")

    if parse_repo_version(
            "<Project><PropertyGroup><LastPublishedVersion>7.7.7</LastPublishedVersion>"
            "</PropertyGroup></Project>") != "7.7.7":
        ok = False
        print("  SELF-TEST FAILED: last-published version parser")

    # The anchor moved from <Version> to <LastPublishedVersion> on 2026-08-14, and this case exists so the
    # move cannot silently half-happen: reading <Version> would anchor the template on a package that is
    # not on nuget.org yet. The self-test caught exactly that when the parser was changed and this case
    # was not.
    if parse_repo_version("<Project><PropertyGroup><Version>7.7.7</Version></PropertyGroup></Project>") is not None:
        ok = False
        print("  SELF-TEST FAILED: <Version> must NOT be accepted as the anchor")

    print("  self-test: the checker catches every seeded drift and parses all three forms" if ok
          else "  SELF-TEST FAILED — a clean result below would mean nothing")

    return ok


print("== self-test")

if not self_test():
    sys.exit(2)

if not TEMPLATES.is_dir():
    print(f"\nno {TEMPLATES} — nothing to check")
    sys.exit(0)

central_versions = parse_central_versions(read_text(CPM_FILE))
repository_version = parse_repo_version(read_text(BUILD_PROPS))

print(f"\n== compared against")
print(f"  <LastPublishedVersion> (nuget)    {repository_version}")
print(f"  Directory.Packages.props          {len(central_versions)} centrally pinned package(s)")

projects = sorted(TEMPLATES.rglob("*.csproj"))
total_pins = 0
drifted = 0

print(f"\n== {len(projects)} template project(s)")

for project in projects:
    pins = parse_pins(read_text(project))
    total_pins += len(pins)
    problems = audit(pins, central_versions, repository_version)
    relative = project.relative_to(ROOT).as_posix()

    if not problems:
        print(f"  [ok] {relative} — {len(pins)} pin(s), all anchored and current")

        continue

    drifted += len(problems)
    print(f"  [X] {relative}")

    for problem in problems:
        print(f"        {problem}")

print(f"\n{total_pins} pin(s) across {len(projects)} project(s), {drifted} that drifted or cannot be checked")
sys.exit(1 if drifted else 0)
