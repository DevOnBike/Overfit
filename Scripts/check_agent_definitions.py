"""Every `.claude/agents/*.md` must actually register as an agent.

WHY THIS EXISTS. On 2026-08-10 a commit that added `model: opus` to `overfit-developer.md` overwrote the
`memory: project` line with a bare 0x01 byte. The YAML frontmatter stopped parsing, the harness registered
nothing, and **nothing reported it**: the file was on disk, `CLAUDE.md` still described eleven agents, and
only the session's own agent list disagreed — ten. The agent that is the only one permitted to modify
source was unavailable for two days, across commits, without a single error message.

A control character is invisible in every ordinary view: `cat`, an editor and a diff all show it as a blank
line. So this reads BYTES.

The failure is silent in the worst direction — a missing agent presents exactly like an agent nobody
happened to dispatch — which is the same shape this repository's anomaly work is organised against.

    python Scripts/check_agent_definitions.py

Exit code 0 = every definition parses, 1 = at least one would not register.
"""
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENTS = ROOT / ".claude" / "agents"

# Tab, CR and LF are legal; everything else below 0x20 is not.
CONTROL = re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]")

REQUIRED = ("name", "description")


def audit(raw: bytes, stem: str):
    """Problems that would stop this file registering as an agent, worst first."""
    problems = []

    for match in CONTROL.finditer(raw):
        line = raw[:match.start()].count(b"\n") + 1
        problems.append(f"CONTROL CHARACTER 0x{raw[match.start()]:02x} on line {line}")

    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append("UTF-8 BOM before the opening '---'")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return problems + [f"not valid UTF-8: {exc}"]

    lines = text.replace("\r\n", "\n").split("\n")

    if not lines or lines[0].strip() != "---":
        return problems + ["no opening '---' on line 1"]

    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)

    if end is None:
        return problems + ["no closing '---'"]

    fields = {}

    for i in range(1, end):
        line = lines[i]

        if not line.strip():
            problems.append(f"blank line inside the frontmatter (line {i + 1})")

            continue

        if line[:1] in (" ", "\t"):
            continue  # continuation of the previous value

        if ":" not in line:
            problems.append(f"no ':' on frontmatter line {i + 1}: {line[:40]!r}")

            continue

        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()

    for key in REQUIRED:
        if not fields.get(key):
            problems.append(f"missing '{key}' — without it the agent cannot be selected")

    if fields.get("name") and fields["name"] != stem:
        problems.append(f"name '{fields['name']}' does not match the filename '{stem}'")

    return problems


def self_test():
    """A checker that finds nothing looks exactly like a tree with nothing wrong.

    So before trusting a clean run, prove the checker can still fail: three corruptions that have
    either happened here or are one keystroke away, plus the unmodified file as the negative control.
    """
    sample = b"---\nname: probe\ndescription: d\n---\n\nbody\n"
    cases = {
        "control character": sample.replace(b"description: d", b"\x01"),
        "no closing marker": sample.replace(b"---\n\nbody\n", b"\nbody\n"),
        "blank line in frontmatter": sample.replace(b"name: probe", b"name: probe\n"),
        "missing description": sample.replace(b"description: d\n", b""),
        "UNMODIFIED (must be clean)": sample,
    }

    ok = True

    for label, payload in cases.items():
        found = audit(payload, "probe")
        expect_clean = label.startswith("UNMODIFIED")

        if bool(found) == expect_clean:
            ok = False
            print(f"  SELF-TEST FAILED: {label} -> {found or 'silence'}")

    print("  self-test: the checker catches every seeded corruption" if ok
          else "  SELF-TEST FAILED — a clean result below would mean nothing")

    return ok


print("== self-test")

if not self_test():
    sys.exit(2)

print("\n== definitions")

if not AGENTS.is_dir():
    print(f"  no {AGENTS} — nothing to check")
    sys.exit(0)

files = sorted(AGENTS.glob("*.md"))
broken = 0

for path in files:
    problems = audit(path.read_bytes(), path.stem)

    if problems:
        broken += 1
        print(f"  [X] {path.name}")

        for problem in problems:
            print(f"        {problem}")

print(f"\n{len(files)} definition(s), {broken} that would not register")
sys.exit(1 if broken else 0)
