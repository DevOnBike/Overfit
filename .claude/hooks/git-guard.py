"""PreToolUse hook: allow only read-only git and gh commands.

WHY A HOOK AND NOT ONLY A DENY LIST
    `.claude/settings.json` denies ~120 git/gh forms by prefix match. Prefix matching cannot see through
    `git -C . commit`, `cd x && git commit`, `GIT_DIR=y git push` or `$(git tag -f v1)`. This parses the
    command instead: it splits on shell separators, strips environment prefixes and git's global options,
    finds the real subcommand, and checks it against an ALLOWLIST of read-only operations.

    An allowlist is the point. A deny list is wrong here by construction — every new mutating subcommand is
    permitted until somebody remembers to add it.

FAILURE POLICY, and it is deliberate
    On an unexpected exception this hook ALLOWS the command and prints a warning to stderr. Failing closed
    would block every Bash call the moment this script had a bug, leaving no way to work without editing
    settings by hand. Degrading to the settings.json deny list — which stays in place — is the safer failure.
    A git-ish command this script cannot parse is still BLOCKED: that path is narrow enough to fail closed.

Exit codes: 0 = allow, 2 = block (stderr is shown to Claude).
"""
import sys, json, re, shlex, os

# --- git subcommands that only read. Everything absent from this list is blocked.
GIT_READONLY = {
    "status", "diff", "log", "show", "describe", "blame", "shortlog", "grep",
    "rev-parse", "rev-list", "name-rev", "whatchanged", "cherry",
    "ls-files", "ls-tree", "ls-remote", "cat-file", "count-objects", "for-each-ref",
    "check-ignore", "check-attr", "verify-commit", "verify-tag", "var", "help", "version",
    "diff-tree", "diff-index", "diff-files", "merge-base", "range-diff", "difftool",
}
# Read-only only in certain forms; each has its own check below.
GIT_CONDITIONAL = {"branch", "config", "remote", "reflog", "symbolic-ref", "tag", "stash", "worktree", "notes", "bisect"}

# NOT read-only, but allowed on purpose: CLAUDE.md's contract is to finish at a "clean or STAGED" tree, so
# staging is part of the agreed end state. It destroys nothing — `git add` only moves content into the index.
# Delete this set if you decide staging is the user's alone; nothing else needs to change.
GIT_STAGING = {"add"}

# --- gh: "<command> <subcommand>" pairs that only read.
GH_READONLY = {
    ("run", "list"), ("run", "view"), ("run", "watch"), ("run", "download"),
    ("pr", "list"), ("pr", "view"), ("pr", "diff"), ("pr", "checks"), ("pr", "status"),
    ("issue", "list"), ("issue", "view"), ("issue", "status"),
    ("release", "list"), ("release", "view"),
    ("repo", "view"), ("repo", "list"),
    ("workflow", "list"), ("workflow", "view"),
    ("auth", "status"), ("browse", None), ("search", None),
    ("label", "list"), ("cache", "list"), ("secret", "list"), ("variable", "list"),
}

BLOCK_MESSAGE = (
    "BLOCKED by .claude/hooks/git-guard.py: `{cmd}`\n"
    "Git and GitHub are read-only for agents in this repository. Only read-only subcommands are allowed.\n"
    "Finish at a clean or staged working tree and report the exact command for the user to run themselves."
)


def split_segments(command: str):
    """Split a command line into the pieces a shell would run separately."""
    parts = re.split(r"&&|\|\||[;\n|]", command)
    out = []
    for p in parts:
        out.append(p)
        # Command substitutions run their own commands; check those too.
        for m in re.finditer(r"\$\(([^()]*)\)|`([^`]*)`", p):
            out.append(m.group(1) or m.group(2) or "")
    return [s.strip() for s in out if s and s.strip()]


def tokenize(segment: str):
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return segment.split()


def strip_env_prefix(tokens):
    """Drop leading VAR=value assignments and shell noise so the real program is first."""
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", t) or t in ("env", "sudo", "command", "time", "nohup"):
            i += 1
            continue
        break
    return tokens[i:]


def git_subcommand(tokens):
    """Return the real subcommand, skipping git's global options (-C, -c, --git-dir=, ...)."""
    i = 1
    while i < len(tokens):
        t = tokens[i]
        if t in ("-C", "-c", "--exec-path", "--git-dir", "--work-tree", "--namespace"):
            i += 2
            continue
        if t.startswith("-"):
            i += 1
            continue
        return t, tokens[i + 1:]
    return None, []


def git_conditional_ok(sub, args):
    """Forms of otherwise-mutating subcommands that only read."""
    flat = " ".join(args)
    if sub == "branch":
        # Listing is fine; -d/-D/-m/-M/--delete/--move/--copy/--set-upstream are not.
        return not re.search(r"(^|\s)(-[dDmMcC]|--delete|--move|--copy|--set-upstream|--unset-upstream|--edit-description)(\s|$)", " " + flat)
    if sub == "config":
        return bool(re.search(r"(^|\s)(--get|--get-all|--get-regexp|--list|-l)(\s|$)", " " + flat))
    if sub == "remote":
        return not args or args[0] in ("-v", "--verbose", "show", "get-url")
    if sub == "reflog":
        return not args or args[0] == "show"
    if sub == "symbolic-ref":
        # Reading takes one ref; writing takes a ref AND a value.
        return len([a for a in args if not a.startswith("-")]) <= 1
    if sub in ("tag", "stash", "notes", "worktree"):
        return bool(args) and args[0] in ("list", "-l", "--list", "show")
    return False


def check_git(tokens):
    sub, args = git_subcommand(tokens)
    if sub is None:
        return False, "bare `git` with no subcommand"
    if sub in GIT_READONLY or sub in GIT_STAGING:
        return True, ""
    if sub in GIT_CONDITIONAL:
        if git_conditional_ok(sub, args):
            return True, ""
        return False, f"`git {sub}` in a form that writes"
    return False, f"`git {sub}` is not a read-only subcommand"


def check_gh(tokens):
    args = [t for t in tokens[1:] if not t.startswith("-")]
    if not args:
        return False, "bare `gh`"
    cmd = args[0]
    sub = args[1] if len(args) > 1 else None

    if cmd == "api":
        # Default method is GET; anything else mutates.
        flat = " ".join(tokens)
        m = re.search(r"(?:-X|--method)\s+(\w+)", flat)
        if m and m.group(1).upper() != "GET":
            return False, f"`gh api` with method {m.group(1).upper()}"
        return True, ""

    if (cmd, sub) in GH_READONLY or (cmd, None) in GH_READONLY:
        return True, ""
    return False, f"`gh {cmd}{' ' + sub if sub else ''}` is not a read-only operation"


def main():
    raw = sys.stdin.read()
    payload = json.loads(raw) if raw.strip() else {}
    command = (payload.get("tool_input") or {}).get("command", "") or ""

    # Fast path: nothing git-ish, nothing to do. Keeps the blast radius tiny.
    # `\.exe` is matched explicitly — an earlier version excluded any following '.', to avoid matching paths
    # like `foo.git`, and `git.exe push` walked straight through it on Windows.
    if not re.search(r"(^|[^\w.-])(git|gh)(\.exe)?([^\w-]|$)", command, re.IGNORECASE):
        return 0

    for segment in split_segments(command):
        tokens = strip_env_prefix(tokenize(segment))
        if not tokens:
            continue
        program = os.path.basename(tokens[0]).lower()
        program = program[:-4] if program.endswith(".exe") else program

        if program == "git":
            ok, why = check_git(tokens)
        elif program == "gh":
            ok, why = check_gh(tokens)
        else:
            continue

        if not ok:
            sys.stderr.write(BLOCK_MESSAGE.format(cmd=segment.strip()) + f"\nReason: {why}\n")
            return 2

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — see the failure policy at the top of this file
        sys.stderr.write(f"git-guard hook error (allowing, settings.json deny list still applies): {exc}\n")
        sys.exit(0)
