"""Does every row in docs/TASKS.md carry a status a machine can read?

WHY THIS EXISTS. The registry defines six status words at the top of the file and then, over about a month,
accumulated twenty-three rows that used none of them -- `IMPLEMENTED`, `REVIEWED`, `MEASURED`, `DECIDED`,
`CLOSED`, `AUDITED`, `SUPERSEDED`, `REFUTED`, and one row whose status began "(a) and (b) DELIVERED". Each
was true and none was answerable, so the one question the file exists to answer -- what is open -- could not
be answered from it.

That is not hypothetical. `XC-47`'s own note ended in "DONE 2026-08-14" while its status cell still read
`OPEN`, so it was proposed as work and cost a re-verification pass; `XC-69` asked for a test that had
already been committed, mutation-proved and shipped. Both were found by reading rows one at a time, which is
exactly what a registry is supposed to make unnecessary.

WHAT IT CHECKS, and deliberately nothing more:

  1. every table row whose first cell is a task id has a non-empty status cell;
  2. that cell BEGINS with one of the six vocabulary words;
  3. no id is defined twice -- ids are never reused, and one collision has already happened here
     (`AN-D12`, 2026-08-12).

It does NOT check whether the status is TRUE. Nothing can: "this row says DONE and the work is done" is a
judgement about the world. What this buys is that the field is readable, so a wrong status is a
disagreement someone can have rather than a sentence nobody can parse.

    python Scripts/check_task_statuses.py

EXIT CODES
    0  every row is well-formed
    1  at least one row is not -- each is printed with its id and what is wrong
    2  docs/TASKS.md could not be read, or contains no task rows at all (which is itself a failure: a
       silently-empty scan and a clean file look identical, and this repository has been bitten by that)
"""
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASKS = ROOT / "docs" / "TASKS.md"

VOCABULARY = ("DONE", "PART", "OPEN", "FAILED", "UNGATED", "DEFER")

ROW = re.compile(r"^\|\s*`([A-Z]{2}-[A-Za-z0-9]+)`\s*\|([^|]*)\|")


def main():
    if not TASKS.is_file():
        print("cannot read %s" % TASKS)

        return 2

    text = TASKS.read_text(encoding="utf-8")
    problems = []
    seen = {}
    rows = 0

    for number, line in enumerate(text.split("\n"), start=1):
        found = ROW.match(line)

        if not found:
            continue

        rows += 1
        task_id = found.group(1)

        # The vocabulary word is what matters; ** and whitespace around it are formatting.
        status = found.group(2).replace("*", "").strip()

        if task_id in seen:
            problems.append("%s:%d  `%s` is already defined at line %d — ids are never reused"
                            % (TASKS.name, number, task_id, seen[task_id]))

        seen.setdefault(task_id, number)

        if not status:
            problems.append("%s:%d  `%s` has an EMPTY status cell" % (TASKS.name, number, task_id))

            continue

        # startswith on the raw text, not on a split token: "DONE — REFUTED 2026-07-05" and
        # "DONE, residue re-pointed" are both fine, while "DONEISH" is not.
        if any(status.startswith(word) for word in VOCABULARY):
            continue

        problems.append(
            "%s:%d  `%s` status starts with %r — must begin with one of %s"
            % (TASKS.name, number, task_id, status[:40], ", ".join(VOCABULARY)))

    # An empty scan and a clean file are the same output otherwise, and that confusion is on record here.
    if rows == 0:
        print("no task rows matched in %s — the file, or this script's pattern, is wrong. NOT a pass."
              % TASKS.name)

        return 2

    print("%d task rows checked, %d distinct ids." % (rows, len(seen)))

    if not problems:
        print("Every status begins with one of: %s" % ", ".join(VOCABULARY))

        return 0

    print("\n%d problem(s):" % len(problems))

    for problem in problems:
        print("  " + problem)

    print("\nThe status cell is the only machine-readable field in the registry. Detail belongs AFTER the "
          "vocabulary word: `DONE — REFUTED 2026-07-05, …`.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
