"""Run every LongFact area the timings log does not already record for today.

Lives in Scripts/, not .claude/, because .gitignore excludes .claude wholesale and a helper that must
survive cannot live where version control is not looking -- the same reason lab.py sits here.
"""
import pathlib
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path(r"D:\Overfit")
TESTS = REPO / "Tests"
TIMINGS = TESTS / "bin" / "longfact-timings.log"
HEAVY = ["LanguageModels.Demo", "LanguageModels.LoRA", "LanguageModels.Loading"]
SKIP_DIRS = {"bin", "obj", "test_fixtures", "TestSupport", "Properties"}
TODAY = "2026-08-15"

areas = []
for d in sorted(TESTS.iterdir()):
    if not d.is_dir() or d.name in SKIP_DIRS or d.name.startswith("."):
        continue
    subs = [s for s in sorted(d.iterdir())
            if s.is_dir() and s.name not in SKIP_DIRS and not s.name.startswith(".")]
    if subs and d.name == "LanguageModels":
        areas.extend(f"{d.name}.{s.name}" for s in subs)
        continue
    areas.append(d.name)

log = TIMINGS.read_text(encoding="utf-8", errors="replace") if TIMINGS.exists() else ""
done = {m.group(1) for m in re.finditer(TODAY + r"\S*\s+\S+\s+(\S+)\s", log)}

# Heavy areas last: 79 minutes of timeout in front of everything else is how two earlier attempts
# produced nothing at all.
todo = [a for a in areas if a not in HEAVY and a not in done]
todo += [a for a in HEAVY if a not in done]
print(f"START {time.strftime('%H:%M:%S')}  areas={len(areas)} recorded={len(done & set(areas))} to run={len(todo)}", flush=True)

for i, area in enumerate(todo, 1):
    out = REPO / ".claude" / f"lf-{area}.log"
    print(f"[{i}/{len(todo)}] {area:34s} started {time.strftime('%H:%M:%S')}", flush=True)
    with out.open("wb") as sink:
        r = subprocess.run(
            ["python", str(REPO / "Scripts" / "longfact_gate.py"), "--area", area],
            stdout=sink, stderr=subprocess.STDOUT, cwd=str(REPO),
        )
    text = out.read_bytes().decode("utf-8", errors="replace")
    el = re.search(r"ELAPSED\s+(\S+ min)\s+exit (\d+)", text)
    trx = re.search(r"TRX: total (\d+)\s+passed (\d+)\s+failed (\d+)\s+skipped (\d+)", text)
    line = f"[{i}/{len(todo)}] {area:34s} rc={r.returncode}"
    if el:
        line += f"  {el.group(1)}"
    if trx:
        line += f"  total={trx.group(1)} passed={trx.group(2)} failed={trx.group(3)} skipped={trx.group(4)}"
    print(line, flush=True)
    # `$` after `\S+` NEVER MATCHES on this log. The child writes CRLF, so the position after the test
    # name sits on `\r`, and `$` only matches at end-of-string or immediately before `\n`. The pattern
    # therefore found nothing across five failing areas on 2026-08-15 and the campaign reported 19
    # failures by number alone. Normalise the line endings instead of tightening the pattern.
    for f in re.findall(r"^\s{4}(DevOnBike\.\S+)\s*$", text.replace("\r\n", "\n"), re.M):
        print(f"        FAILED {f}", flush=True)

print(f"CAMPAIGN COMPLETE {time.strftime('%H:%M:%S')}", flush=True)
