"""`XC-92`'s bar, re-runnable: convolution scaling from 1 core to 16, Overfit against ONNX Runtime.

    python D:/Overfit/Scripts/xc92_conv_scaling.py                # 3 sittings, the four numbers
    python D:/Overfit/Scripts/xc92_conv_scaling.py --probes       # instrument and lever checks only
    python D:/Overfit/Scripts/xc92_conv_scaling.py --sittings 1 --slow 6 --fast 4    # quick pass

It drives `Scripts/ProfHarness`, which is the instrument the 2026-08-19 reading came from. Build it
first: `dotnet build Scripts/ProfHarness/ProfHarness.csproj -c Release`.

**It lives here rather than in a scratch file because a per-agent `.claude/do-*.py` is rewritten by the
next task**, and this repository has already lost a helper to `.gitignore` once (see `Scripts/lab.py`).

--------------------------------------------------------------------------------------------------
WHAT THE FOUR NUMBERS ARE, AND WHY THE WHOLE-MODEL NUMBER CANNOT ANSWER THE QUESTION

VGG-16 is roughly a third memory-bound dense layers that NEITHER engine scales — `fc1` measures 1.17x
here against 16 cores. Averaging them into a whole-model figure hides what convolution is doing, which
is how the morning of 2026-08-19 reached the opposite conclusion. So both sides are read PER NODE:
ours from `PROF_NODES=1`, theirs from ONNX Runtime's own profiler behind `PROF_ORT_PROFILE=1`.

--------------------------------------------------------------------------------------------------
FOUR TRAPS, EACH MEASURED ON 2026-08-25 RATHER THAN ASSUMED

1. `OVERFIT_PARALLEL_WORKERS` IS THE WRONG LEVER FOR THIS MEASUREMENT, and it does not fail loudly.
   `Conv2DGemmKernels.ResolveMBlocks` (line 1590) sizes the M-split against
   `OverfitParallel.MaxDegreeOfParallelism`, which is `Environment.ProcessorCount` and NOT the pool's
   worker count. Setting `OVERFIT_PARALLEL_WORKERS=16` on this 16C/32T box shrinks the pool and leaves
   the work decomposition computed against 32, and the 16 threads then roam all 32 logical cores.
   Measured convolution: 18.34 and 21.08 ms on two consecutive runs (15% apart) against 17.27-17.81 ms
   by the affinity route. Scaling comes out 5.85x instead of 7.02x -- a 17% miss, which reads as "the
   bar does not reproduce". The route that reproduces is `PROF_AFFINITY` + `DOTNET_PROCESSOR_COUNT`.

2. THE ONNX RUNTIME PROFILE COVERS THE WARM-UP RUNS TOO. `EnableProfiling` is on from session
   creation, so the file holds `WARMUP_CALLS` runs of tier-0 and first-touch faults before steady
   state. Each node emits exactly one `*_kernel_time` event per `model_run` in file order, so the Nth
   occurrence of a node name is the Nth run and the first `WARMUP_CALLS` are dropped.

3. BOTH PER-NODE PROFILERS COST SOMETHING, AND THE COST IS A FIXED AMOUNT PER CALL, so it is a much
   larger share of the 16-core arm than of the 1-core arm and therefore DEPRESSES both measured
   scaling figures. Measured whole-model, 16 cores: ONNX Runtime 16.58 ms profiling against 15.71 ms
   not profiling (+5.5%); ours 27.50 against 27.13 (+1.4%). How much of that lands INSIDE the per-node
   durations is not established here, so the ratios below are conservative in the direction that
   matters -- the true gap is at least as wide as the one printed.

4. THE `1` IN "1 CORE" IS A PHYSICAL CORE. `MASK16` is `0x55555555` -- the even logical ids, one
   thread per physical core on this part. Running unmasked on all 32 logical processors measures
   17.19 ms convolution against 17.34 masked, so SMT siblings are worth nothing here and are not the
   variable under study.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from machine import quiet_guard

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = os.path.join(REPO, "Scripts", "ProfHarness", "bin", "Release", "net10.0", "ProfHarness.exe")
DEFAULT_MODEL = r"C:\onnxmodels\vgg16.onnx"
SCRATCH = os.path.join(REPO, "Tests", "bin", "xc92")

#: One thread per physical core on a 16C/32T part: the even logical ids.
MASK16 = "55555555"
MASK1 = "1"

#: `ProfHarness` warms up this many calls before steady state. Kept in step with `Program.cs`; the ORT
#: profile trimming is wrong by exactly this many runs if they diverge.
WARMUP_CALLS = 40

#: Recorded 2026-08-19, the figures this script exists to re-measure.
RECORDED = {"ours-1": 125.34, "ours-16": 17.70, "ort-1": 95.68, "ort-16": 7.89,
            "ours-scaling": 7.08, "ort-scaling": 12.12}

NUM = r"([-+]?[0-9]+[.,][0-9]+|[-+]?[0-9]+)"


def _f(text):
    # ProfHarness prints under the machine's culture and this box is pl-PL: "27,201 ms/call".
    return float(text.replace(",", "."))


def _median(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return None
    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2.0


def run_harness(model, env, seconds, timeout=2400):
    """One ProfHarness process. Environment goes through `env=`, never a shell prefix.

    The seconds argument is rounded to a whole number before it is passed. `ProfHarness/Program.cs:57`
    calls `double.Parse(String)` with no `IFormatProvider`, so it reads the machine's culture: on this
    pl-PL box `5.0` throws `FormatException` and the arm dies with no reading, while `5` parses
    everywhere. Reported rather than patched -- editing the instrument during a measurement would
    invalidate the numbers it just produced.
    """
    merged = dict(os.environ)
    merged.update(env)
    started = time.time()
    completed = subprocess.run([EXE, model, "%d" % round(seconds)], cwd=SCRATCH, env=merged,
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=timeout)
    return completed, time.time() - started


def parse_overfit(stdout):
    """Our side: whole-model ms/call plus the per-layer table when `PROF_NODES=1` was set."""
    out = {"procs": None, "ms_per_call": None, "layers": []}
    match = re.search(r"ProcessorCount=(\d+)", stdout)
    if match:
        out["procs"] = int(match.group(1))
    match = re.search(r"= " + NUM + r" ms/call", stdout)
    if match:
        out["ms_per_call"] = _f(match.group(1))
    for line in stdout.splitlines():
        match = re.match(r"\s*\[\s*(\d+)\]\s+(\S+)\s+out=\s*(\d+)\s+" + NUM + r" ms", line)
        if match:
            out["layers"].append({"index": int(match.group(1)), "type": match.group(2),
                                  "out": int(match.group(3)), "ms": _f(match.group(4))})
    out["conv_ms"] = sum(x["ms"] for x in out["layers"] if x["type"] == "ConvLayer")
    out["dense_ms"] = sum(x["ms"] for x in out["layers"] if x["type"] == "LinearLayer")
    out["pool_ms"] = sum(x["ms"] for x in out["layers"] if "Pool" in x["type"])
    return out


def parse_ort(stdout):
    out = {"procs": None, "ms_per_call": None, "profile": None}
    match = re.search(r"ProcessorCount=(\d+)", stdout)
    if match:
        out["procs"] = int(match.group(1))
    match = re.search(r"= " + NUM + r" ms/call", stdout)
    if match:
        out["ms_per_call"] = _f(match.group(1))
    match = re.search(r"ort profile: (\S+\.json)", stdout)
    if match:
        out["profile"] = match.group(1)
    return out


def ort_per_node(profile_name):
    """ONNX Runtime's own per-node budget for ONE steady-state run, warm-up runs dropped.

    `SessionOptions.ProfileOutputPathPrefix` does not take effect on this version -- the file is
    written as `onnxruntime_profile__*.json` whatever prefix is asked for -- so the name is taken
    from the harness's own stdout rather than guessed from a glob.
    """
    path = os.path.join(SCRATCH, os.path.basename(profile_name))
    with open(path, "r", encoding="utf-8") as handle:
        events = json.load(handle)
    runs = sum(1 for e in events if e.get("name") == "model_run")
    seen = {}
    per_op = {}
    per_conv_node = {}
    for event in events:
        if event.get("cat") != "Node" or not str(event.get("name", "")).endswith("_kernel_time"):
            continue
        name = event["name"]
        index = seen.get(name, 0)
        seen[name] = index + 1
        op = event.get("args", {}).get("op_name", "?")
        ms = event["dur"] / 1000.0
        series = per_op.setdefault(op, [])
        while len(series) <= index:
            series.append(0.0)
        series[index] += ms
        if op == "Conv":
            per_conv_node.setdefault(name, []).append(ms)

    def steady(series):
        tail = series[WARMUP_CALLS:]
        return (sum(tail) / len(tail)) if tail else None

    by_op = {op: steady(series) for op, series in per_op.items()}
    return {"runs_in_file": runs, "steady_runs": max(0, runs - WARMUP_CALLS),
            "by_op_ms": by_op, "conv_ms": by_op.get("Conv"),
            "dense_ms": (by_op.get("FusedGemm") or 0.0) + (by_op.get("Gemm") or 0.0),
            "conv_layers_ms": {n: steady(s) for n, s in per_conv_node.items()},
            "profile_file": os.path.basename(profile_name)}


def arms(fast, slow):
    return {
        "ours-1":  ({"PROF_AFFINITY": MASK1,  "DOTNET_PROCESSOR_COUNT": "1",  "PROF_NODES": "1"}, slow),
        "ours-16": ({"PROF_AFFINITY": MASK16, "DOTNET_PROCESSOR_COUNT": "16", "PROF_NODES": "1"}, fast),
        "ort-1":   ({"PROF_ENGINE": "ort", "PROF_ORT_THREADS": "1", "PROF_AFFINITY": MASK1,
                     "DOTNET_PROCESSOR_COUNT": "1", "PROF_ORT_PROFILE": "1"}, slow),
        "ort-16":  ({"PROF_ENGINE": "ort", "PROF_ORT_THREADS": "16", "PROF_AFFINITY": MASK16,
                     "DOTNET_PROCESSOR_COUNT": "16", "PROF_ORT_PROFILE": "1"}, fast),
    }


#: The canary: `OverfitParallel` on balanced, register-resident, memory-free work at 16 cores. Nothing
#: in this experiment can change it, so if it moves between sittings the box moved and not the code.
CANARY = ({"PROF_ENGINE": "pool", "PROF_AFFINITY": MASK16, "DOTNET_PROCESSOR_COUNT": "16"}, 5)

#: Arm order is rotated between sittings. If every A precedes its own B, "this arm is faster" and "the
#: later arm is faster" are the same observation, and a smooth curve out of single readings is more
#: dangerous than a noisy one.
ORDERS = [
    ["ours-1", "ours-16", "ort-1", "ort-16"],
    ["ort-16", "ours-1", "ort-1", "ours-16"],
    ["ort-1", "ours-16", "ours-1", "ort-16"],
]


def measure(model, sittings, fast, slow):
    table = arms(fast, slow)
    results = {"model": model, "mask16": MASK16, "warmup_calls": WARMUP_CALLS,
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sittings": []}
    for index in range(sittings):
        order = ORDERS[index % len(ORDERS)]
        print("\n########## SITTING %d - order %s ##########" % (index + 1, order), flush=True)
        sitting = {"index": index + 1, "order": order, "arms": {}, "canary": []}
        with quiet_guard("XC-92 sitting %d" % (index + 1)) as window:
            env, secs = CANARY
            completed, wall = run_harness(model, env, secs)
            canary = parse_overfit(completed.stdout)
            print("  canary-pre  %.3f ms/call (%.1fs)" % (canary["ms_per_call"], wall), flush=True)
            sitting["canary"].append({"pos": "pre", "ms": canary["ms_per_call"]})

            for arm in order:
                env, secs = table[arm]
                completed, wall = run_harness(model, env, secs)
                if completed.returncode != 0:
                    print("  %s: EXIT %d\n%s\n%s" % (arm, completed.returncode, completed.stdout,
                                                     completed.stderr[-1500:]), flush=True)
                    sitting["arms"][arm] = {"exit": completed.returncode}
                    continue
                if arm.startswith("ours"):
                    data = parse_overfit(completed.stdout)
                    conv = data["conv_ms"]
                else:
                    data = parse_ort(completed.stdout)
                    data["pernode"] = ort_per_node(data["profile"])
                    conv = data["pernode"]["conv_ms"]
                data["exit"] = 0
                sitting["arms"][arm] = data
                print("  %-8s procs=%s whole %s ms  conv %.2f ms  (%.1fs)"
                      % (arm, data["procs"], data["ms_per_call"], conv, wall), flush=True)

            env, secs = CANARY
            completed, wall = run_harness(model, env, secs)
            canary = parse_overfit(completed.stdout)
            print("  canary-post %.3f ms/call (%.1fs)" % (canary["ms_per_call"], wall), flush=True)
            sitting["canary"].append({"pos": "post", "ms": canary["ms_per_call"]})

        sitting["quiet"] = window.quiet
        sitting["quiet_verdict"] = str(window.verdict)
        results["sittings"].append(sitting)
    return results


def conv_of(sitting, arm):
    entry = sitting["arms"].get(arm)
    if not entry or entry.get("exit") != 0:
        return None
    return entry["conv_ms"] if arm.startswith("ours") else entry["pernode"]["conv_ms"]


def report(results):
    print("\n########## CONVOLUTION ms (sum of Conv nodes, steady state) ##########")
    print("%-9s %3s %9s %9s %9s   %9s   all readings" % ("arm", "n", "min", "median", "max", "recorded"))
    series = {}
    for arm in ["ours-1", "ours-16", "ort-1", "ort-16"]:
        values = [v for v in (conv_of(s, arm) for s in results["sittings"]) if v is not None]
        series[arm] = values
        if values:
            print("%-9s %3d %9.2f %9.2f %9.2f   %9.2f   %s"
                  % (arm, len(values), min(values), _median(values), max(values), RECORDED[arm],
                     ", ".join("%.2f" % v for v in values)))
        else:
            print("%-9s   0   NO DATA" % arm)

    print("\n########## SCALING, 1 core -> 16 cores ##########")
    for engine, one, sixteen, key in [("Overfit", "ours-1", "ours-16", "ours-scaling"),
                                      ("ONNX Runtime", "ort-1", "ort-16", "ort-scaling")]:
        if not series[one] or not series[sixteen]:
            continue
        ratio = _median(series[one]) / _median(series[sixteen])
        print("  %-13s %8.2f -> %6.2f ms = %6.2fx  (recorded %5.2fx, delta %+.1f%%)  "
              "worst-case range %.2f..%.2fx"
              % (engine, _median(series[one]), _median(series[sixteen]), ratio, RECORDED[key],
                 100.0 * (ratio - RECORDED[key]) / RECORDED[key],
                 min(series[one]) / max(series[sixteen]), max(series[one]) / min(series[sixteen])))

    if all(series.values()):
        per_core = _median(series["ours-1"]) / _median(series["ort-1"])
        at16 = _median(series["ours-16"]) / _median(series["ort-16"])
        scaling = (_median(series["ort-1"]) / _median(series["ort-16"])) / \
                  (_median(series["ours-1"]) / _median(series["ours-16"]))
        print("\n  decomposition: per-core work %.2fx  x  scaling %.2fx  =  %.2fx   (16-core gap %.2fx)"
              % (per_core, scaling, per_core * scaling, at16))

    canaries = [c["ms"] for s in results["sittings"] for c in s["canary"] if c["ms"] is not None]
    if canaries:
        print("\n########## CANARY (OverfitParallel, balanced register-only work, 16 cores) ##########")
        print("  n=%d  min %.3f  median %.3f  max %.3f  spread %.1f%%"
              % (len(canaries), min(canaries), _median(canaries), max(canaries),
                 100.0 * (max(canaries) - min(canaries)) / _median(canaries)))

    print("\n########## QUIET GUARD ##########")
    for s in results["sittings"]:
        print("  sitting %d: quiet=%s" % (s["index"], s["quiet"]))
    quiet = [s for s in results["sittings"] if s["quiet"]]
    if quiet and len(quiet) != len(results["sittings"]):
        print("  -- quiet sittings only --")
        report({"sittings": quiet})


def per_layer(results):
    # A sitting with a failed arm carries no table. Skipping it here rather than crashing keeps the
    # arms that DID run readable; the failure itself is already printed and recorded above.
    usable = [s for s in results["sittings"]
              if all(s["arms"].get(a, {}).get("exit") == 0
                     for a in ("ours-1", "ours-16", "ort-1", "ort-16"))]
    if not usable:
        print("\n(no sitting has all four arms; per-layer tables skipped)")
        return
    results = {"sittings": usable}
    print("\n########## OUR PER-LAYER SCALING (median across sittings) ##########")
    one, sixteen = {}, {}
    for sitting in results["sittings"]:
        for arm, bucket in (("ours-1", one), ("ours-16", sixteen)):
            for layer in sitting["arms"][arm]["layers"]:
                bucket.setdefault(layer["index"], {"type": layer["type"], "ms": []})
                bucket[layer["index"]]["ms"].append(layer["ms"])
    print("%-4s %-20s %10s %10s %9s" % ("idx", "type", "1 core", "16 cores", "scaling"))
    for index in sorted(one):
        a, b = _median(one[index]["ms"]), _median(sixteen[index]["ms"])
        print("%-4d %-20s %10.2f %10.2f %8.2fx" % (index, one[index]["type"], a, b, (a / b) if b else 0.0))

    print("\n########## THEIR PER-LAYER CONV SCALING (median across sittings) ##########")
    tone, tsix = {}, {}
    for sitting in results["sittings"]:
        for arm, bucket in (("ort-1", tone), ("ort-16", tsix)):
            for name, ms in sitting["arms"][arm]["pernode"]["conv_layers_ms"].items():
                bucket.setdefault(name, []).append(ms)
    print("%-52s %8s %8s %9s" % ("node", "1 thr", "16 thr", "scaling"))
    for name in sorted(tone, key=lambda k: -_median(tone[k])):
        a, b = _median(tone[name]), _median(tsix.get(name, [0]))
        print("%-52s %8.2f %8.2f %8.2fx" % (name[-52:], a, b, (a / b) if b else 0.0))


def probes(model, fast, slow):
    """The four checks that decide how to read the table above. See the module docstring."""
    print("\n########## PROBES ##########", flush=True)
    rows = []
    with quiet_guard("XC-92 probes") as window:
        env, secs = CANARY
        completed, _ = run_harness(model, env, secs)
        print("  canary-pre  %.3f ms" % parse_overfit(completed.stdout)["ms_per_call"], flush=True)
        for label, env, secs in [
            ("ort-16 no profiler", {"PROF_ENGINE": "ort", "PROF_ORT_THREADS": "16",
                                    "PROF_AFFINITY": MASK16, "DOTNET_PROCESSOR_COUNT": "16"}, fast),
            ("ours-16 no PROF_NODES", {"PROF_AFFINITY": MASK16, "DOTNET_PROCESSOR_COUNT": "16"}, fast),
            ("WORKERS=16, no affinity", {"OVERFIT_PARALLEL_WORKERS": "16", "PROF_NODES": "1"}, fast),
            ("WORKERS=1, no affinity", {"OVERFIT_PARALLEL_WORKERS": "1", "PROF_NODES": "1"}, slow),
            ("no lever at all (32 logical)", {"PROF_NODES": "1"}, fast),
        ]:
            completed, wall = run_harness(model, env, secs)
            data = parse_overfit(completed.stdout)
            conv = ("%.2f" % data["conv_ms"]) if data["layers"] else "n/a"
            print("  %-30s exit=%d procs=%s whole %s ms  conv %s  (%.1fs)"
                  % (label, completed.returncode, data["procs"], data["ms_per_call"], conv, wall),
                  flush=True)
            rows.append({"label": label, "procs": data["procs"], "ms": data["ms_per_call"],
                         "conv": data["conv_ms"] if data["layers"] else None})
        env, secs = CANARY
        completed, _ = run_harness(model, env, secs)
        print("  canary-post %.3f ms" % parse_overfit(completed.stdout)["ms_per_call"], flush=True)
    print("  quiet:", window.quiet)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("OVERFIT_CNN_ONNX", DEFAULT_MODEL))
    parser.add_argument("--sittings", type=int, default=3)
    parser.add_argument("--fast", type=float, default=8.0, help="steady-state seconds, 16-core arms")
    parser.add_argument("--slow", type=float, default=12.0, help="steady-state seconds, 1-core arms")
    parser.add_argument("--probes", action="store_true", help="run only the instrument/lever probes")
    parser.add_argument("--out", default=os.path.join(SCRATCH, "xc92-results.json"))
    args = parser.parse_args(argv)

    if not os.path.exists(EXE):
        print("ProfHarness not built. Run:\n"
              "  dotnet build Scripts/ProfHarness/ProfHarness.csproj -c Release")
        return 2
    if not os.path.exists(args.model):
        print("model not found: %s" % args.model)
        return 2
    os.makedirs(SCRATCH, exist_ok=True)

    if args.probes:
        probes(args.model, args.fast, args.slow)
        return 0

    results = measure(args.model, args.sittings, args.fast, args.slow)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    report(results)
    per_layer(results)
    print("\nresults json: %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
