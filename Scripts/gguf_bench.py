"""End-to-end GGUF prefill/decode throughput, measured the same way on every engine.

    python D:/Overfit/Scripts/gguf_bench.py --validate
    python D:/Overfit/Scripts/gguf_bench.py --engine llamacpp --phase tg --tokens 128
    python D:/Overfit/Scripts/gguf_bench.py --engine overfit  --phase pp --tokens 512

**Why this exists.** `XC-76`: sixty-plus classes in ``Sources/Benchmark`` and not one loads a real GGUF and
measures end-to-end decode, so every llama.cpp comparison this project has published came from something
that is not in the repository. This module is the instrument. The benchmark class it drives on the Overfit
side is ``Sources/Benchmark/GgufEndToEndThroughputBenchmark.cs``.

**The quantities are llama.cpp's, deliberately.** ``pp512`` is the tokens-per-second of processing a
512-token prompt in one go; ``tg128`` is the tokens-per-second of generating 128 tokens one at a time from
an empty context. Nothing here invents a metric, because a metric only this repository computes cannot be
compared with anything.

**The harness owns the clock, and that is the whole point.** On 2026-08-20 this project compared itself with
another engine using that engine's mature CLI on one side and a harness written that morning on the other.
The harness had two defects inside two hours and every suspicious number came through it. An asymmetric
comparison puts all the uncertainty on your side of the table. So the primary protocol here is applied
**identically to both engines** and uses **neither engine's internal timer**:

  1. Run the engine's benchmark host at several repetition counts ``r`` of the *same* work unit.
  2. Wall-clock each process from outside.
  3. Least-squares fit ``t(r) = fixed + r * work``. The slope is the work unit's cost; the intercept
     absorbs process start, model load and warm-up, which is why no assumption about them is needed.
  4. ``tokens / work`` is the rate.

**Repetition counts are run in a ROTATED order, never ascending.** A single ascending sweep is how the
2026-08-20 context measurement produced a smooth curve out of drift — the smoothness came from the ordering,
and it suppressed suspicion instead of raising it. Rotating means drift over the campaign cannot align with
``r``.

**The fit reports R-squared, and a poor one voids the reading rather than rounding it.** ``t = fixed + r*w``
is a model, and if it does not hold the slope means nothing. Three points are the minimum that can disagree
with a straight line at all.

**The validity arm is the reason to trust anything else here.** ``--validate`` drives llama.cpp through the
protocol above and compares the harness's own numbers with what ``llama-bench`` reports for itself in the
same session. If the harness cannot reproduce a mature tool's figure for that tool, nothing it later says
about Overfit is worth reading. It is a gate, not a diagnostic: run it before every campaign.

**What this module does NOT hold still.** The machine mutex ``Global\\DevOnBike.Overfit.MachineMeasurement``
is held across llama.cpp windows and **released around every Overfit child**, because
``Sources/Benchmark/Program.cs`` takes the same mutex itself and exits with code 2 if it cannot. So a build
started at exactly the wrong moment can land inside an Overfit window. ``quiet_guard`` is what catches that
after the fact; this module does not prevent it, and the guard was not weakened to make it prevent it.
"""

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from machine import quiet_guard  # noqa: E402

#: llama.cpp's own benchmark host. Built outside the repository; the path is recorded rather than searched
#: for, because "whatever llama.cpp a reader would install today" is not a citable reference.
LLAMA_BENCH = r"D:\llamacpp-tmp\build-avx2\bin\Release\llama-bench.exe"

#: The Overfit benchmark host, in its non-BenchmarkDotNet driver mode. Release only: a Debug build is a
#: different engine.
OVERFIT_BENCH = r"D:\Overfit\Sources\Benchmark\bin\Release\net10.0\Benchmarks.exe"

#: The model every published Overfit-versus-llama.cpp figure is about.
DEFAULT_MODEL = r"C:\qwen3b\qwen.q4km.gguf"

#: Physical cores on this box, and llama-bench's own default here. Thread count is a lever and not a
#: detail: `ROADMAP-COMPLETED.md:1174` records llama-bench choosing 16 over the machine's 32 and beating
#: the 32-thread run, so a ratio taken at one thread count is a statement about that configuration.
#:
#: **AND THE TWO ENGINES DO NOT PEAK AT THE SAME VALUE, so this default silently handicaps Overfit's
#: prefill.** Measured 2026-08-25 through the driver, one reading per point (a diagnostic, not a fit):
#:
#:     -t     8      12      16      24      32     BenchmarkDotNet, no override
#:   pp512  171.15  228.78  250.61  271.59  324.62  320.00 t/s
#:   tg128   23.40   28.04   27.88   27.93   28.00   28.76 t/s
#:
#: ``pp512`` climbs to the machine's 32 logical CPUs — **+30% from 16 to 32** — while ``tg128`` is flat
#: from 12 upward because the decode dispatch caps itself at 10 workers whatever is requested. At the
#: shared ``-t 16`` the prefill ratio came out 1.438x; against each side's own best it is about 1.21x.
#: **Sweep both sides and report each side's best, or the ratio measures a configuration choice.** The
#: llama.cpp side of that sweep has NOT been re-run here; 16 is taken from its own default and from the
#: ROADMAP entry above.
DEFAULT_THREADS = 16

#: Repetition counts the slope is fitted through. Three points is the minimum that can disagree with a
#: straight line; the spacing is even so no single point dominates the fit.
DEFAULT_REPS_POINTS = (2, 4, 6)

#: Repetition counts per phase, and **the spread is chosen from the conditioning of the fit, not from
#: taste**. The slope is a difference of wall times, so the quantity that decides its precision is the span
#: ``(r_max - r_min) * work`` against the run-to-run wobble of the intercept. Measured here on 2026-08-25
#: with ``(2, 4, 6)`` for both phases:
#:
#:   - ``tg128``: work 4.03 s, span 16.1 s, intercept 1.27-1.47 s. Three fits landed 31.67 / 31.80 / 31.90
#:     t/s — a spread of **0.4%**.
#:   - ``pp512``: work 1.30 s, span 5.2 s, intercept 2.43-2.71 s — the intercept is TWICE the work unit and
#:     moved 0.28 s between fits, which is 21% of one work unit. Three fits landed 373.6 / 401.0 / 406.1
#:     t/s — a spread of **4.4%**, against llama-bench's own 1.3%.
#:
#: So ``pp`` gets a four-times wider span. This costs wall time and buys resolving power, which is the only
#: trade worth making in an instrument.
PHASE_REPS_POINTS = {
    "pp": (4, 12, 20),
    "tg": (2, 4, 6),
}

#: How many independent fits make up a campaign. A number measured once is not a fact.
DEFAULT_CAMPAIGN = 3

_MUTEX_NAME = r"Global\DevOnBike.Overfit.MachineMeasurement"
_WAIT_OBJECT_0 = 0x00000000
_WAIT_ABANDONED = 0x00000080
_WAIT_TIMEOUT = 0x00000102


class MachineLock:
    """The machine-measurement mutex, held across a campaign and released around Overfit children.

    ``Sources/Benchmark/Program.cs`` takes this same name and refuses to start when it is held, which is
    correct and is not worked around here: the lock is released for the duration of an Overfit child and
    re-taken afterwards. Everything between children — including every llama.cpp window — is covered.
    """

    def __init__(self):
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateMutexW.restype = ctypes.c_void_p
        self._handle = None
        self.held = False

    def _open(self):
        if self._handle is None:
            self._handle = self._kernel32.CreateMutexW(None, False, _MUTEX_NAME)

            if not self._handle:
                raise OSError(ctypes.get_last_error(), "CreateMutexW failed")

    def acquire(self, timeout_ms=0):
        self._open()
        status = self._kernel32.WaitForSingleObject(ctypes.c_void_p(self._handle), timeout_ms)

        # An abandoned mutex means the previous holder died without releasing it, so nobody is measuring.
        self.held = status in (_WAIT_OBJECT_0, _WAIT_ABANDONED)

        return self.held

    def release(self):
        if self.held:
            self._kernel32.ReleaseMutex(ctypes.c_void_p(self._handle))
            self.held = False


def our_side(model):
    """What `docs/measured-baselines.md:472` says a citable ratio needs, for OUR half of it.

    That row records the llama.cpp decode baseline as reconstructed from the local clone's **reflog** —
    *"evidence about the checkout rather than about the run"*. The other side's version is captured here
    from llama-bench's own `build_commit`; this function captures the same for Overfit, so neither half of
    a ratio has to be recovered afterwards.

    **The repack sidecar is part of the identity of the run, not a footnote.** `GgufLlamaLoader` opens
    `<model>.repack` unconditionally when it exists, with no switch and no announcement, and llama.cpp has
    no equivalent file. A row that does not say whether it was present cannot be placed.
    """
    from provenance import snapshot

    record = snapshot("gguf_bench")
    sidecar = model + ".repack"

    return {
        "head": record.head,
        "dirty": record.dirty,
        "stale": record.stale,
        "usable": record.usable,
        "assemblies": [{"name": item.name, "sha256": item.sha256, "path": item.path}
                       for item in record.assemblies],
        "repack_sidecar": sidecar,
        "repack_sidecar_present": os.path.exists(sidecar),
        "repack_sidecar_bytes": os.path.getsize(sidecar) if os.path.exists(sidecar) else 0,
    }


def sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)

    return digest.hexdigest()


def llamacpp_command(model, phase, tokens, reps, threads):
    """``llama-bench`` runs one test per (n_prompt, n_gen) pair; zero on the other side isolates a phase.

    ``phase == "mixed"`` is the shape the published reference came from — ``-p 512 -n 128`` in ONE process,
    which llama-bench splits into two tests internally. It is kept because a phase-isolated run is a
    DIFFERENT experiment from the one the reference came from: two model loads instead of one, and a decode
    arm that starts from a cold cache rather than after a 512-token prefill. Measuring the split shape
    against a mixed-shape reference cannot tell a harness defect from an invocation difference.
    """
    if phase == "mixed":
        return [
            LLAMA_BENCH, "-m", model, "-t", str(threads),
            "-p", "512", "-n", "128", "-r", str(reps), "-o", "json",
        ]

    return [
        LLAMA_BENCH,
        "-m", model,
        "-t", str(threads),
        "-p", str(tokens if phase == "pp" else 0),
        "-n", str(tokens if phase == "tg" else 0),
        "-r", str(reps),
        "-o", "json",
    ]


def overfit_command(model, phase, tokens, reps, threads):
    """The Overfit benchmark host's driver mode, which emits the same JSON fields llama-bench does."""
    return [
        OVERFIT_BENCH,
        "--gguf-bench",
        "-m", model,
        "-t", str(threads),
        "-p", str(tokens if phase == "pp" else 0),
        "-n", str(tokens if phase == "tg" else 0),
        "-r", str(reps),
        "-o", "json",
    ]


ENGINES = {
    "llamacpp": llamacpp_command,
    "overfit": overfit_command,
}


def timed_run(command, env=None):
    """Wall-clock a process from outside it. This clock is the harness's own and is the primary instrument."""
    start = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)
    elapsed = time.perf_counter() - start

    if proc.returncode != 0:
        raise RuntimeError(
            f"{os.path.basename(command[0])} exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}")

    return elapsed, proc.stdout, proc.stderr


def parse_engine_rows(stdout):
    """Every test row the engine printed. One for an isolated phase, two for the mixed shape."""
    start = stdout.find("[")
    end = stdout.rfind("]")

    if start < 0 or end < 0:
        raise RuntimeError(f"no JSON array in engine output:\n{stdout[-4000:]}")

    rows = json.loads(stdout[start:end + 1])

    if not rows:
        raise RuntimeError(f"engine printed an empty test array:\n{stdout[-4000:]}")

    return rows


def parse_engine_json(stdout):
    """The engine's OWN reported figure for a single-phase run, never substituted for the harness's."""
    rows = parse_engine_rows(stdout)

    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one test row, got {len(rows)}")

    return rows[0]


def engine_own_seconds(stdout):
    """Seconds of work the engine says it timed in one repetition, summed over the rows it printed.

    Summed rather than averaged because the mixed shape's repetition IS both tests: one ``pp512`` and one
    ``tg128``. Averaging would compare a repetition against half of itself.
    """
    return sum(row["avg_ns"] for row in parse_engine_rows(stdout)) / 1e9


def least_squares(points):
    """Fits ``t = fixed + r * work`` and returns (work, fixed, r_squared).

    The intercept is the part nobody has to model: process start, model load and the engine's own warm-up
    are all constant in ``r`` and drop out of the slope. R-squared is returned because a slope from a fit
    that does not describe its own points is not a measurement of anything.
    """
    n = len(points)
    mean_r = sum(r for r, _ in points) / n
    mean_t = sum(t for _, t in points) / n

    covariance = sum((r - mean_r) * (t - mean_t) for r, t in points)
    variance = sum((r - mean_r) ** 2 for r, _ in points)
    work = covariance / variance
    fixed = mean_t - work * mean_r

    residual = sum((t - (fixed + work * r)) ** 2 for r, t in points)
    total = sum((t - mean_t) ** 2 for _, t in points)
    r_squared = 1.0 - residual / total if total > 0 else 0.0

    return work, fixed, r_squared


def one_fit(engine, model, phase, tokens, threads, reps_points, rotation, env=None, verbose=True):
    """One independent estimate of the rate, from the harness's clock alone.

    ``rotation`` shifts the order the repetition counts are run in. Ascending order is how drift over a
    campaign gets to wear the costume of the variable, which is exactly the shape that produced a smooth
    and entirely false context curve here on 2026-08-20.

    **The engine's own per-repetition timing is also collected, and it is a cross-check, never the result.**
    Both engines print ``avg_ns`` for the same processes this function is wall-clocking, so the slope and
    the engine's own timer can be compared **inside one session** — which removes the drift that separates
    a slope measured now from a reference measured earlier. The slope must be at least the engine's timed
    work, because a process cannot grow by less per repetition than the work it claims to have timed; a
    ratio below 1.0 means the two are not describing the same repetitions.
    """
    build = ENGINES[engine]
    order = list(reps_points[rotation:]) + list(reps_points[:rotation])
    points = []
    engine_own = []

    for reps in order:
        command = build(model, phase, tokens, reps, threads)
        elapsed, stdout, _ = timed_run(command, env=env)
        points.append((reps, elapsed))

        try:
            seconds = engine_own_seconds(stdout)
            engine_own.append(seconds)
            own = f"  engine's own {seconds:7.4f} s/rep"
        except (RuntimeError, ValueError, KeyError):
            own = "  engine's own    n/a"

        if verbose:
            print(f"      r={reps:<2d} wall {elapsed:8.3f} s{own}")

    points.sort()
    work, fixed, r_squared = least_squares(points)
    engine_work = statistics.mean(engine_own) if engine_own else None

    return {
        "order": order,
        "points": points,
        "work_seconds": work,
        "fixed_seconds": fixed,
        "r_squared": r_squared,
        "tokens_per_second": tokens / work if work > 0 else float("nan"),
        "engine_own_work_seconds": engine_work,
        "engine_own_tokens_per_second": tokens / engine_work if engine_work else None,
        "slope_over_engine_own": work / engine_work if engine_work else None,
    }


def campaign(engine, model, phase, tokens, threads,
             reps_points=DEFAULT_REPS_POINTS, repeats=DEFAULT_CAMPAIGN, env=None, lock=None):
    """``repeats`` independent fits, rotated, with the mutex released around Overfit children."""
    fits = []

    for index in range(repeats):
        print(f"    fit {index + 1}/{repeats}  (rotation {index % len(reps_points)})")

        if engine == "overfit" and lock is not None:
            lock.release()

        try:
            fits.append(one_fit(engine, model, phase, tokens, threads,
                                reps_points, index % len(reps_points), env=env))
        finally:
            if engine == "overfit" and lock is not None:
                lock.acquire()

        last = fits[-1]
        print(f"      -> {last['tokens_per_second']:8.2f} t/s   "
              f"fixed {last['fixed_seconds']:.3f} s   R2 {last['r_squared']:.5f}")

    rates = [fit["tokens_per_second"] for fit in fits]

    return {
        "engine": engine,
        "phase": phase,
        "tokens": tokens,
        "threads": threads,
        "fits": fits,
        "mean": statistics.mean(rates),
        "stdev": statistics.stdev(rates) if len(rates) > 1 else 0.0,
        "min": min(rates),
        "max": max(rates),
        "worst_r_squared": min(fit["r_squared"] for fit in fits),
    }


def native_reference(model, threads, prompt_tokens, gen_tokens, reps):
    """llama-bench's own reported figures, taken in the same session as the harness's.

    Comparing against a number from an hour ago compares two boxes; comparing against one taken now
    compares two instruments, which is the question.
    """
    command = [
        LLAMA_BENCH, "-m", model, "-t", str(threads),
        "-p", str(prompt_tokens), "-n", str(gen_tokens), "-r", str(reps), "-o", "json",
    ]
    elapsed, stdout, _ = timed_run(command)
    start = stdout.find("[")
    rows = json.loads(stdout[start:stdout.rfind("]") + 1])
    by_phase = {}

    for row in rows:
        phase = "pp" if row["n_gen"] == 0 else "tg"
        samples = row["samples_ns"]
        by_phase[phase] = {
            "row": row,
            "tokens": row["n_prompt"] if phase == "pp" else row["n_gen"],
            "avg_ts": row["avg_ts"],
            "stddev_ts": row["stddev_ts"],
            "samples_ns": samples,
            # Recomputed from the raw samples so a parsing or unit error in this module shows up as a
            # disagreement with the engine's own arithmetic rather than as a plausible number.
            "recomputed_ts": statistics.mean(
                [(row["n_prompt"] if phase == "pp" else row["n_gen"]) / (ns / 1e9) for ns in samples]),
        }

    return elapsed, by_phase


def validate(model, threads, campaign_repeats, reps_points, native_reps):
    """The gate: can this harness reproduce llama.cpp's own figure for llama.cpp?

    Three arms, and they fail differently.

      * **arithmetic** — the harness's own tokens-per-second from llama-bench's raw ``samples_ns`` against
        llama-bench's reported ``avg_ts``. Catches a unit or parsing error and nothing else.
      * **fixture** — model path, size and sha256, thread count, build commit. Catches a comparison of two
        different things, which is the failure that looks most like a result.
      * **clock** — the harness's slope protocol, using none of llama.cpp's timing, against llama-bench's
        own figure taken in the same session. This is the arm that licenses the instrument.

    **The two instruments are INTERLEAVED, alternating which goes first.** The first version of this
    function ran the native reference once, at the start, and all the harness fits after it. That is
    all-A-then-all-B, and it produced a real artefact on 2026-08-25: the harness read ``tg128`` **+0.87%**
    above llama-bench and, arithmetically, a marginal cost per repetition of 4.026 s against the 4.061 s
    llama-bench claimed to have timed *inside the same process* — which is impossible for identical work,
    so it was drift and not a difference between the instruments. The ``pp512`` arm showed the same drift
    undisguised: three consecutive fits measured the work unit at 1.3706, 1.2767 and 1.2609 s, **8% over
    one campaign, monotonically**. Alternating the order is what stops "the harness is faster" and "later
    is faster" being the same observation.

    **llama-bench's own ``stddev_ts`` is NOT the reproducibility of its figure, and using it as the pass
    band is a category error.** It is the spread of consecutive samples inside one warm process, and for
    ``tg128`` it came out at 0.03 t/s — 0.11%. The same tool, same box, same model, same thread count, one
    hour apart, reported 30.97 and then 31.52 t/s: **+1.78%, fifty times its own reported sd**. So the band
    used here is built from the run-to-run spread of both instruments measured in this session, and the
    intra-process figure is printed beside it, labelled, for context only.
    """
    print("=" * 96)
    print("VALIDITY ARM — the harness measures llama.cpp, llama.cpp measures llama.cpp")
    print("=" * 96)

    digest = sha256(model)
    print(f"  model      {model}")
    print(f"  size       {os.path.getsize(model)} bytes")
    print(f"  sha256     {digest}")
    print(f"  threads    {threads}")
    print(f"  schedule   {campaign_repeats} rounds, native and harness INTERLEAVED, order alternating")

    results = {
        "model": model,
        "sha256": digest,
        "size": os.path.getsize(model),
        "threads": threads,
        "rounds": [],
    }

    phases = (("pp", 512), ("tg", 128))
    native_runs = {"pp": [], "tg": []}
    native_intra = {"pp": [], "tg": []}
    harness_runs = {"pp": [], "tg": []}
    harness_fits = {"pp": [], "tg": []}
    quiet_all = True

    for index in range(campaign_repeats):
        native_first = index % 2 == 0
        order = ["native", "harness"] if native_first else ["harness", "native"]
        print()
        print(f"  round {index + 1}/{campaign_repeats}  order: {' then '.join(order)}")
        round_record = {"index": index, "order": order}

        for step in order:
            if step == "native":
                with quiet_guard(f"round {index + 1} native llama-bench") as window:
                    _, native = native_reference(model, threads, 512, 128, native_reps)

                quiet_all = quiet_all and window.quiet
                round_record["native_quiet"] = window.quiet
                round_record["native"] = {}

                for phase, tokens in phases:
                    entry = native[phase]
                    native_runs[phase].append(entry["avg_ts"])
                    native_intra[phase].append(entry["stddev_ts"])
                    drift = 100.0 * (entry["recomputed_ts"] - entry["avg_ts"]) / entry["avg_ts"]
                    round_record["native"][phase] = {
                        "avg_ts": entry["avg_ts"],
                        "stddev_ts": entry["stddev_ts"],
                        "recomputed_ts": entry["recomputed_ts"],
                        "samples_ns": entry["samples_ns"],
                    }
                    print(f"    native  {phase}{tokens:<4d} {entry['avg_ts']:8.2f} t/s "
                          f"(intra-process sd {entry['stddev_ts']:.2f}) "
                          f"| harness arithmetic on its raw samples {entry['recomputed_ts']:8.2f} t/s "
                          f"({drift:+.4f}%)")

                results.setdefault("build_commit", native["pp"]["row"]["build_commit"])
                results.setdefault("build_number", native["pp"]["row"]["build_number"])
                results.setdefault("cpu_info", native["pp"]["row"]["cpu_info"].strip())
                continue

            round_record["harness"] = {}

            for phase, tokens in phases:
                points = reps_points[phase]
                print(f"    harness {phase}{tokens}, slope fit over r={list(points)}")

                with quiet_guard(f"round {index + 1} harness {phase}{tokens}") as window:
                    fit = one_fit("llamacpp", model, phase, tokens, threads,
                                  points, index % len(points))

                quiet_all = quiet_all and window.quiet
                harness_runs[phase].append(fit["tokens_per_second"])
                harness_fits[phase].append(fit)
                round_record["harness"][phase] = {
                    "tokens_per_second": fit["tokens_per_second"],
                    "work_seconds": fit["work_seconds"],
                    "fixed_seconds": fit["fixed_seconds"],
                    "r_squared": fit["r_squared"],
                    "order": fit["order"],
                    "points": fit["points"],
                    "engine_own_work_seconds": fit["engine_own_work_seconds"],
                    "engine_own_tokens_per_second": fit["engine_own_tokens_per_second"],
                    "slope_over_engine_own": fit["slope_over_engine_own"],
                    "quiet": window.quiet,
                }
                ratio = fit["slope_over_engine_own"]
                print(f"      -> {fit['tokens_per_second']:8.2f} t/s   "
                      f"work {fit['work_seconds']:.4f} s   fixed {fit['fixed_seconds']:.3f} s   "
                      f"R2 {fit['r_squared']:.5f}   quiet={window.quiet}")
                print(f"         same-session cross-check: slope / engine's own timer = "
                      f"{ratio:.4f}" if ratio else "         same-session cross-check: n/a")

        results["rounds"].append(round_record)

    print()
    print("  " + "-" * 92)
    print(f"  build {results.get('build_commit')} ({results.get('build_number')})  "
          f"on {results.get('cpu_info')}")
    results["quiet"] = quiet_all
    results["verdict"] = {}

    for phase, tokens in phases:
        harness_mean = statistics.mean(harness_runs[phase])
        harness_sd = statistics.stdev(harness_runs[phase]) if len(harness_runs[phase]) > 1 else 0.0
        native_mean = statistics.mean(native_runs[phase])
        native_sd = statistics.stdev(native_runs[phase]) if len(native_runs[phase]) > 1 else 0.0
        intra = statistics.mean(native_intra[phase])
        delta = harness_mean - native_mean
        band = native_sd + harness_sd
        worst_r2 = min(fit["r_squared"] for fit in harness_fits[phase])

        print()
        print(f"  {phase}{tokens}")
        print(f"    harness clock     {harness_mean:8.2f} +/- {harness_sd:5.2f} t/s  "
              f"(run-to-run, n={len(harness_runs[phase])}; "
              f"min {min(harness_runs[phase]):.2f} max {max(harness_runs[phase]):.2f}; "
              f"worst R2 {worst_r2:.5f})")
        print(f"    llama-bench       {native_mean:8.2f} +/- {native_sd:5.2f} t/s  "
              f"(run-to-run, n={len(native_runs[phase])}; "
              f"min {min(native_runs[phase]):.2f} max {max(native_runs[phase]):.2f})")
        print(f"    llama-bench intra-process sd {intra:.2f} t/s "
              f"({100.0 * intra / native_mean:.2f}%) — NOT the reproducibility, context only")
        print(f"    delta             {delta:+8.2f} t/s = {100.0 * delta / native_mean:+.2f}%   "
              f"| run-to-run band +/-{band:.2f}")

        within = abs(delta) <= band
        print(f"    verdict           {'PASS' if within else 'FAIL'}")

        results["verdict"][phase] = {
            "harness_mean": harness_mean,
            "harness_stdev": harness_sd,
            "harness_runs": harness_runs[phase],
            "native_mean": native_mean,
            "native_stdev": native_sd,
            "native_runs": native_runs[phase],
            "native_intra_process_stdev": intra,
            "delta": delta,
            "delta_percent": 100.0 * delta / native_mean,
            "band": band,
            "within_combined": within,
            "worst_r_squared": worst_r2,
        }

    return results


def mixed_anchor(model, threads, repeats, reps_points, native_reps):
    """Like-for-like: the harness's clock against llama-bench's, on llama-bench's OWN invocation.

    **Why this exists and why the split-phase arm alone is not enough.** The published reference came from
    one process, ``-p 512 -n 128 -t 16 -r 3``, which llama-bench splits into two tests internally. The
    harness's normal protocol runs each phase in its own process. That is arguably the better instrument,
    but it is a **different experiment**: two model loads instead of one, and a decode arm starting from a
    cold cache rather than after a 512-token prefill. A 5% disagreement between the split arm and the
    mixed reference could be a harness defect or an invocation difference, and this piece exists to tell
    those apart.

    So the quantity here is **seconds to complete one repetition of the mixed shape** — one ``pp512`` plus
    one ``tg128``, in one process. The harness gets it from the slope; llama-bench gets it by summing the
    ``avg_ns`` of the two rows it prints. Tokens per second is deliberately NOT reported: a rate over 640
    tokens of two different kinds is a mongrel and nobody could compare it with anything.

    The split-versus-mixed difference is then reported as its own line. If it is real it is a finding about
    the workload, not about this code.
    """
    print("=" * 96)
    print("MIXED-SHAPE ANCHOR — the reference's own invocation, measured by both clocks")
    print("=" * 96)
    print(f"  llama-bench -m {os.path.basename(model)} -p 512 -n 128 -t {threads} -r {native_reps}")
    print("  quantity: SECONDS for one repetition of (pp512 + tg128) in one process")

    harness_runs = []
    native_runs = []
    split_runs = []
    quiet_all = True
    rounds = []

    for index in range(repeats):
        native_first = index % 2 == 0
        order = ["native", "harness"] if native_first else ["harness", "native"]
        print()
        print(f"  round {index + 1}/{repeats}  order: {' then '.join(order)}")
        record = {"index": index, "order": order}

        for step in order:
            if step == "native":
                with quiet_guard(f"round {index + 1} native mixed") as window:
                    _, stdout, _ = timed_run(
                        llamacpp_command(model, "mixed", 0, native_reps, threads))

                quiet_all = quiet_all and window.quiet
                rows = parse_engine_rows(stdout)
                seconds = sum(row["avg_ns"] for row in rows) / 1e9
                native_runs.append(seconds)
                record["native_seconds"] = seconds
                record["native_rows"] = [
                    {"n_prompt": row["n_prompt"], "n_gen": row["n_gen"],
                     "avg_ns": row["avg_ns"], "avg_ts": row["avg_ts"],
                     "stddev_ts": row["stddev_ts"], "samples_ns": row["samples_ns"]}
                    for row in rows]
                parts = "  ".join(
                    f"{'pp' if row['n_gen'] == 0 else 'tg'}"
                    f"{row['n_prompt'] if row['n_gen'] == 0 else row['n_gen']} "
                    f"{row['avg_ns'] / 1e9:.4f} s ({row['avg_ts']:.2f} t/s)" for row in rows)
                print(f"    native  {seconds:.4f} s/rep   = {parts}")
                continue

            points = reps_points["tg"]
            print(f"    harness slope fit over r={list(points)}")

            with quiet_guard(f"round {index + 1} harness mixed") as window:
                fit = one_fit("llamacpp", model, "mixed", 640, threads, points, index % len(points))

            quiet_all = quiet_all and window.quiet
            harness_runs.append(fit["work_seconds"])
            record["harness_seconds"] = fit["work_seconds"]
            record["harness_fit"] = fit
            print(f"      -> {fit['work_seconds']:.4f} s/rep   fixed {fit['fixed_seconds']:.3f} s   "
                  f"R2 {fit['r_squared']:.5f}   quiet={window.quiet}")

        # The same quantity built from two separate single-phase processes, so the invocation difference
        # is measured rather than argued about.
        print("    split shape, same round, one process per phase")
        split = 0.0

        for phase, tokens in (("pp", 512), ("tg", 128)):
            with quiet_guard(f"round {index + 1} harness split {phase}") as window:
                fit = one_fit("llamacpp", model, phase, tokens, threads,
                              reps_points[phase], index % len(reps_points[phase]))

            quiet_all = quiet_all and window.quiet
            split += fit["work_seconds"]
            record[f"split_{phase}_seconds"] = fit["work_seconds"]
            print(f"      {phase}{tokens} {fit['work_seconds']:.4f} s   "
                  f"R2 {fit['r_squared']:.5f}   quiet={window.quiet}")

        split_runs.append(split)
        record["split_seconds"] = split
        print(f"      split total {split:.4f} s/rep")
        rounds.append(record)

    print()
    print("  " + "-" * 92)

    def stats(values):
        return (statistics.mean(values),
                statistics.stdev(values) if len(values) > 1 else 0.0,
                min(values), max(values))

    harness_mean, harness_sd, harness_lo, harness_hi = stats(harness_runs)
    native_mean, native_sd, native_lo, native_hi = stats(native_runs)
    split_mean, split_sd, split_lo, split_hi = stats(split_runs)

    print(f"  harness, mixed shape   {harness_mean:.4f} +/- {harness_sd:.4f} s/rep "
          f"(min {harness_lo:.4f}, max {harness_hi:.4f})")
    print(f"  llama-bench, own timer {native_mean:.4f} +/- {native_sd:.4f} s/rep "
          f"(min {native_lo:.4f}, max {native_hi:.4f})")
    delta = harness_mean - native_mean
    band = harness_sd + native_sd
    print(f"  LIKE-FOR-LIKE delta    {delta:+.4f} s = {100.0 * delta / native_mean:+.2f}%   "
          f"| run-to-run band +/-{band:.4f} s ({100.0 * band / native_mean:.2f}%)")
    print(f"  verdict                {'PASS' if abs(delta) <= band else 'FAIL'}")
    print()
    print(f"  harness, split shape   {split_mean:.4f} +/- {split_sd:.4f} s/rep "
          f"(min {split_lo:.4f}, max {split_hi:.4f})")
    invocation = split_mean - harness_mean
    print(f"  SPLIT minus MIXED      {invocation:+.4f} s = {100.0 * invocation / harness_mean:+.2f}%   "
          f"— a property of the workload, measured, not argued")

    return {
        "model": model,
        "threads": threads,
        "rounds": rounds,
        "harness_mixed": {"mean": harness_mean, "stdev": harness_sd,
                          "min": harness_lo, "max": harness_hi, "runs": harness_runs},
        "native_mixed": {"mean": native_mean, "stdev": native_sd,
                         "min": native_lo, "max": native_hi, "runs": native_runs},
        "harness_split": {"mean": split_mean, "stdev": split_sd,
                          "min": split_lo, "max": split_hi, "runs": split_runs},
        "like_for_like_delta_seconds": delta,
        "like_for_like_delta_percent": 100.0 * delta / native_mean,
        "band_seconds": band,
        "within_band": abs(delta) <= band,
        "split_minus_mixed_seconds": invocation,
        "split_minus_mixed_percent": 100.0 * invocation / harness_mean,
        "quiet": quiet_all,
    }


def sweep(model, repeats, thread_points, sweep_reps, lock):
    """Each engine's rate against its thread count, so a ratio can be taken at each engine's own best.

    **`-t N` does not name the same thing on the two sides**, which is why this is swept rather than
    assumed. llama.cpp runs N threads. Overfit resolves N to N general workers **and** a separately capped
    decode-worker count — `OverfitParallel.ResolveDecodeMaxWorkers` gives `min(N - 1, 10)`, so `-t 16` and
    `-t 32` both decode on 10. That cap is deliberate: the comment there records decode reaching ~37 GB/s
    on a handful of workers and being DRAM-bound past that. Forcing more would measure a configuration this
    engine does not ship.

    **This arm uses each engine's OWN timer, not the slope protocol, and that is a licensed shortcut rather
    than a corner cut.** The slope arm and the mixed anchor both established agreement between the external
    clock and the internal ones — `slope / engine's own timer` came out 1.0027 +/- 0.0115 over six
    llama.cpp fits, and the mixed anchor landed +0.32% against a +/-0.52% band. A sweep locates an optimum,
    and the differences it is looking for here are tens of percent. It is four times cheaper, which is what
    makes three rotated repeats affordable at all.

    **The thread points run in a ROTATED order and the engines alternate.** An ascending sweep is exactly
    how this project produced a smooth context curve out of pure drift, and a sweep is the shape most prone
    to it because the variable and the running order are the same list.
    """
    print("=" * 96)
    print("THREAD SWEEP — each engine against its own thread count")
    print("=" * 96)
    print(f"  points {list(thread_points)}   {repeats} rotated repeats   r={sweep_reps} per point")
    print("  instrument: each engine's own timer (licensed by the slope and mixed-anchor arms)")

    engines = ("llamacpp", "overfit")
    phases = (("pp", 512), ("tg", 128))
    rates = {(engine, phase, threads): []
             for engine in engines for phase, _ in phases for threads in thread_points}
    quiet_all = True

    for index in range(repeats):
        order = engines if index % 2 == 0 else tuple(reversed(engines))
        rotation = index % len(thread_points)
        points = list(thread_points[rotation:]) + list(thread_points[:rotation])
        print()
        print(f"  repeat {index + 1}/{repeats}  engines {' then '.join(order)}  threads {points}")

        for engine in order:
            if engine == "overfit":
                lock.release()

            try:
                with quiet_guard(f"repeat {index + 1} sweep {engine}", quiet_output=True) as window:
                    for threads in points:
                        line = f"    {engine:9s} -t {threads:<3d}"

                        for phase, tokens in phases:
                            command = ENGINES[engine](model, phase, tokens, sweep_reps, threads)
                            _, stdout, _ = timed_run(command)
                            row = parse_engine_json(stdout)
                            rates[(engine, phase, threads)].append(row["avg_ts"])
                            line += (f"   {phase}{tokens} {row['avg_ts']:7.2f} t/s"
                                     f" (sd {row['stddev_ts']:.2f})")

                            if engine == "overfit" and phase == "pp":
                                line += f" [gen {row['n_threads']}/dec {row['n_threads_decode']}]"

                        print(line)
            finally:
                if engine == "overfit":
                    lock.acquire()

            quiet_all = quiet_all and window.quiet
            window.verdict.report(f"repeat {index + 1} sweep {engine}")

    print()
    print("  " + "-" * 92)
    summary = {}

    for phase, tokens in phases:
        print()
        print(f"  {phase}{tokens}   tokens/second, mean of {repeats} rotated repeats")
        header = "    engine     " + "".join(f"{threads:>12d}" for threads in thread_points)
        print(header)

        for engine in engines:
            row = f"    {engine:<11s}"

            for threads in thread_points:
                values = rates[(engine, phase, threads)]
                row += f"{statistics.mean(values):>12.2f}"

            print(row)

            spread = "    " + " " * 11

            for threads in thread_points:
                values = rates[(engine, phase, threads)]
                sd = statistics.stdev(values) if len(values) > 1 else 0.0
                spread += f"{'+/-' + format(sd, '.2f'):>12s}"

            print(spread)

        for engine in engines:
            best = max(thread_points, key=lambda t: statistics.mean(rates[(engine, phase, t)]))
            values = rates[(engine, phase, best)]
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            summary[f"{engine}.{phase}"] = {
                "best_threads": best,
                "best_mean": statistics.mean(values),
                "best_stdev": sd,
                "curve": {str(t): {"mean": statistics.mean(rates[(engine, phase, t)]),
                                   "stdev": (statistics.stdev(rates[(engine, phase, t)])
                                             if repeats > 1 else 0.0),
                                   "runs": rates[(engine, phase, t)]}
                          for t in thread_points},
            }
            print(f"    best {engine:9s} -t {best:<3d} {statistics.mean(values):7.2f} +/- {sd:.2f} t/s")

    return {
        "model": model,
        "thread_points": list(thread_points),
        "repeats": repeats,
        "sweep_reps": sweep_reps,
        "summary": summary,
        "quiet": quiet_all,
    }


def sidecar_ab(with_model, without_model, threads, repeats, reps_points, lock,
               per_phase_threads=None):
    """Overfit with and without its ``*.gguf.repack`` sidecar, interleaved — the only proof it is used.

    **Why an A/B and not a field.** ``GgufLlamaLoader.TryOpenSidecar`` opens ``<model>.repack`` whenever it
    exists, with no environment variable and no flag, so the sidecar cannot be switched off. Worse, its
    presence on disk is **not** proof that it did anything: ``AttachPrepacked`` silently skips a tensor
    whose dimensions disagree, and ``TryOpenSidecar`` swallows a corrupt file and returns null. A measured
    difference between two paths — one with the sidecar beside it, one without — is the only evidence that
    the mechanism is live.

    **Neither the original model nor the sidecar is touched.** ``without_model`` is a symlink inside
    ``Tests/bin`` pointing at the same bytes, with no sidecar next to it.

    **Both arms are reported, and a single number here would be quoted in the wrong direction.** The
    with-sidecar arm is what a user of this engine gets today. The without-sidecar arm is the like-for-like
    comparison against an engine that has no such file and structurally cannot have one.
    """
    print("=" * 96)
    print("SIDECAR A/B — Overfit with and without *.gguf.repack, interleaved")
    print("=" * 96)
    print(f"  with     {with_model}")
    print(f"  without  {without_model}   (the same bytes, no sidecar beside it)")

    arms = ("with", "without")
    models = {"with": with_model, "without": without_model}

    # PRE-FLIGHT, and it exists because of a real 140-second loss on 2026-08-25. The first attempt used a
    # SYMLINK for the without-arm and the engine could not load it at all: `MemoryMappedModelFile.cs:39`
    # takes its bookkeeping length from `new FileInfo(path).Length`, which on Windows is the size of the
    # reparse point — measured 0 against the target's 2104932768 — while `CreateFromFile(capacity: 0)`
    # maps the real file. Every Slice then failed its own bounds check. The campaign died in round 1 after
    # the with-arm had already run. Two seconds of loading each path first turns that into an immediate,
    # readable failure.
    # The lock is released here for the same reason it is released around every other Overfit child:
    # `Sources/Benchmark/Program.cs` takes the same mutex and exits 2 rather than queueing. The first
    # version of this pre-flight forgot, and every arm failed with "Another Overfit benchmark process is
    # already running on this machine" — the guard working correctly against the harness that holds it.
    lock.release()

    try:
        for arm in arms:
            print(f"  pre-flight {arm}: loading {models[arm]}")
            timed_run(ENGINES["overfit"](models[arm], "pp", 8, 1, threads))
            print(f"  pre-flight {arm}: OK")
    finally:
        lock.acquire()
    phases = (("pp", 512), ("tg", 128))
    rates = {(arm, phase): [] for arm in arms for phase, _ in phases}
    fits = {(arm, phase): [] for arm in arms for phase, _ in phases}
    quiet_all = True
    rounds = []

    def threads_for(phase):
        return threads if per_phase_threads is None else per_phase_threads[phase]

    for index in range(repeats):
        order = arms if index % 2 == 0 else tuple(reversed(arms))
        print()
        print(f"  round {index + 1}/{repeats}  order: {' then '.join(order)}")
        record = {"index": index, "order": list(order), "fits": {}}

        for arm in order:
            for phase, tokens in phases:
                points = reps_points[phase]
                arm_threads = threads_for(phase)
                print(f"    {arm:7s} {phase}{tokens} -t {arm_threads}, slope fit over r={list(points)}")
                lock.release()

                try:
                    with quiet_guard(f"round {index + 1} sidecar-{arm} {phase}{tokens}") as window:
                        fit = one_fit("overfit", models[arm], phase, tokens, arm_threads,
                                      points, index % len(points))
                finally:
                    lock.acquire()

                quiet_all = quiet_all and window.quiet
                rates[(arm, phase)].append(fit["tokens_per_second"])
                fits[(arm, phase)].append(fit)
                record["fits"][f"{arm}.{phase}"] = fit | {"quiet": window.quiet}
                print(f"      -> {fit['tokens_per_second']:8.2f} t/s   work {fit['work_seconds']:.4f} s   "
                      f"fixed {fit['fixed_seconds']:.3f} s   R2 {fit['r_squared']:.5f}   "
                      f"quiet={window.quiet}")

        rounds.append(record)

    print()
    print("  " + "-" * 92)
    summary = {}

    for phase, tokens in phases:
        entry = {}

        for arm in arms:
            values = rates[(arm, phase)]
            entry[arm] = {
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "runs": values,
                "worst_r_squared": min(fit["r_squared"] for fit in fits[(arm, phase)]),
            }

        gain = entry["with"]["mean"] - entry["without"]["mean"]
        band = 100.0 * ((entry["with"]["stdev"] / entry["with"]["mean"]) ** 2
                        + (entry["without"]["stdev"] / entry["without"]["mean"]) ** 2) ** 0.5
        percent = 100.0 * gain / entry["without"]["mean"]

        print()
        print(f"  {phase}{tokens}  -t {threads_for(phase)}")

        for arm in arms:
            item = entry[arm]
            print(f"    {arm:8s} {item['mean']:8.2f} +/- {item['stdev']:5.2f} t/s "
                  f"(min {item['min']:.2f}, max {item['max']:.2f}, "
                  f"worst R2 {item['worst_r_squared']:.5f})")

        verdict = ("the sidecar IS used and worth this much"
                   if abs(percent) > band else
                   "DOES NOT SEPARATE — this phase does not measurably use the sidecar")
        print(f"    sidecar worth {gain:+.2f} t/s = {percent:+.2f}%   "
              f"| within-session band +/-{band:.2f}%   -> {verdict}")

        entry["gain"] = gain
        entry["gain_percent"] = percent
        entry["band_percent"] = band
        entry["separates"] = abs(percent) > band
        summary[phase] = entry

    return {
        "with_model": with_model,
        "without_model": without_model,
        "threads": {phase: threads_for(phase) for phase, _ in phases},
        "rounds": rounds,
        "summary": summary,
        "quiet": quiet_all,
    }


def compare(model, threads, repeats, reps_points, lock, per_engine_threads=None, label="matched"):
    """Both engines, one instrument, interleaved — the measurement this module exists for.

    ``per_engine_threads`` maps ``(engine, phase)`` to a thread count, so each engine can be measured at
    **its own** optimum. Without it both sides run at ``threads``, which is the naive matched comparison.
    Both are worth reporting and they do not agree: measured 2026-08-25 by ``--sweep``, llama.cpp's decode
    peaks at **12 threads (32.46 t/s)** and falls to 31.71 at 16 and 30.12 at 32, while Overfit's is flat
    from 12 upward. A matched ``-t 16`` therefore measures llama.cpp **below its own best**.

    **The engines alternate which goes first, round by round.** Running all of one engine and then all of
    the other makes "engine A is faster" and "the first half of the session is faster" the same
    observation, and this box moves enough for that to matter: llama.cpp's own ``pp512`` measured 389.4 t/s
    at 11:30 and 397.6 t/s at 13:10 on 2026-08-25, **+2.1% with nothing changed**.

    **The ratio's resolving power is stated with it and is not the sd of either arm alone.** It is built
    from both arms' run-to-run spread, so a difference smaller than the band is reported as not separating
    rather than as a number.
    """
    print("=" * 96)
    print("COMPARISON — llama.cpp and Overfit, one clock, interleaved")
    print("=" * 96)

    digest = sha256(model)
    phases = (("pp", 512), ("tg", 128))
    engines = ("llamacpp", "overfit")

    def threads_for(engine, phase):
        if per_engine_threads is None:
            return threads

        return per_engine_threads[(engine, phase)]

    print(f"  model      {model}")
    print(f"  sha256     {digest}")
    print(f"  label      {label}")

    for engine in engines:
        for phase, tokens in phases:
            print(f"  threads    {engine:9s} {phase}{tokens}: -t {threads_for(engine, phase)}")

    print("  NOTE: -t does not name the same thing on both sides — see n_threads_decode in the record")
    rates = {(engine, phase): [] for engine in engines for phase, _ in phases}
    fits = {(engine, phase): [] for engine in engines for phase, _ in phases}
    quiet_all = True
    results = {
        "model": model,
        "sha256": digest,
        "label": label,
        "threads": {f"{engine}.{phase}": threads_for(engine, phase)
                    for engine in engines for phase, _ in phases},
        "rounds": [],
    }

    for index in range(repeats):
        order = engines if index % 2 == 0 else tuple(reversed(engines))
        print()
        print(f"  round {index + 1}/{repeats}  order: {' then '.join(order)}")
        round_record = {"index": index, "order": list(order), "fits": {}}

        for engine in order:
            for phase, tokens in phases:
                points = reps_points[phase]
                engine_threads = threads_for(engine, phase)
                print(f"    {engine:8s} {phase}{tokens} -t {engine_threads}, "
                      f"slope fit over r={list(points)}")

                if engine == "overfit":
                    lock.release()

                try:
                    with quiet_guard(f"round {index + 1} {engine} {phase}{tokens}") as window:
                        fit = one_fit(engine, model, phase, tokens, engine_threads,
                                      points, index % len(points))
                finally:
                    if engine == "overfit":
                        lock.acquire()

                quiet_all = quiet_all and window.quiet
                rates[(engine, phase)].append(fit["tokens_per_second"])
                fits[(engine, phase)].append(fit)
                round_record["fits"][f"{engine}.{phase}"] = fit | {"quiet": window.quiet}
                ratio = fit["slope_over_engine_own"]
                print(f"      -> {fit['tokens_per_second']:8.2f} t/s   work {fit['work_seconds']:.4f} s   "
                      f"fixed {fit['fixed_seconds']:.3f} s   R2 {fit['r_squared']:.5f}   "
                      f"quiet={window.quiet}")
                print("      " + (f"   slope / engine's own timer = {ratio:.4f}"
                                  if ratio else "   engine's own timer: n/a"))

        results["rounds"].append(round_record)

    print()
    print("  " + "-" * 92)
    results["quiet"] = quiet_all
    results["summary"] = {}

    for phase, tokens in phases:
        summary = {}

        for engine in engines:
            values = rates[(engine, phase)]
            summary[engine] = {
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "runs": values,
                "worst_r_squared": min(fit["r_squared"] for fit in fits[(engine, phase)]),
                "engine_own_mean": statistics.mean(
                    [fit["engine_own_tokens_per_second"] for fit in fits[(engine, phase)]
                     if fit["engine_own_tokens_per_second"]] or [float("nan")]),
            }

        reference = summary["llamacpp"]
        subject = summary["overfit"]
        ratio = reference["mean"] / subject["mean"] if subject["mean"] > 0 else float("nan")

        # Relative run-to-run spread of both arms, added in quadrature: below this the two engines do not
        # separate and the ratio must not be quoted as a number.
        band = 100.0 * ((reference["stdev"] / reference["mean"]) ** 2
                        + (subject["stdev"] / subject["mean"]) ** 2) ** 0.5

        print()
        print(f"  {phase}{tokens}")

        for engine in engines:
            entry = summary[engine]
            print(f"    {engine:9s} -t {threads_for(engine, phase):<3d} "
                  f"{entry['mean']:8.2f} +/- {entry['stdev']:5.2f} t/s  "
                  f"(n={len(entry['runs'])}, min {entry['min']:.2f}, max {entry['max']:.2f}, "
                  f"worst R2 {entry['worst_r_squared']:.5f}; "
                  f"its own timer said {entry['engine_own_mean']:.2f})")

        # Two bands, and the wider one governs. The within-session band is what these arms resolve; the
        # cross-session floor is what anybody quoting the ratio later will actually be up against, and it
        # was measured three times on this box at about 2%.
        print(f"    llama.cpp / Overfit = {ratio:.3f}x   "
              f"| within-session resolving power +/-{band:.2f}%"
              f"   | cross-session floor on this box +/-2%")
        summary["ratio_llamacpp_over_overfit"] = ratio
        summary["resolving_power_percent"] = band
        results["summary"][phase] = summary

    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--validate", action="store_true",
                        help="run the llama.cpp validity gate and exit")
    parser.add_argument("--compare", action="store_true",
                        help="measure both engines, interleaved, with one clock")
    parser.add_argument("--mixed-anchor", action="store_true",
                        help="like-for-like gate on llama-bench's OWN invocation (-p 512 -n 128)")
    parser.add_argument("--sidecar-ab", default=None,
                        help="path to a model with NO *.repack beside it; runs the with/without A/B")
    parser.add_argument("--phase-threads", default=None,
                        help="per-phase thread counts for the sidecar A/B, e.g. pp=32,tg=24")
    parser.add_argument("--best-threads", default=None,
                        help="per-engine thread counts, e.g. llamacpp.pp=32,llamacpp.tg=12,"
                             "overfit.pp=32,overfit.tg=24")
    parser.add_argument("--sweep", action="store_true",
                        help="rate against thread count for both engines, so each can pick its own best")
    parser.add_argument("--thread-points", default="8,10,12,16,24,32")
    parser.add_argument("--sweep-reps", type=int, default=3,
                        help="repetitions inside each swept process")
    parser.add_argument("--engine", choices=sorted(ENGINES), default=None)
    parser.add_argument("--phase", choices=("pp", "tg"), default="tg")
    parser.add_argument("--tokens", type=int, default=None,
                        help="512 for pp, 128 for tg — llama-bench's own defaults")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--reps-points", default=None,
                        help="override the per-phase defaults in PHASE_REPS_POINTS, e.g. 2,4,6")
    parser.add_argument("--native-reps", type=int, default=3)
    parser.add_argument("--out", default=None, help="write the whole record as JSON")
    args = parser.parse_args(argv)

    if args.reps_points:
        override = tuple(int(part) for part in args.reps_points.split(","))
        reps_points = {"pp": override, "tg": override}
    else:
        reps_points = dict(PHASE_REPS_POINTS)

    if not os.path.exists(args.model):
        print(f"model not found: {args.model}", file=sys.stderr)
        return 2

    lock = MachineLock()

    if not lock.acquire():
        print("Another measurement holds " + _MUTEX_NAME + " — refusing to start.", file=sys.stderr)
        return 2

    # Printed BEFORE the measurement, not attached after it, so a run that dies half-way still says what
    # it was measuring. `docs/measured-baselines.md:472` is the incident: a baseline whose version was not
    # written down had to be reconstructed from a reflog afterwards.
    ours = our_side(args.model)
    print("  our side   HEAD " + str(ours["head"])
          + ("  TREE DIRTY: " + str(len(ours["dirty"])) + " path(s)" if ours["dirty"] else "  tree clean")
          + ("  *** SOME ASSEMBLY IS STALE ***" if ours["stale"] else ""))

    for assembly in ours["assemblies"]:
        print(f"             {assembly['name']}  {str(assembly['sha256'])[:16]}")

    print("  repack sidecar " + ("PRESENT " + str(ours["repack_sidecar_bytes"]) + " bytes — the loader "
                                 "consumes it with no switch; llama.cpp has no equivalent"
                                 if ours["repack_sidecar_present"] else "ABSENT"))

    try:
        if args.validate:
            record = validate(args.model, args.threads, args.repeats, reps_points, args.native_reps)
            ok = all(record["verdict"][phase]["within_combined"] for phase in ("pp", "tg"))
        elif args.sidecar_ab:
            phase_threads = None

            if args.phase_threads:
                phase_threads = {}

                for part in args.phase_threads.split(","):
                    phase, value = part.split("=")
                    phase_threads[phase] = int(value)

            record = sidecar_ab(args.model, args.sidecar_ab, args.threads, args.repeats,
                                reps_points, lock, per_phase_threads=phase_threads)
            ok = True
        elif args.sweep:
            record = sweep(args.model, args.repeats,
                           tuple(int(part) for part in args.thread_points.split(",")),
                           args.sweep_reps, lock)
            ok = True
        elif args.mixed_anchor:
            record = mixed_anchor(args.model, args.threads, args.repeats, reps_points,
                                  args.native_reps)
            ok = record["within_band"]
        elif args.compare:
            per_engine = None
            label = f"matched -t {args.threads}"

            if args.best_threads:
                # "llamacpp.pp=32,llamacpp.tg=12,overfit.pp=32,overfit.tg=24"
                per_engine = {}

                for part in args.best_threads.split(","):
                    key, value = part.split("=")
                    engine, phase = key.split(".")
                    per_engine[(engine, phase)] = int(value)

                label = "each engine at its own best: " + args.best_threads

            record = compare(args.model, args.threads, args.repeats, reps_points, lock,
                             per_engine_threads=per_engine, label=label)
            ok = True
        else:
            if args.engine is None:
                print("--engine is required unless --validate", file=sys.stderr)
                return 2

            tokens = args.tokens if args.tokens else (512 if args.phase == "pp" else 128)

            with quiet_guard(f"{args.engine} {args.phase}{tokens}") as window:
                record = campaign(args.engine, args.model, args.phase, tokens, args.threads,
                                  reps_points=reps_points[args.phase], repeats=args.repeats, lock=lock)

            record["quiet"] = window.quiet
            print(f"  {args.engine} {args.phase}{tokens}: {record['mean']:.2f} +/- {record['stdev']:.2f} "
                  f"t/s (min {record['min']:.2f}, max {record['max']:.2f}, "
                  f"worst R2 {record['worst_r_squared']:.5f}, quiet={window.quiet})")
            ok = True

        if args.out:
            record["our_side"] = ours
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, default=str)
            print(f"  record written to {args.out}")

        return 0 if ok else 1
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
