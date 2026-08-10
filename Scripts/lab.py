"""Helpers for driving the anomaly-guard lab, written once instead of re-pasted.

**Why this file exists.** Every scratch script in `.claude/do*.py` re-implemented the same
four things — run kubectl, query Prometheus, find a pod, replay a window — and each copy
carried whatever bug the previous copy had. The same broken unpacking
(`_, target, _ = kubectl(...)`, which returns two values) was pasted three times in one
evening and failed three times. That is not a thinking error, it is copying a fragment along
with a defect nobody fixed at the source.

Import it instead:

    import sys
    sys.path.insert(0, r"D:\\Overfit\\.claude")
    from lab import kubectl, prom, guard_pod, workload_pods, inject, replay_signals

**This file is TRACKED on purpose** — see the negation for it in `.gitignore`. The first
version was written on 2026-08-09 into `.claude/`, which is ignored wholesale, and was gone
by the next morning; an agent then re-derived the port-forward helpers by hand from
`run_two_hour_check.py`, which is exactly the paste-propagation the file exists to stop. A
helper that must survive cannot live somewhere version control is not looking.

Everything here returns plain values and raises nothing on the normal failure paths — a
scratch script that dies on a transient query is worse than one that reports an empty result
and carries on. **The caller still has to check** the result is non-empty before building on
it: an empty answer and a negative answer look identical downstream.
"""
import calendar
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = r"D:\Overfit"
PROMETHEUS = "http://127.0.0.1:9090"
NAMESPACE = "lab"


def kubectl(args, timeout=400):
    """Runs kubectl. Returns (stdout, stderr), both stripped — always TWO values."""
    proc = subprocess.run(
        ["kubectl"] + list(args), capture_output=True, text=True,
        timeout=timeout, encoding="utf-8", errors="replace")

    return (proc.stdout or "").strip(), (proc.stderr or "").strip()


def kubectl_out(args, timeout=400):
    """Just the stdout, for the common case where the error is not interesting."""
    return kubectl(args, timeout)[0]


_forward = None


def prometheus_up(timeout=6):
    """Whether the instrument answers at all. Cheap, and the answer is not a query result."""
    try:
        with urllib.request.urlopen(f"{PROMETHEUS}/-/ready", timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def ensure_prometheus(wait_seconds=25):
    """
    Makes sure something is listening on PROMETHEUS, starting a port-forward if not.

    **Missing instrument is not missing data, and this module used to conflate them.** On
    2026-08-10 a host outage killed the forward; `prom()` then returned `[]` for every query
    and a sizing script read that as "the pods report no memory". The two are indistinguishable
    downstream, which is the same failure the whole anomaly subsystem is built around.

    Returns True if Prometheus answers afterwards.
    """
    global _forward

    if prometheus_up():
        return True

    port = PROMETHEUS.rsplit(":", 1)[-1]
    _forward = subprocess.Popen(
        ["kubectl", "port-forward", "-n", "monitoring",
         "svc/overfit-lab-prometheus", f"{port}:9090"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(wait_seconds):
        time.sleep(1)

        if prometheus_up(timeout=3):
            return True

    return False


def _query(path, params, timeout):
    """
    Shared query path. Raises if the INSTRUMENT is unreachable; returns [] if the instrument
    answered and had nothing. A transient failure of one query is [], a dead tunnel is not.
    """
    if not ensure_prometheus():
        raise RuntimeError(
            f"Prometheus is not reachable at {PROMETHEUS} and a port-forward could not be "
            "established. Stopping — an empty result here would read as 'no series'.")

    url = f"{PROMETHEUS}{path}?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode()).get("data", {}).get("result", [])
    except (OSError, ValueError):
        return []


def prom(expr, timeout=25):
    """An instant query. [] means the instrument answered and had nothing."""
    return _query("/api/v1/query", {"query": expr}, timeout)


def prom_range(expr, minutes, step=15, timeout=45):
    """A range query ending now. [] means the instrument answered and had nothing."""
    end = int(time.time())

    return _query("/api/v1/query_range",
                  {"query": expr, "start": end - minutes * 60, "end": end, "step": step},
                  timeout)


def utc(stamp):
    """
    Epoch seconds from an ISO UTC stamp ("2026-08-09T21:52:00Z").

    **Not `time.mktime(...) - time.timezone`.** That reads the tuple as LOCAL time and
    `time.timezone` is the non-DST offset, so between March and October here it lands an hour
    off — which on 2026-08-09 made a Python-side window overlap a fault while the replay,
    given the same string, covered the quiet period. The two halves of one arm disagreed and
    the arm was recorded as a failure that had not happened.
    """
    return calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))


def by_pod(rows):
    """{pod: value} from an instant query, dropping NaN."""
    return {r["metric"].get("pod", "?"): float(r["value"][1])
            for r in rows if r["value"][1] != "NaN"}


def guard_pod():
    """
    The anomaly guard's pod name — the LIVE one.

    Reading fresh is not enough. Right after `kubectl rollout restart` the terminating pod is
    still listed and `.items[0]` returned it, ten seconds after `rollout status` had reported
    success. The log of a dead pod is empty, which reads as "the guard is not cycling" and is
    really "wrong pod".

    `-o name`, not a jsonpath template: a template with quoted separators is one escaping
    mistake away from `unterminated quoted string`, and kubectl reports that on stderr while
    stdout comes back EMPTY — which a caller reads as "no pods" rather than "bad query".
    """
    out = kubectl_out([
        "get", "pods", "-n", NAMESPACE, "-l", "app.kubernetes.io/name=anomaly-guard",
        "--field-selector=status.phase=Running",
        "--sort-by=.metadata.creationTimestamp", "-o", "name"])

    names = [line.split("/", 1)[-1].strip() for line in out.splitlines() if line.strip()]

    if not names:
        return ""

    # Newest last after the sort. Terminating pods keep phase Running, so drop any already
    # condemned — checked one at a time, and only when there is a choice to make.
    for name in reversed(names):
        if len(names) == 1 or not kubectl_out([
                "get", "pod", "-n", NAMESPACE, name,
                "-o", "jsonpath={.metadata.deletionTimestamp}"]):
            return name

    return names[-1]


def workload_pods():
    """Every workload pod name, sorted."""
    out = kubectl_out(["get", "pods", "-n", NAMESPACE, "-l",
                       "app.kubernetes.io/name=lab-workload", "-o", "name"])

    return sorted(line.split("/", 1)[-1].strip()
                  for line in out.splitlines() if line.strip())


def guard_cycles(since="15m", pod=None):
    """
    [(timestamp, cycle line)] from the guard's log.

    The pod is looked up fresh unless given, because reading the name BEFORE a rollout and the
    log AFTER it tails a dead pod and shows nothing — which reads as "no cycles" and is really
    "wrong pod".
    """
    pod = pod or guard_pod()

    if not pod:
        return []

    lines = kubectl_out(["logs", "-n", NAMESPACE, pod, f"--since={since}"]).splitlines()
    stamp = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")
    found = []

    for i, line in enumerate(lines):
        match = stamp.match(line)

        if not match:
            continue

        body = lines[i + 1].strip() if i + 1 < len(lines) else ""

        if "cycle:" in body:
            found.append((match.group(1), body))

    return found


def apply_and_read_back(path, section, key, field):
    """
    Applies a manifest and reads the value back OUT OF THE CLUSTER.

    `kubectl apply` reports success for a field it dropped and silently removes anything the
    file does not carry, so the read-back is the only evidence the change landed.
    Returns (landed_value, apply_output).
    """
    out, err = kubectl(["apply", "-f", path])

    if re.search(r"unknown field|Warning", out + err, re.I):
        return None, out + err

    body = kubectl_out(["get", "configmap", "-n", NAMESPACE, "anomaly-guard-config",
                        "-o", r"jsonpath={.data.guard\.json}"])

    try:
        entry = json.loads(body).get(section, {}).get(key, {})
    except ValueError:
        return None, out

    return entry.get(field), out


def inject(pod, path, hold_seconds=0, port=18300):
    """
    POSTs a fault to one pod through a port-forward, optionally holds, then clears.

    Returns the endpoint's reply, or None if the pod could not be reached — a fault that was
    never injected must not be mistaken for one the guard failed to notice.

    **Timeouts are generous on purpose.** A pod pinned at its CPU quota cannot answer in 15
    seconds, and a `/fault/clear` that times out leaves the fault running.
    """
    forward = subprocess.Popen(
        ["kubectl", "-n", NAMESPACE, "port-forward", f"pod/{pod}", f"{port}:8080"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace")

    try:
        for _ in range(30):
            time.sleep(1)

            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/fault", timeout=5).read()
                break
            except (urllib.error.URLError, OSError, TimeoutError):
                continue
        else:
            return None

        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", method="POST", data=b"")

        with urllib.request.urlopen(request, timeout=60) as response:
            reply = response.read().decode()[:150]

        if hold_seconds:
            time.sleep(hold_seconds)

            for _ in range(5):
                try:
                    urllib.request.urlopen(urllib.request.Request(
                        f"http://127.0.0.1:{port}/fault/clear", method="POST", data=b""),
                        timeout=120).read()
                    break
                except (urllib.error.URLError, OSError, TimeoutError):
                    continue

        return reply
    except (urllib.error.URLError, OSError, TimeoutError):
        return None
    finally:
        forward.terminate()

        try:
            forward.wait(timeout=10)
        except subprocess.TimeoutExpired:
            forward.kill()


def replay_signals(minutes=None, cycles=12, config=None, timeout=2400,
                   cadence=300, start_utc=None):
    """
    Replays a window through the guard and returns the per-signal breakdown line.

    This is how a finding gets attributed: the guard logs individual findings only when an
    incident OPENS, so while one is ongoing the log names nothing.

    **`cycles` and `cadence` are load-bearing, not cosmetics.** The window a rule sees holds
    ONE sample per cycle, and `AnomalyGuardConfigReader.BuildRule` pins `MinimumSamples: 20`
    for every configured absolute rule — under 20 cycles `SustainedThresholdRule` returns
    `WarmingUp` and CANNOT produce a finding, whatever the data does. `cycles * cadence` is
    therefore the wall-clock the replay covers, and `MinBreachFraction` (0.25 by default) is a
    share OF THAT. An 8-minute fault replayed at the deployed 300 s cadence is at best 2
    breaching samples out of 20 — 10%, under the gate, silent by arithmetic. Replay a short
    fault at a short cadence (`cadence=30, cycles=24` covers 12 minutes) and only then is the
    silence evidence about the detector.

    `start_utc` pins the window to something that actually happened ("2026-08-09T22:10:00Z");
    `minutes` is the convenience form, counted back from now.
    """
    start = start_utc or time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - (minutes or 15) * 60))
    env = dict(os.environ)
    env.update({
        "OVERFIT_RUN_LONG": "1",
        "OVERFIT_LAB_PROMETHEUS": PROMETHEUS,
        "OVERFIT_REPLAY_CONFIG": config
        or rf"{REPO}\k8s\anomaly-guard\guard.lab-workload.json",
        "OVERFIT_REPLAY_START_UTC": start,
        "OVERFIT_REPLAY_CYCLES": str(cycles),
        "OVERFIT_REPLAY_CADENCE_SECONDS": str(cadence),
    })

    proc = subprocess.run(
        ["dotnet", "test", rf"{REPO}\Tests\Tests.csproj", "-c", "Release",
         "--filter", "FullyQualifiedName~AnomalyGuardReplayDiagnostics",
         "-l", "console;verbosity=detailed"],
        capture_output=True, text=True, timeout=timeout, env=env,
        encoding="utf-8", errors="replace")

    text = (proc.stdout or "") + (proc.stderr or "")

    return next((l.strip() for l in text.splitlines() if "rows per signal" in l),
                "(no breakdown — did the replay run?)")


def suite(filter_expression=None, timeout=3000):
    """
    Runs the test suite and returns (exit code, summary line, failing test names).

    `Tests/Tests.csproj`, never `Overfit.sln`: the semantic-navigator MCP server holds
    `Tools/SemanticNavigator`'s own DLL open while it is running, and a solution build fails
    with MSB3021 copying over it.
    """
    args = ["dotnet", "test", rf"{REPO}\Tests\Tests.csproj", "-c", "Release"]

    if filter_expression:
        args += ["--filter", filter_expression]

    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                          encoding="utf-8", errors="replace")
    text = (proc.stdout or "") + (proc.stderr or "")
    summary = next((l.strip() for l in text.splitlines()
                    if "niepowodzenie:" in l or "Powodzenie!" in l or "Failed!" in l
                    or "Passed!" in l), "(no summary)")
    failed = sorted({n for n in re.findall(r"(?:Niepowodzenie|Failed) (DevOnBike\S+)", text)})

    return proc.returncode, summary, failed
