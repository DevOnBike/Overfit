#!/usr/bin/env python3
"""
Generates realistic K8s pod metrics in HistoricalCsvLoader format.

Columns match HistoricalCsvLoader.ExpectedHeaders exactly:
  timestamp, pod_name, cpu_usage_ratio, cpu_throttle_ratio,
  memory_working_set_bytes, oom_events_rate, latency_p50_ms,
  latency_p95_ms, latency_p99_ms, requests_per_second,
  error_rate, gc_gen2_heap_bytes, gc_pause_ratio, thread_pool_queue_length

Anomalies injected (labelled for validation):
  - OOM + restart cycle (worker-processor)
  - Latency spike / slow DB query (db-proxy, 2x)
  - CPU runaway (scheduler, 2x)
  - Cascade failure: ml-inference crash → api-gateway error surge
  - Memory leak (worker-processor)

Usage:
  python3 generate_k8s_metrics.py --days 7 --out k8s_metrics.csv
  python3 generate_k8s_metrics.py --days 1 --out k8s_metrics_quick.csv
"""

import csv
import math
import random
import datetime
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",    type=int, default=7)
    parser.add_argument("--out",     type=str, default="k8s_metrics.csv")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--interval",type=int, default=15, help="Scrape interval in seconds")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    INTERVAL    = args.interval
    STEPS       = int(args.days * 24 * 3600 / INTERVAL)
    START       = datetime.datetime(2026, 1, 1, 0, 0, 0)
    PODS        = ["api-gateway", "worker-processor", "db-proxy", "scheduler", "ml-inference"]

    # Anomaly windows: (start_step, duration_steps, pod, type)
    scale = STEPS / (7 * 24 * 3600 / INTERVAL)  # scale with days
    ANOMALIES = {
        "oom_worker":         (int(8000*scale),  200,  "worker-processor", "oom"),
        "latency_db_1":       (int(15000*scale), 100,  "db-proxy",         "latency"),
        "cpu_scheduler_1":    (int(22000*scale), 300,  "scheduler",        "cpu_runaway"),
        "cascade_ml":         (int(30000*scale), 500,  "ml-inference",     "crash"),
        "cascade_gw":         (int(30200*scale), 300,  "api-gateway",      "error_surge"),
        "memory_leak_worker": (int(35000*scale), 400,  "worker-processor", "memory_leak"),
        "latency_db_2":       (int(3000*scale),   80,  "db-proxy",         "latency"),
        "cpu_scheduler_2":    (int(10000*scale), 150,  "scheduler",        "cpu_runaway"),
    }

    def in_anomaly(step, pod):
        for name, (start, dur, apod, atype) in ANOMALIES.items():
            if apod == pod and start <= step < start + dur:
                return atype, (step - start) / max(dur, 1)
        return None, 0.0

    def noise(scale=1.0): return rng.gauss(0, scale)
    def clamp(v, lo, hi): return max(lo, min(hi, v))

    def day_factor(t):
        h = t.hour + t.minute / 60
        if 8 <= h <= 20:
            return 0.3 + 0.7 * math.sin(math.pi * (h - 8) / 12)
        return 0.15

    states = {p: {"mem_leak": 0.0, "restart_count": 0} for p in PODS}

    headers = [
        "timestamp", "pod_name",
        "cpu_usage_ratio", "cpu_throttle_ratio",
        "memory_working_set_bytes", "oom_events_rate",
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        "requests_per_second", "error_rate",
        "gc_gen2_heap_bytes", "gc_pause_ratio",
        "thread_pool_queue_length"
    ]

    total = normal = anom_count = 0

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()

        for step in range(STEPS):
            t = START + datetime.timedelta(seconds=step * INTERVAL)
            df = day_factor(t)

            for pod in PODS:
                atype, prog = in_anomaly(step, pod)
                s = states[pod]

                # --- Baseline ---
                if pod == "api-gateway":
                    cpu = clamp(0.15*df + noise(0.03), 0.02, 0.9)
                    throttle = clamp(0.02*df + noise(0.005), 0, 0.5)
                    mem = clamp(300e6 + noise(20e6), 200e6, 800e6)
                    oom = 0.0
                    p50 = clamp(12 + noise(2), 5, 200)
                    p95 = p50 * clamp(2.5 + noise(0.3), 1.5, 5)
                    p99 = p95 * clamp(2.0 + noise(0.3), 1.2, 4)
                    rps = clamp(300*df + noise(30), 10, 1000)
                    err = clamp(0.002 + noise(0.001), 0, 0.05)
                    gc_heap = clamp(50e6 + noise(5e6), 30e6, 200e6)
                    gc_pause = clamp(0.005 + noise(0.002), 0, 0.05)
                    tpq = clamp(8 + noise(3), 0, 100)

                elif pod == "worker-processor":
                    s["mem_leak"] += rng.uniform(0, 0.003)
                    if s["mem_leak"] > 0.4:
                        s["mem_leak"] = 0.0
                        s["restart_count"] += 1
                    cpu = clamp(0.30*df + noise(0.05), 0.05, 0.8)
                    throttle = clamp(0.05*df + noise(0.01), 0, 0.6)
                    mem = clamp((350e6 + s["mem_leak"]*1e9) + noise(20e6), 200e6, 8e9)
                    oom = 0.0
                    p50 = clamp(40 + noise(8), 10, 500)
                    p95 = p50 * clamp(2.0 + noise(0.3), 1.2, 4)
                    p99 = p95 * clamp(2.0 + noise(0.3), 1.2, 4)
                    rps = clamp(80*df + noise(10), 5, 200)
                    err = clamp(0.01 + noise(0.003), 0, 0.15)
                    gc_heap = clamp(80e6 + s["mem_leak"]*500e6 + noise(10e6), 40e6, 4e9)
                    gc_pause = clamp(0.01 + s["mem_leak"]*0.1 + noise(0.003), 0, 0.5)
                    tpq = clamp(15 + noise(5), 0, 200)

                elif pod == "db-proxy":
                    cpu = clamp(0.08 + noise(0.015), 0.02, 0.4)
                    throttle = clamp(0.01 + noise(0.003), 0, 0.2)
                    mem = clamp(600e6 + noise(30e6), 500e6, 800e6)
                    oom = 0.0
                    p50 = clamp(8 + noise(1.5), 3, 100)
                    p95 = p50 * clamp(2.0 + noise(0.2), 1.5, 4)
                    p99 = p95 * clamp(2.0 + noise(0.2), 1.3, 4)
                    rps = clamp(600*df + noise(50), 50, 2000)
                    err = clamp(0.001 + noise(0.0005), 0, 0.03)
                    gc_heap = clamp(30e6 + noise(5e6), 20e6, 80e6)
                    gc_pause = clamp(0.003 + noise(0.001), 0, 0.02)
                    tpq = clamp(5 + noise(2), 0, 50)

                elif pod == "scheduler":
                    cpu = clamp(0.05 + noise(0.02), 0.02, 0.3)
                    throttle = clamp(0.005 + noise(0.002), 0, 0.1)
                    mem = clamp(280e6 + noise(15e6), 200e6, 500e6)
                    oom = 0.0
                    p50 = clamp(5 + noise(1), 2, 50)
                    p95 = p50 * clamp(1.5 + noise(0.2), 1.2, 3)
                    p99 = p95 * clamp(1.5 + noise(0.2), 1.1, 3)
                    rps = clamp(25*df + noise(5), 2, 100)
                    err = clamp(0.0005 + noise(0.0002), 0, 0.01)
                    gc_heap = clamp(20e6 + noise(3e6), 10e6, 60e6)
                    gc_pause = clamp(0.002 + noise(0.001), 0, 0.01)
                    tpq = clamp(3 + noise(1), 0, 30)

                elif pod == "ml-inference":
                    cpu = clamp(0.55*df + noise(0.07), 0.2, 0.95)
                    throttle = clamp(0.1*df + noise(0.02), 0, 0.7)
                    mem = clamp(6e9 + noise(200e6), 4e9, 15e9)
                    oom = 0.0
                    p50 = clamp(80 + noise(15), 40, 500)
                    p95 = p50 * clamp(2.5 + noise(0.3), 1.5, 5)
                    p99 = p95 * clamp(2.0 + noise(0.3), 1.2, 4)
                    rps = clamp(25*df + noise(5), 2, 100)
                    err = clamp(0.005 + noise(0.002), 0, 0.05)
                    gc_heap = clamp(500e6 + noise(50e6), 200e6, 3e9)
                    gc_pause = clamp(0.02 + noise(0.005), 0, 0.15)
                    tpq = clamp(20 + noise(5), 5, 100)

                # --- Anomaly overlays ---
                if atype == "oom":
                    mem = clamp(mem * (1 + prog*3), 0, 8e9)
                    oom = prog * 0.5
                    err = clamp(err + prog*0.3, 0, 1)
                    cpu = clamp(cpu + prog*0.4, 0, 1)
                    if prog > 0.9:
                        s["restart_count"] += 1
                        s["mem_leak"] = 0

                elif atype == "memory_leak":
                    mem = clamp(mem + prog * 7e9, 0, 8e9)
                    gc_heap = clamp(gc_heap + prog * 3e9, 0, 8e9)
                    gc_pause = clamp(gc_pause + prog * 0.4, 0, 1)
                    err = clamp(err + prog * 0.1, 0, 1)

                elif atype == "latency":
                    spike = 3000 * math.sin(math.pi * prog)
                    p50 = clamp(p50 + spike*0.3 + noise(30), 0, 10000)
                    p95 = clamp(p95 + spike*0.7 + noise(50), 0, 10000)
                    p99 = clamp(p99 + spike    + noise(80), 0, 10000)
                    err = clamp(err + 0.3*prog, 0, 1)
                    cpu = clamp(cpu + 0.2*prog, 0, 1)

                elif atype == "cpu_runaway":
                    cpu = clamp(0.05 + prog*0.95 + noise(0.02), 0, 1)
                    throttle = clamp(throttle + prog*0.8, 0, 1)
                    p50 = clamp(p50 + cpu*200, 0, 10000)
                    p95 = clamp(p95 + cpu*500, 0, 10000)
                    p99 = clamp(p99 + cpu*1000, 0, 10000)
                    err = clamp(err + prog*0.5, 0, 1)
                    tpq = clamp(tpq + prog*450, 0, 500)

                elif atype == "crash":
                    cpu = clamp(cpu * max(0, 1 - prog*2), 0, 1)
                    mem = clamp(mem * max(0.05, 1 - prog), 0, 20e9)
                    rps = clamp(rps * max(0, 1 - prog*3), 0, 2000)
                    p50 = 0 if prog > 0.3 else p50
                    p95 = 0 if prog > 0.3 else p95
                    p99 = 0 if prog > 0.3 else p99
                    err = clamp(err + prog*0.95, 0, 1)
                    oom = prog * 0.1

                elif atype == "error_surge":
                    err = clamp(0.3 + prog*0.65, 0, 1)
                    p50 = clamp(p50 + 300*prog, 0, 10000)
                    p95 = clamp(p95 + 800*prog, 0, 10000)
                    p99 = clamp(p99 + 2000*prog, 0, 10000)
                    rps = clamp(rps * (1 - prog*0.6), 0, 2000)

                w.writerow({
                    "timestamp":                t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "pod_name":                 pod,
                    "cpu_usage_ratio":          round(clamp(cpu, 0, 1), 4),
                    "cpu_throttle_ratio":       round(clamp(throttle, 0, 1), 4),
                    "memory_working_set_bytes": int(clamp(mem, 0, 20e9)),
                    "oom_events_rate":          round(clamp(oom, 0, 10), 6),
                    "latency_p50_ms":           round(clamp(p50, 0, 10000), 2),
                    "latency_p95_ms":           round(clamp(p95, 0, 10000), 2),
                    "latency_p99_ms":           round(clamp(p99, 0, 10000), 2),
                    "requests_per_second":      round(clamp(rps, 0, 5000), 2),
                    "error_rate":               round(clamp(err, 0, 1), 5),
                    "gc_gen2_heap_bytes":       int(clamp(gc_heap, 0, 20e9)),
                    "gc_pause_ratio":           round(clamp(gc_pause, 0, 1), 5),
                    "thread_pool_queue_length": round(clamp(tpq, 0, 1000), 1),
                })
                total += 1
                if atype: anom_count += 1
                else:      normal += 1

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Generated: {total:,} snapshots ({args.days} days, {len(PODS)} pods, {INTERVAL}s interval)")
    print(f"Normal: {normal:,} ({100*normal/total:.1f}%)")
    print(f"Anomaly: {anom_count:,} ({100*anom_count/total:.1f}%)")
    print(f"File: {args.out} ({size_mb:.1f}MB)")
    print()
    print("Anomaly windows:")
    for name, (start, dur, pod, atype) in ANOMALIES.items():
        print(f"  {name}: {pod} @ step {start}-{start+dur} ({dur*INTERVAL//60}min)")

if __name__ == "__main__":
    main()
