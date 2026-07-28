"""Pull a dashboard back out of Grafana and write it into its ConfigMap in this repository.

The provisioning direction (repo -> cluster) is already automatic: the ConfigMap carries the
`grafana_dashboard: "1"` label and Grafana's sidecar imports it on install. This script closes the other
half of the loop, which nothing does automatically — edits made by clicking around in the Grafana UI live
only in Grafana's own database and are lost the moment the release is reinstalled.

Two details decide whether the exported JSON can actually be re-imported:

  * `id` must be removed. It is Grafana's internal database row id, meaningful only in the instance that
    produced it. Left in place, provisioning either fails or silently targets the wrong dashboard.
  * `uid` must be kept. It is the stable identity, and it is why re-importing updates this dashboard in
    place instead of creating a second copy every time.

`version` is dropped for the same reason as `id` — it is the source instance's revision counter, and git
is the revision history that matters here.

Usage:
    python export-dashboard.py [uid] [--file <configmap.yaml>] [--port <n>]
"""

import argparse
import base64
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

DEFAULT_UID = "overfit-peers"
DEFAULT_FILE = pathlib.Path(__file__).with_name("dashboard-overfit-peers.yaml")

# Fields Grafana adds that describe the *instance*, not the dashboard.
INSTANCE_FIELDS = ("id", "version")


def fetch(uid: str, port: int) -> dict:
    """Port-forwards Grafana, reads one dashboard, and always tears the tunnel down again."""
    forward = subprocess.Popen(
        ["kubectl", "port-forward", "-n", "monitoring", "svc/overfit-lab-grafana", f"{port}:80"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        auth = base64.b64encode(b"admin:overfit").decode()
        headers = {"Authorization": f"Basic {auth}"}

        for _ in range(25):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=3).read()
                break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("Grafana did not answer — is the monitoring lab running?")

        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/dashboards/uid/{uid}", headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode())["dashboard"]
    finally:
        forward.terminate()
        try:
            forward.wait(timeout=8)
        except subprocess.TimeoutExpired:
            forward.kill()


def strip(dashboard: dict) -> dict:
    for field in INSTANCE_FIELDS:
        dashboard.pop(field, None)

    # A dashboard saved from the UI records where the user happened to be looking. Pinning the window here
    # would make every export a spurious diff, so it is normalised instead.
    dashboard["refresh"] = dashboard.get("refresh") or "10s"
    dashboard["time"] = {"from": "now-30m", "to": "now"}

    return dashboard


def write_into_configmap(path: pathlib.Path, key: str, dashboard: dict) -> bool:
    """Replaces only the embedded JSON, so the YAML's explanatory comments survive the round trip."""
    original = path.read_text(encoding="utf-8")
    marker = re.search(rf"^(\s+){re.escape(key)}: \|-\s*$", original, re.M)

    if marker is None:
        raise SystemExit(f"Could not find the `{key}: |-` block in {path.name}.")

    indent = marker.group(1) + "  "
    body = json.dumps(dashboard, indent=2, ensure_ascii=False)
    block = "\n".join(indent + line if line else "" for line in body.splitlines())

    updated = original[:marker.end()] + "\n" + block + "\n"

    if updated == original:
        return False

    path.write_text(updated, encoding="utf-8")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uid", nargs="?", default=DEFAULT_UID)
    parser.add_argument("--file", type=pathlib.Path, default=DEFAULT_FILE)
    parser.add_argument("--port", type=int, default=3001,
                        help="local port for the temporary forward; 3001 so it cannot collide with a "
                             "forward you already have open on 3000")
    args = parser.parse_args()

    print(f"=== exporting `{args.uid}` from Grafana ===")
    dashboard = strip(fetch(args.uid, args.port))
    print(f"  title  : {dashboard.get('title')}")
    print(f"  panels : {len(dashboard.get('panels', []))}")

    key = args.file.name.replace("dashboard-", "").replace(".yaml", ".json")
    changed = write_into_configmap(args.file, key, dashboard)

    print(f"\n  {'updated' if changed else 'no change'}: {args.file}")
    if changed:
        print("\n  Apply it back to the cluster with:")
        print(f"    kubectl apply -f {args.file.name}")
        print("  and commit the file — that is now the source of truth, not Grafana's database.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
