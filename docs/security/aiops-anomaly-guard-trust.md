## Security and trust

This section is written for two readers: a customer's security reviewer deciding whether the anomaly guard
may run inside their cluster, and the engineer who has to answer their questionnaire. Every claim below names
the file it was checked against on 2026-08-09; where a claim could not be verified from code, that is stated
rather than omitted.

**Placement note (not part of the merged content):** this file was written by `overfit-ciso`, whose write
access is `SECURITY.md`, `docs/security/**` and its own memory directory — not `docs/aiops/`. The requesting
task named `docs/aiops/aiops-architecture-security.md` as the final location; that move has to be done by
whoever owns write access there (`overfit-architect` or the user), by copying the section below in whole. The
headings are self-contained so it lifts without edits.

### What it reads, and what leaves the cluster

**The guard's only outbound dependency is one HTTP connection to the Prometheus URL in its own config.**
Verified by reading every network call the subsystem makes:

- Metric windows: `PrometheusMetricWindowSource` / `PrometheusMetricSource` (`Sources/Anomalies/Monitoring/PrometheusMetricSource.cs`,
  `PrometheusMetricWindowSource.cs`) — instant and range queries against `PrometheusBaseUrl`.
- Pod ownership and node placement: `PrometheusTopologySource` (`Sources/Anomalies/Monitoring/PrometheusTopologySource.cs`)
  — reads `kube-state-metrics` series **through the same Prometheus**, not through the Kubernetes API server.
  The class's own doc comment states this explicitly: *"topology comes from kube-state-metrics through
  Prometheus, not from the API server."*
- No other `HttpClient`, socket, or file-based egress exists in `Sources/Anomalies/**`. Grepped for
  `HttpClient`, `http://`, `https://` across the subsystem — every hit is one of the two Prometheus source
  files above, their config records, or the historical/live-monitoring option types that carry the same URL.

**Nothing trains on customer data and nothing phones home.** `AnomalyGuardRegistration.AddGuardCore`
(`Sources/Anomalies/Hosting/AnomalyGuardRegistration.cs`) wires exactly one sink by default —
`LoggerIncidentSink`, which writes to the process's own `ILogger` and nowhere else
(`Sources/Anomalies/Hosting/LoggerIncidentSink.cs`). `IAlertSink` / `AlertEngine`
(`Sources/Anomalies/Alerting/`) exist as an extension point, but **no concrete network sink ships in this
repository** — grepped for `: IAlertSink` and found no implementation. A deployment that wants paging,
Slack, or a webhook has to supply its own `IAlertSink`; until then, the guard's output stays inside the pod's
log stream and its own `/metrics` endpoint.

**The guard's own metrics endpoint is a second surface, and it is also read-only in its useful path but has
one write path — see "Permissions" below for that gap.** `GuardMetricsEndpoint`
(`Sources/Cli/GuardMetricsEndpoint.cs`) serves `/metrics` (Prometheus scrape format), `/healthz`,
`/suppressions` (read), and `/ack` (write — an operator's noise/real judgement on an incident). It is a bare
`HttpListener`, deliberately: the class comment states the reasoning — a DI container, routing and middleware
for two hundred bytes of text "would be a larger attack surface and a slower start for no gain."

**No per-node agent, no sidecar, no operator, no CRD.** Confirmed against the deployed shape in
`k8s/lab/anomaly-guard.yaml`: one `Deployment` (`replicas: 1`), one `ConfigMap`, one `PersistentVolumeClaim`,
one `Service`, one `ServiceMonitor` (a CRD **consumed** by the customer's Prometheus Operator, not created or
required by the guard itself — if the CRD is absent, apply everything except that object and point Prometheus
at the `Service` by hand).

### What it sees, and what a compromise yields

The guard reads cluster-wide series (per-pod CPU, memory, GC, latency, error rate, restarts, throttling) and
pod-to-ReplicaSet-to-Deployment-to-node ownership, scoped to the configured namespace and pod regex
(`PrometheusMetricSourceConfig.Namespace` / `PodRegex`, `Sources/Anomalies/Contracts/PrometheusMetricSourceConfig.cs`).

**What a compromise of the guard pod yields beyond what any pod with the same Prometheus read access already
has:**

- **The correlated topology view** — which pod belongs to which ReplicaSet, Deployment and node, resolved from
  `kube-state-metrics` and cached in memory (`PrometheusTopologySource._snapshot`). This is derivable from
  Prometheus directly by anyone with the same query access; the guard does not hold a credential or scope that
  a generic Prometheus reader lacks.
- **Read/write of its own state file** on the mounted `PersistentVolumeClaim` — open incidents and the learned
  baseline (see "Data at rest"). Tampering here degrades or silences detection; it does not expose anything
  about the workload beyond what the metrics already carry.
- **The `/ack` write path**, if network-reachable — see "Permissions."

**What it explicitly does not hold**: no Kubernetes API credential, no bearer token or client certificate for
Prometheus (`PrometheusMetricSourceConfig` and `PrometheusTopologySource` set no `Authorization` header
anywhere — grepped for `Authorization`/`Bearer`/`DefaultRequestHeaders` in `Sources/Anomalies/**` and found
none), and no secret material of any kind in its `ConfigMap`. **If the customer's Prometheus itself requires
authentication or mTLS, the guard as shipped cannot supply it** — this is a functional gap worth stating
plainly rather than a security control.

### Permissions

**What Kubernetes API access it needs: none.** No `k8s`/`KubernetesClient`/`IKubernetes` reference anywhere in
`Sources/Anomalies/**` or `Sources/Cli/Anomaly*.cs` — grepped and confirmed. Every fact the guard has about
the cluster (metrics, topology, ownership) comes through Prometheus HTTP, never the API server.

**What the manifest grants: no RBAC object of any kind.** `k8s/lab/anomaly-guard.yaml` defines no
`ServiceAccount`, `Role`, `RoleBinding`, `ClusterRole` or `ClusterRoleBinding` — grepped the whole `k8s/`
directory for those kinds; only `k8s/overfit/deployment.yaml` and `k8s/lab/workload.yaml` (unrelated
workloads) match. The claim in `docs/aiops/aiops-client-readiness.md` ("no API-server access, no RBAC, no
CRDs, no operator") **is accurate as checked.**

**Gap found — the manifest relies on absence rather than stating it.** The `Deployment` sets no
`serviceAccountName` and no `automountServiceAccountToken: false`, so the pod runs under the namespace's
`default` ServiceAccount and — unless the cluster or namespace has changed the default — gets a token for the
Kubernetes API automounted at `/var/run/secrets/kubernetes.io/serviceaccount` even though the guard makes zero
use of it. This is the CIS Kubernetes Benchmark's "minimize automounting of service account tokens"
recommendation (5.1.6 in recent revisions), and it costs nothing to close here because the guard genuinely
needs no API access at all:

```yaml
spec:
  template:
    spec:
      automountServiceAccountToken: false   # the guard never calls the API server
```

**Also absent: a `securityContext`.** No `runAsNonRoot`, `readOnlyRootFilesystem`,
`allowPrivilegeEscalation: false` or capability drop is set on the container or pod. What the effective UID
and capability set are today was **not verified** — that depends on the base image's own default, which was
not run and inspected as part of this review. Recommend adding an explicit `securityContext` rather than
relying on the image default either way, since the image choice can change between rebuilds.

**Also worth a maintainer decision, not embargoed — it is a documented design property, visible from the
endpoint's own responses:** `/ack` accepts an unauthenticated `POST` from anyone who can reach port `9469`
(`Sources/Cli/GuardMetricsEndpoint.cs`). The manifest defines a `Service` exposing that port
(`k8s/lab/anomaly-guard.yaml`) with no accompanying `NetworkPolicy`, so on a cluster without a default-deny
policy, `/ack` is reachable from **any pod in the cluster**, not just the namespace. The consequence is
specific to this product's failure mode: a caller who can guess or enumerate an incident ID can
`POST /ack?id=<n>&kind=noise&for=90d` and silence a real signal for up to ninety days, with no record of who
did it beyond a log line. Given how much of this subsystem's design is built around "silence must never look
like health," an unauthenticated path to manufactured silence is the one gap that cuts against the product's
own thesis. Mitigation available today without a code change: a `NetworkPolicy` restricting ingress on port
`9469` to the Prometheus scraper and a named operator-tooling namespace; longer term, gate `/ack` behind a
shared token or mTLS the way the metrics scrape itself is not gated (Prometheus scrapes are conventionally
open on a cluster-internal network, but a *write* endpoint on the same port is a different risk class).

### Data at rest

Two files on the mounted `PersistentVolumeClaim` (`anomaly-guard-state`, 128 Mi,
`k8s/lab/anomaly-guard.yaml`):

| File | Written by | Contents | Verified against |
|---|---|---|---|
| `incidents.json` | `FileIncidentStore` | `PersistedIncident` records: numeric ID, timestamps, namespace/workload/ReplicaSet/pod/node **names**, signal name, severity (a float), a one-line summary string | `Sources/Anomalies/Incidents/FileIncidentStore.cs`, `Sources/Anomalies/Contracts/PersistedIncident.cs` |
| `learned-state.txt` | `FileLearnedStateStore` | Per-hour seasonal baselines and calibrated floor values, keyed by metric and subject — numbers, not text | `Sources/Anomalies/Incidents/FileLearnedStateStore.cs` |

**Contents are Kubernetes object names and numbers, not customer data.** `PersistedIncident`'s doc comment is
explicit about scope: "who the incident is about, what named it, how bad it got and when it started" — no
request bodies, no log lines from the customer's application, no environment variables. The one-line
`Summary` string is composed from the same fields (signal name, subject label, percentages) — confirmed by
reading `Incident.Summary` construction and `IncidentNarrative.cs`, neither of which touches any customer
payload because the guard never reads one; its inputs are Prometheus numeric series.

**In a regulated environment, treat pod and node names as the sensitive part**, not the numbers. A workload
name can itself be informative (e.g. naming a project, a customer, or a feature under NDA) even though no
personal or transaction data is present. The PVC should be encrypted at the storage-class level the same way
any other cluster volume with identifying labels would be — this project does not add its own encryption,
so whatever the cluster's default is for other stateful workloads is what protects this one too.

**Loss is a silence risk, not a leak risk.** Losing the volume does not disclose anything; it costs days of
recalibration and a burst of duplicate incident notifications on the next restart (documented in the
manifest's own comment on the `state` volume, and consistent with `FileIncidentStore`'s "restart reopens
every incident that was running" behaviour in `AnomalyGuardCommand.cs`).

### Logs

**Incident and finding lines are structured log records that name pods and quote numeric values, and they
leave the process the moment they are written — into whatever the cluster's logging pipeline collects.**
`LoggerIncidentSink` (`Sources/Anomalies/Hosting/LoggerIncidentSink.cs`) writes every incident and finding
through `ILogger` at `Information` by default (`IncidentLogOptions.Shadow`); the console formatter is
`AddSimpleConsole` (`Sources/Cli/AnomalyGuardCommand.cs`), i.e. stdout, which is exactly what most cluster log
collectors (Fluent Bit, Promtail, the container runtime's own log driver) ship onward by default. A narrative
line looks like: *"Anomaly finding on lab-workload-...-8k5w4: MemoryWorkingSetBytes [Resource] severity 0.96
— Series rose by 210.0% of typical..."* (`k8s/lab/anomaly-guard.yaml`, comment block; the shape matches
`IncidentNarrative.Describe` in `Sources/Anomalies/Incidents/IncidentNarrative.cs`).

This is not a defect — it is the intended notification path in shadow mode — but it means **whatever
retention, access control and export policy the customer applies to their own log pipeline now also applies
to pod/node names and resource-usage figures for the monitored workload.** State this at install time rather
than let it surface later as a compliance question: nothing about the guard's logging is more sensitive than
the metric names and pod names already visible to anyone with `kubectl logs` access to the namespace, but it
does concentrate them into one continuously-narrated stream rather than leaving them scattered across raw
metric queries.

### Supply chain

**Two separate images exist for two separate purposes, and they are not equally hardened — say which one a
customer is actually running.**

| | `Sources/Cli/Dockerfile` (shipping artefact) | `k8s/lab/guard.Dockerfile` (lab-only) |
|---|---|---|
| Build | Native-AOT, `linux-x64`, multi-stage | Framework-dependent, single-stage |
| Runtime base | `mcr.microsoft.com/dotnet/runtime-deps:10.0-noble-chiseled` — no shell, no .NET runtime, glibc + zlib only | `mcr.microsoft.com/dotnet/aspnet:10.0` — full ASP.NET runtime image |
| Base image pinning | Floating tag (`10.0`, `10.0-noble-chiseled`), **not digest-pinned** | Floating tag (`10.0`), **not digest-pinned** |
| Verified via | `docker-publish.yml` (`.github/workflows/`) | not built in CI; the Dockerfile's own header says "a lab image rebuilt whenever the guard changes" |

Both float on a tag rather than a digest, so a base-image rebuild upstream changes what ships on the next
build without a corresponding change to this repository — consistent with the project's known base-image gap
(see the CISO memory snapshot, re-verify before quoting an age). **The lab image is explicitly not the
production artefact** — its own comment says so — so a customer evaluating this for their cluster should be
told which Dockerfile actually builds what they will run; the lab manifest exists to test the deployed shape
described in this document, not to ship it.

**Publish pipeline (`.github/workflows/docker-publish.yml`, `publish-nuget.yml`), checked 2026-08-09:**

- `docker-publish.yml`: manual (`workflow_dispatch`) only, `permissions: contents: read`, six `uses:` action
  references, **none pinned to a commit SHA** (`actions/checkout@v4`, `docker/setup-buildx-action@v3`,
  `docker/login-action@v3`, `docker/metadata-action@v5` ×2, `docker/build-push-action@v6`). No SBOM
  generation, no image signing (`cosign` or otherwise), no provenance attestation step.
- `publish-nuget.yml`: sets `ContinuousIntegrationBuild=true` and `SourceRevisionId=<sha>` on every build/pack
  step, which is what makes `Microsoft.SourceLink.GitHub` meaningful — a consumer can map a shipped assembly
  back to the exact commit. **No package signing** (`dotnet nuget push` does not sign; NuGet.org's own
  author-signing is opt-in and not configured here), no SBOM.

**What this means concretely for a customer verifying what they run**: they can confirm the source commit
behind a NuGet package (SourceLink) but cannot today verify a Docker image against a signed digest, cannot
consume a machine-readable SBOM, and cannot verify that the GitHub Action that built either artefact was the
version reviewed rather than a same-tag replacement pushed later upstream. This matches the project's
existing, previously-identified supply-chain gap and is not a new finding specific to the guard — it applies
to every image and package this project publishes.

### Regulatory framing

**What this subsystem is:** an anomaly detector over infrastructure telemetry (CPU, memory, GC, latency,
error rate, restarts, throttling) for a customer's own Kubernetes workload. It makes no decision about a
natural person, processes no personal data as input, and — in its shipped shadow-mode default — takes no
automated action on the infrastructure it watches: every incident is a log line and a metric until an
operator explicitly runs `overfit anomaly-ack ... --real`, and even the arming path (`aiops-client-flow.md`,
Stage 4) routes to a human, not to an actuator.

**On the EU AI Act specifically**, checked against the Commission's non-binding draft guidelines on Annex III
classification published 19 May 2026 and third-party summaries of them (Baker Botts, March 2026; Modulos,
2026; artificialintelligenceact.eu's Annex III text) — treat the following as a hypothesis about this
codebase checked against primary guidance, not as a compliance determination, because that determination
depends on the customer's own deployment context and no lawyer has reviewed this text:

- Annex III §2 covers AI systems used as a **safety component in the management or operation of critical
  infrastructure** (electricity, gas, water, and similar essential services). The Commission's May 2026 draft
  guidance further narrows this: a system is classified high-risk under this heading **only when used by an
  entity formally designated as a critical entity under the Critical Entities Resilience (CER) Directive.**
- This guard, as built, monitors **application- and container-level operational metrics of a customer's own
  workload** — not a SCADA, pipeline, grid, or water-system control loop — and takes no automated control
  action; the loop terminates at a human acknowledgement. It is not, by design, a safety component that
  operates infrastructure.
- **If a CER-designated critical-infrastructure operator deploys this guard to watch systems that are
  themselves part of their designated critical infrastructure**, that customer's own AI Act obligations may
  attach to their use of it regardless of what this project claims — the classification test in the draft
  guidance runs on the deployer's designation and use case, not on the tool's architecture alone. That
  determination is the customer's to make with their own counsel; this document states the mechanism honestly
  (human-in-the-loop, no personal data, no infrastructure actuation) and stops there rather than asserting a
  classification.

**What this section does not claim**: compliance with the EU AI Act, a legal opinion, or that no customer
deployment of this guard could ever fall inside Annex III. It states what the guard technically does and does
not do, sourced to the code, so a customer's own compliance team can apply their classification to accurate
facts.

Sources: [Baker Botts, "The EU AI Act: What Energy Executives Should Know Before August 2026" (March 2026)](https://www.bakerbotts.com/thought-leadership/publications/2026/march/the-eu-ai-act); [Modulos, "EU AI Act Annex III Draft Guidelines: What Changed" (2026)](https://www.modulos.ai/blog/eu-ai-act-annex-iii-draft-guidelines-what-changed/); [artificialintelligenceact.eu, Annex III text](https://artificialintelligenceact.eu/annex/3/).

### What it is blind to, as a security property

Restated from `docs/aiops/aiops-client-readiness.md` because a security reviewer reads it differently from an
SRE: every blind spot in that document is also a place where a real incident — including one an attacker
caused on purpose (resource exhaustion, a crash loop, a noisy-neighbour attack from a co-located tenant) —
will not be reported. The two most relevant to a security review: **a single OOM kill is invisible to peer
comparison** (caught only by the separate rules/restart channels), and **CPU throttling is invisible on any
container without a CPU limit set**, which is a customer-side configuration choice, not a guard defect.

### Known limitations, stated rather than implied

- No authentication or transport encryption between the guard and Prometheus is supported by the guard itself
  — TLS, if used, comes entirely from how the customer terminates it in front of Prometheus; the guard adds no
  certificate validation logic of its own beyond whatever `HttpClient` does by default for an `https://` URL.
- No multi-tenancy isolation beyond the namespace/pod-regex scope in one guard's own configuration — a second
  team's workload in the same namespace matching the same pod regex would be visible to this guard, which is
  why `docs/aiops/aiops-client-readiness.md` recommends one guard per namespace.
- The guard's own supply chain (image, package) has the same gaps as the rest of the project — no digest
  pinning, no image signing, no SBOM — see "Supply chain" above; nothing about the anomaly subsystem is held
  to a stricter standard than the rest of the repository today.
