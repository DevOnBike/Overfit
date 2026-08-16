@echo off
setlocal

REM ---------------------------------------------------------------------------------------------------
REM Opens Prometheus (and Grafana) on localhost for the lab diagnostics.
REM
REM port-forward rather than a NodePort or an Ingress on purpose: nothing about this lab should be
REM reachable from outside the workstation, and forwarding leaves the cluster's networking untouched.
REM
REM   Grafana     http://127.0.0.1:3000   admin / overfit
REM   Prometheus  http://127.0.0.1:9090   and also :9098 and :9099
REM
REM THREE PORTS FOR ONE PROMETHEUS, and it is not an accident worth cleaning up blindly. The lab
REM diagnostics under Tests/Anomalies/Diagnostics were written at different times and default to
REM different local ports — 9090, 9098, 9099 — each overridable by an environment variable. Until
REM 2026-08-07 this script forwarded only 9090, so anyone who followed the documentation still had two
REM of the five diagnostics failing with a bare "connection refused" that named no cause. Forwarding all
REM three costs nothing and makes the documented route actually work:
REM
REM   :9090  LabFixtureRecorderDiagnostics (OVERFIT_LAB_PROM)
REM          AnomalyGuardShadowRunDiagnostics (OVERFIT_LAB_PROMETHEUS)
REM   :9098  LabFloorCalibrationDiagnostics (OVERFIT_LAB_PROMETHEUS)
REM   :9099  AnomalyGuardEndToEndDiagnostics, PrometheusMetricSourceLabDiagnostics
REM          (OVERFIT_LAB_PROMETHEUS)
REM
REM Each opens in its own window; close the window to stop that forward.
REM ---------------------------------------------------------------------------------------------------

kubectl get ns monitoring >nul 2>&1
if errorlevel 1 (
    echo Namespace `monitoring` not found — run install.cmd first.
    endlocal
    exit /b 1
)

echo Forwarding Prometheus -^> http://127.0.0.1:9090, :9098, :9099
start "overfit-lab prometheus" kubectl port-forward -n monitoring svc/overfit-lab-prometheus 9090:9090 9098:9090 9099:9090

echo Forwarding Grafana    -^> http://127.0.0.1:3000   (admin / overfit)
echo   (this one fails if Grafana is scaled to 0 — it is a discretionary consumer and gets scaled down
echo    during measurements. Bring it back with:
echo      kubectl scale deployment overfit-lab-grafana -n monitoring --replicas=1^)
start "overfit-lab grafana" kubectl port-forward -n monitoring svc/overfit-lab-grafana 3000:80

echo.
echo Windows opened. Close them to stop forwarding.
echo.
echo Check the forwards are actually up before trusting a red test — a refused connection from these
echo diagnostics means the forward is missing far more often than it means the guard is broken:
echo.
echo   curl "http://127.0.0.1:9090/api/v1/query?query=up"
echo.
echo A first query worth running, straight from the blueprint — this is the topology the MVP reads
echo instead of talking to the API server:
echo.
echo   kube_pod_owner{owner_kind="ReplicaSet"}
echo   sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="default"}[5m]))
echo.
endlocal
