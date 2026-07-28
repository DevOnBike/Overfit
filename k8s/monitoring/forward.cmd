@echo off
setlocal

REM ---------------------------------------------------------------------------------------------------
REM Opens Grafana and Prometheus on localhost.
REM
REM port-forward rather than a NodePort or an Ingress on purpose: nothing about this lab should be
REM reachable from outside the workstation, and forwarding leaves the cluster's networking untouched.
REM
REM   Grafana     http://127.0.0.1:3000   admin / overfit
REM   Prometheus  http://127.0.0.1:9090
REM
REM Each opens in its own window; close the window to stop that forward.
REM ---------------------------------------------------------------------------------------------------

kubectl get ns monitoring >nul 2>&1
if errorlevel 1 (
    echo Namespace `monitoring` not found — run install.cmd first.
    endlocal
    exit /b 1
)

echo Forwarding Grafana    -^> http://127.0.0.1:3000   (admin / overfit)
start "overfit-lab grafana" kubectl port-forward -n monitoring svc/overfit-lab-grafana 3000:80

echo Forwarding Prometheus -^> http://127.0.0.1:9090
start "overfit-lab prometheus" kubectl port-forward -n monitoring svc/overfit-lab-prometheus 9090:9090

echo.
echo Two windows opened. Close them to stop forwarding.
echo.
echo A first query worth running, straight from the blueprint — this is the topology the MVP reads
echo instead of talking to the API server:
echo.
echo   kube_pod_owner{owner_kind="ReplicaSet"}
echo   sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="default"}[5m]))
echo.
endlocal
