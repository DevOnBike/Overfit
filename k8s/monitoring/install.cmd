@echo off
setlocal
pushd "%~dp0"

REM ---------------------------------------------------------------------------------------------------
REM Installs the Overfit AIOps development lab: Prometheus + Grafana + kube-state-metrics + node-exporter
REM into the `monitoring` namespace of whatever cluster kubectl currently points at.
REM
REM Deliberately scoped to its own namespace and its own Helm release: this cluster may already be running
REM other things, and none of them should notice this install.
REM ---------------------------------------------------------------------------------------------------

echo.
echo === target cluster ===
kubectl config current-context
if errorlevel 1 goto :nocluster
kubectl get nodes --no-headers
if errorlevel 1 goto :nocluster

echo.
echo === helm repo ===
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update prometheus-community

echo.
echo === installing (this pulls several images on a first run; --wait can take a few minutes) ===
helm upgrade --install overfit-lab prometheus-community/kube-prometheus-stack ^
  --namespace monitoring --create-namespace ^
  --values values.yaml ^
  --wait --timeout 15m
if errorlevel 1 goto :failed

echo.
echo === installed ===
kubectl get pods -n monitoring

echo.
echo Next:
echo   status.cmd    - pods, PVCs and scrape targets
echo   forward.cmd   - Grafana on http://127.0.0.1:3000 (admin / overfit), Prometheus on 9090
echo   uninstall.cmd - remove everything this script created
popd
endlocal
exit /b 0

:nocluster
echo.
echo No reachable Kubernetes cluster. Start Docker Desktop and enable Kubernetes, then run this again.
popd
endlocal
exit /b 1

:failed
echo.
echo Install failed. `kubectl get pods -n monitoring` and `kubectl describe` on anything not Running will
echo usually say why — most often an image pull or a PVC that no storage class can satisfy.
popd
endlocal
exit /b 1
