@echo off
setlocal

REM Shows whether the lab is actually healthy — pods, storage, and the scrape targets that matter.

kubectl get ns monitoring >nul 2>&1
if errorlevel 1 (
    echo Namespace `monitoring` not found — run install.cmd first.
    endlocal
    exit /b 1
)

echo === pods ===
kubectl get pods -n monitoring -o wide

echo.
echo === storage ===
kubectl get pvc -n monitoring

echo.
echo === the exporters the detectors depend on ===
kubectl get svc -n monitoring | findstr /I "kube-state-metrics node-exporter prometheus grafana"

echo.
echo === anything not Running or Completed ===
kubectl get pods -n monitoring --no-headers | findstr /V /I "Running Completed"
if errorlevel 1 echo   (none — all pods healthy)

echo.
echo Scrape-target health is the thing a green pod list does NOT tell you. Check it in the UI:
echo   forward.cmd, then http://127.0.0.1:9090/targets
echo.
endlocal
