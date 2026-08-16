@echo off
setlocal
pushd "%~dp0"

REM ---------------------------------------------------------------------------------------------------
REM Deploys the Overfit server into the lab cluster as a Prometheus scrape target, three replicas.
REM
REM Requires:
REM   - the monitoring lab installed  (..\monitoring\install.cmd) — supplies the ServiceMonitor CRD
REM   - a local `overfit:latest` image (build.cmd, or the repo-root docker-serve.cmd)
REM   - the model on the host at C:\qwen3b\qwen0.5b.q4km.gguf
REM ---------------------------------------------------------------------------------------------------

echo === prerequisites ===
docker image inspect overfit:latest >nul 2>&1
if errorlevel 1 (
    echo   MISSING image `overfit:latest` — run build.cmd first.
    popd & endlocal & exit /b 1
)
echo   image overfit:latest      OK

kubectl get crd servicemonitors.monitoring.coreos.com >nul 2>&1
if errorlevel 1 (
    echo   MISSING the ServiceMonitor CRD — run ..\monitoring\install.cmd first.
    popd & endlocal & exit /b 1
)
echo   ServiceMonitor CRD        OK

if not exist "C:\qwen3b\qwen0.5b.q4km.gguf" (
    echo   MISSING C:\qwen3b\qwen0.5b.q4km.gguf
    echo   Either put a model there or edit the args/hostPath in deployment.yaml.
    popd & endlocal & exit /b 1
)
echo   model file                OK

echo.
echo === applying ===
kubectl apply -f deployment.yaml
if errorlevel 1 goto :failed

echo.
echo === waiting for the replicas (model load dominates; allow a couple of minutes) ===
kubectl rollout status deployment/overfit-server -n overfit --timeout=5m
if errorlevel 1 goto :notready

echo.
kubectl get pods -n overfit -o wide

echo.
echo Now three identical replicas are emitting Overfit's own metrics. Check Prometheus has found them:
echo   http://127.0.0.1:9090/targets   (run ..\monitoring\forward.cmd first)
echo.
echo The peer-group query this whole lab exists for:
echo   sum by (pod) (rate(process_cpu_seconds_total{namespace="overfit"}[5m]))
echo.
popd
endlocal
exit /b 0

:notready
echo.
echo The rollout did not become ready. Most likely the model path or the image:
echo   kubectl describe pod -n overfit -l app.kubernetes.io/name=overfit-server
echo   kubectl logs -n overfit -l app.kubernetes.io/name=overfit-server --tail=50
popd & endlocal & exit /b 1

:failed
echo.
echo kubectl apply failed — see the message above.
popd & endlocal & exit /b 1
