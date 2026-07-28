@echo off
setlocal
pushd "%~dp0"

REM Pulls a dashboard back out of Grafana into its ConfigMap in this repo — the return leg of the
REM provisioning loop, which nothing does automatically. Edits made in the Grafana UI live only in
REM Grafana's database and vanish on the next reinstall until they are exported here.
REM
REM   export-dashboard.cmd                 exports `overfit-peers`
REM   export-dashboard.cmd <other-uid>     exports another dashboard by uid

python export-dashboard.py %*
if errorlevel 1 (
    popd & endlocal & exit /b 1
)

echo.
echo Review the diff before committing — an export also captures anything you were experimenting with.
echo   git diff -- k8s/monitoring/
popd
endlocal
