@echo off
setlocal
pushd "%~dp0"

REM Removes the Overfit server deployment and its namespace. The monitoring lab is left running —
REM ..\monitoring\uninstall.cmd removes that separately.

echo === removing the Overfit server ===
kubectl delete -f deployment.yaml --ignore-not-found

echo.
echo Done. The monitoring stack is untouched; use ..\monitoring\uninstall.cmd for that.
popd
endlocal
