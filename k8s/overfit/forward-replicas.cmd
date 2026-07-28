@echo off
setlocal enabledelayedexpansion

REM ---------------------------------------------------------------------------------------------------
REM Opens one local port per replica, which is what the load generator needs.
REM
REM `kubectl port-forward svc/...` is not enough: it binds to a single endpoint and stays there, so all
REM traffic would land on one pod and the peer group would look idle. Driving an uneven split — the case
REM worth studying — requires addressing each replica individually.
REM
REM   http://127.0.0.1:8081  ->  replica 1
REM   http://127.0.0.1:8082  ->  replica 2
REM   http://127.0.0.1:8083  ->  replica 3
REM ---------------------------------------------------------------------------------------------------

set PORT=8081
set ENDPOINTS=

for /f "tokens=*" %%p in ('kubectl get pods -n overfit -l app.kubernetes.io/name^=overfit-server -o jsonpath^="{range .items[*]}{.metadata.name}{'\n'}{end}"') do (
    echo   %%p  -^>  http://127.0.0.1:!PORT!
    start "overfit replica !PORT!" kubectl port-forward -n overfit pod/%%p !PORT!:8080
    if "!ENDPOINTS!"=="" (set ENDPOINTS=http://127.0.0.1:!PORT!) else (set ENDPOINTS=!ENDPOINTS!,http://127.0.0.1:!PORT!)
    set /a PORT+=1
)

if "!ENDPOINTS!"=="" (
    echo No replicas found — run deploy.cmd first.
    endlocal & exit /b 1
)

echo.
echo One window per replica; close a window to stop that forward.
echo.
echo Drive traffic through them with the load generator (flip its [LongFact] to [Fact] first):
echo.
echo   set OVERFIT_LAB_ENDPOINTS=!ENDPOINTS!
echo   set OVERFIT_LAB_SECONDS=180
echo   set OVERFIT_LAB_SKEW=4
echo   dotnet test Tests\Tests.csproj -c Release --filter FullyQualifiedName~AnomalyLabLoadGenerator
echo.
echo SKEW is the interesting knob: 1 is even traffic and gives a peer detector nothing to find; 4 sends
echo four times the load to the first replica, which is both the fault a peer detector should catch and
echo the false positive it must not raise until the signal is normalised per unit of work.
echo.
endlocal
