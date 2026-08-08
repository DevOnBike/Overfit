@echo off
REM Replica churn against the anomaly guard: 12 -> 15 -> 12, with the guard left alone in between.
REM
REM WHY THIS IS A SCRIPT AND NOT A NOTE. It was run by hand on 2026-08-07 to settle backlog item A5,
REM whose two claims — that HPA "leaves ghost series and dilutes groups" — were documented and never
REM measured. Both are now measured, and the numbers only mean something if the next run is the SAME
REM experiment: same step size, same dwell, same three cycles per phase.
REM
REM MEASURED 2026-08-07 (12 replicas, 5-minute guard cadence):
REM   baseline, 11 cycles      findings 1-3,  0 incidents opened
REM   after scale UP           findings 5 then 11,  2 opened
REM   after scale DOWN         findings 6 then 8,   2 opened
REM   pods= stayed at 15 for one cycle after the scale-down — Prometheus's lookback, not slow deletion
REM   AGGREGATE: 0 incidents in 11 static cycles, 5 in the 7 cycles spanning the change.
REM
REM MANUAL SCALING IS THE RIGHT SUBSTITUTE FOR AN HPA HERE, and the reason is not convenience: the guard
REM cannot tell what moved the replica count, only that it moved. What this does NOT cover is HPA's own
REM metric traffic and its scale-down stabilisation window; say so rather than claiming HPA is tested.
REM
REM READ THE RESULT WITH --timestamps. The guard's `cycle:` line is an indented continuation and carries
REM no timestamp of its own; a parser that expects one on that line matches nothing and prints empty
REM sections that read like "no events". That cost one run on 2026-08-07.
REM
REM   kubectl -n lab logs deploy/anomaly-guard --timestamps --since=90m ^| findstr "cycle:"

setlocal
set NS=lab
set DEPLOY=deployment/lab-workload
set BASE=12
set UP=15
REM Three guard cycles per phase plus a minute of slack; the cadence is 5 minutes.
set DWELL=960

echo === baseline: %BASE% replicas ===
kubectl -n %NS% get %DEPLOY%
echo.
echo Let at least three cycles pass before starting, so the baseline band is established.
echo.

echo === scaling %BASE% -^> %UP% (dilution: new pods join the peer group with young heaps) ===
kubectl -n %NS% scale %DEPLOY% --replicas=%UP%
if errorlevel 1 goto :failed
timeout /t %DWELL% /nobreak

echo.
echo === scaling %UP% -^> %BASE% (ghost series: does pods= stay high after they are gone?) ===
kubectl -n %NS% scale %DEPLOY% --replicas=%BASE%
if errorlevel 1 goto :failed
timeout /t %DWELL% /nobreak

echo.
echo === final state ===
kubectl -n %NS% get %DEPLOY%
echo.
echo Now read the cycles:
echo   kubectl -n %NS% logs deploy/anomaly-guard --timestamps --since=90m ^| findstr "cycle:"
goto :eof

:failed
echo.
echo SCALE FAILED - the cluster did not accept the command. Do NOT read the guard log as a result:
echo a phase whose premise was never established has nothing to say.
exit /b 1
