cls
dotnet run -c Release -f net10.0 --filter *

@rem The exit code is the only thing a caller can read, and four of its values mean different things.
@rem Captured on the very next line because every command below - echo included - overwrites ERRORLEVEL.
@rem Printed as meaning rather than as a number, which is the whole point: XC-47 was filed because a bare
@rem exit code from this host could not be told apart from a real run.
@set "BENCH_RC=%ERRORLEVEL%"

@if "%BENCH_RC%"=="0" exit /b 0

@echo.
@if "%BENCH_RC%"=="2" echo REFUSED (exit 2): something else is already measuring on this machine, so nothing ran. Wait for it to finish, or stop it, then run again.
@if "%BENCH_RC%"=="3" echo NOTHING RAN (exit 3): the host started but no benchmark produced a measurement. The output above says which - a build error, a rejected option, or a filter that matched nothing.
@if "%BENCH_RC%"=="4" echo PARTIAL (exit 4): some selected cases were measured and some were not, so the table above is INCOMPLETE - it does not say what is missing from it. The unmeasured cases and their reasons are listed above. Re-run with --allow-partial only once you have read them and they are all expected here (XC-48).
@if not "%BENCH_RC%"=="2" if not "%BENCH_RC%"=="3" if not "%BENCH_RC%"=="4" echo FAILED (exit %BENCH_RC%): the benchmark run itself failed.
@exit /b %BENCH_RC%
