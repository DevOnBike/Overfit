@echo off
REM Shows the Overfit MCP registration + spawns every configured server for a health check
REM (expect: "overfit: ... - OK Connected").
call claude mcp get overfit
echo.
call claude mcp list
