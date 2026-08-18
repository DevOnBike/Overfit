@echo off
REM ---------------------------------------------------------------------------
REM Claude Code, permission prompts off, for this repository.
REM
REM Named cc.cmd and NOT claude.cmd on purpose: a claude.cmd sitting in the repo
REM root would shadow the real claude launcher for anything started from here,
REM including Claude Code itself, and the recursion is silent.
REM
REM Any extra arguments are passed straight through, e.g.  cc.cmd --continue
REM ---------------------------------------------------------------------------
cls
claude --permission-mode bypassPermissions %*
