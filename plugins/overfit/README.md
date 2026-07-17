# Overfit plugin

Agent skills for [Overfit](https://github.com/DevOnBike/Overfit) — a pure-C#/.NET LLM runtime that runs
**inside your own process, on the CPU**. No Python, no Ollama, no model server, no cloud key, no data egress.

## Install

```
/plugin marketplace add DevOnBike/Overfit
/plugin install overfit@overfit
```

(`<plugin>@<marketplace>` — both happen to be named `overfit` here.) To update later:
`/plugin marketplace update overfit`.

## Skills

| Skill | Use it when |
|---|---|
| **`overfit-add-llm`** | You want a private/local LLM inside an existing .NET app — load a GGUF in C#, add chat/RAG/embeddings on the CPU, replace an OpenAI/Ollama/Azure call with an on-device model, or expose a local model as a `Microsoft.Extensions.AI` `IChatClient`. |
| **`overfit-skill-eval`** | You want to test/score an agent skill or prompt, catch prompt regressions, or measure whether a prompt change actually helped — locally, deterministically (greedy/seeded), offline, at zero API cost, with schema-guaranteed rubric grading. |

Starting a **new** app instead of extending one? Skip the skill and scaffold it:

```bash
dotnet new install DevOnBike.Overfit.Templates
dotnet new overfit-chat
```

## Local LLM tools in your agent (MCP)

Overfit also ships an MCP server exposing `ask` / `rag_query` / `transcribe` backed by a local model. It is
**not** bundled in this plugin because it needs *your* model path, which a plugin manifest can't know. Add it
yourself in one line:

```bash
dotnet tool install -g DevOnBike.Overfit.Cli
claude mcp add overfit -- overfit mcp <model.gguf>
```

## Other agent hosts

The `SKILL.md` files are portable (markdown + YAML frontmatter); only *discovery* differs per host, so each
host gets its own manifest pointing at the **same** `skills/` directory — no duplicated content:

| Host | Manifest |
|---|---|
| Claude Code | `.claude-plugin/marketplace.json` (repo root) + `plugins/overfit/plugin.json` |
| Cursor | `.cursor-plugin/marketplace.json` (repo root) — same shape, but the blurb lives in `metadata.description` |
| Codex | `plugins/overfit/.codex-plugin/plugin.json` — per-plugin, and its schema *does* take `skills` |

⚠️ **Versioning differs by host.** The Claude `plugin.json` deliberately carries **no `version`**: for a
git-hosted marketplace the commit SHA is the version, so every push ships the current skills and there is no
"forgot to bump → users silently stuck" failure. The **Codex** manifest does carry a `version`, which therefore
**must be bumped by hand on release** (it will not track `Directory.Build.props`).

## Notes

- Requires **.NET 10**. Models are standard **GGUF** (Qwen, Llama, Phi, Gemma, Mistral, …).
- Overfit is dual-licensed **AGPL-3.0-or-later / commercial** — see the repo if you're shipping closed source.
- `overfit-spec` lives in this repo's `.claude/skills/` and is intentionally **not** shipped here: it is for
  developing the Overfit engine itself, not for consuming it.
