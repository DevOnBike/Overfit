# Overfit project templates

`dotnet new` templates for building **local, in-process LLM apps** with
[Overfit](https://github.com/DevOnBike/Overfit) — a pure-.NET, on-device LLM runtime.
No Python, no Ollama, no model server, no cloud.

## Install

```bash
dotnet new install DevOnBike.Overfit.Templates
```

## Templates

### `overfit-chat`

A Minimal API streaming-chat app that loads a local GGUF model and exposes it as a standard
`Microsoft.Extensions.AI.IChatClient`, so it drops into code already written against that
abstraction.

```bash
dotnet new overfit-chat -o MyChatApp
cd MyChatApp
# point it at a GGUF model, then:
dotnet run
```

The scaffolded app pins the Overfit runtime packages to a known-published version, so
`dotnet new` → `dotnet run` works out of the box.

## Adding Overfit to an EXISTING app

Templates are for new apps. To add local inference to a project you already have, use the
`overfit-add-llm` skill from the Overfit plugin, or follow
[the integration guide](https://github.com/DevOnBike/Overfit#readme).

## License

Dual-licensed: AGPL-3.0-or-later for open source, commercial license available. See
[LICENSE.md](https://github.com/DevOnBike/Overfit/blob/main/LICENSE.md).
