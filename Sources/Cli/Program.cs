// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.CommandLine;
using DevOnBike.Overfit.Cli;

// `overfit` — a single self-contained binary (AOT) for running local LLMs in pure .NET.
//   overfit pull <model>     download a GGUF model into ~/.overfit/models
//   overfit list             list downloaded models
//   overfit chat <model>     interactive chat REPL
//   overfit serve <model>    start an OpenAI-compatible HTTP server

var pullModel = new Argument<string>("model")
{
    Description = "A model alias (e.g. qwen2.5-3b), a HuggingFace GGUF repo (owner/repo), or a direct "
        + "https URL to a .gguf (internal repo / mirror). Set HF_ENDPOINT to use a HuggingFace mirror.",
};
var pullFile = new Option<string?>("--file", "-f")
{
    Description = "Explicit GGUF filename (or substring) to download; default picks the q4_k_m quant.",
};
var pullCommand = new Command("pull", "Download a model (HuggingFace GGUF) into the local store.")
{
    pullModel,
    pullFile,
};
pullCommand.SetAction(parseResult => Commands.Pull(parseResult.GetValue(pullModel)!, parseResult.GetValue(pullFile)));

var listCommand = new Command("list", "List downloaded models.");
listCommand.SetAction(_ => Commands.List());

var chatModel = new Argument<string>("model")
{
    Description = "A model name in the local store, or a path to a .gguf file.",
};
var chatCommand = new Command("chat", "Chat with a local model interactively.")
{
    chatModel,
};
chatCommand.SetAction(parseResult => Commands.Chat(parseResult.GetValue(chatModel)!));

var serveModel = new Argument<string>("model")
{
    Description = "A model name in the local store, or a path to a .gguf file.",
};
var serveHost = new Option<string>("--host")
{
    Description = "Bind host. Default 127.0.0.1 (local only); use 0.0.0.0 to expose on the network.",
    DefaultValueFactory = _ => "127.0.0.1",
};
var servePort = new Option<int>("--port", "-p")
{
    Description = "TCP port to listen on.",
    DefaultValueFactory = _ => 11434,
};
var serveEmbedModel = new Option<string?>("--embed-model")
{
    Description = "Optional sentence-embedding model directory (HuggingFace BERT: config.json + vocab.txt "
        + "+ model.safetensors, e.g. all-MiniLM-L6-v2). Enables POST /v1/embeddings, in-process, no data egress.",
};
var serveCommand = new Command("serve", "Start an OpenAI-compatible HTTP server for a model.")
{
    serveModel,
    serveHost,
    servePort,
    serveEmbedModel,
};
serveCommand.SetAction(parseResult => Commands.Serve(
    parseResult.GetValue(serveModel)!,
    parseResult.GetValue(serveHost)!,
    parseResult.GetValue(servePort),
    parseResult.GetValue(serveEmbedModel)));

var rootCommand = new RootCommand("Overfit — run local LLMs, RAG and agents in pure .NET. No Python, no native runtime.")
{
    pullCommand,
    listCommand,
    chatCommand,
    serveCommand,
};

return rootCommand.Parse(args).Invoke();
