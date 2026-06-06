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

// ── tts: text → speech (WAV), in-process, watermarked. Placeholder engine until the neural backend lands. ──
var ttsText = new Option<string>("--text")
{
    Description = "The text to synthesize.",
    Required = true,
};
var ttsOut = new Option<string>("--out", "-o")
{
    Description = "Output WAV path.",
    Required = true,
};
var ttsVoice = new Option<string>("--voice")
{
    Description = "A voice id (preset, or one enrolled via 'overfit voice enroll').",
    DefaultValueFactory = _ => "default",
};
var ttsLanguage = new Option<string>("--language")
{
    Description = "Language tag of the text (e.g. en, pl).",
    DefaultValueFactory = _ => "en",
};
var ttsModel = new Option<string?>("--model")
{
    Description = "Orpheus GGUF (cache name or path) for real neural speech. Default: $OVERFIT_ORPHEUS_DIR or any "
        + "orpheus*.gguf in the model cache. If absent, a placeholder tone is written instead.",
};
var ttsSnac = new Option<string?>("--snac")
{
    Description = "SNAC decoder weights directory (contains snac_24khz.safetensors). Default: $OVERFIT_SNAC_DIR or "
        + "~/.overfit/snac. Required (with --model) for real speech.",
};
var ttsCommand = new Command("tts", "Synthesize speech from text to a WAV — in-process, no cloud, watermarked.")
{
    ttsText,
    ttsOut,
    ttsVoice,
    ttsLanguage,
    ttsModel,
    ttsSnac,
};
ttsCommand.SetAction(parseResult => Commands.Tts(
    parseResult.GetValue(ttsText)!,
    parseResult.GetValue(ttsVoice)!,
    parseResult.GetValue(ttsOut)!,
    parseResult.GetValue(ttsLanguage)!,
    parseResult.GetValue(ttsModel),
    parseResult.GetValue(ttsSnac)));

// ── tts eval: objective quality of generated audio vs a reference ("how close to ideal"). ──
var evalReference = new Option<string>("--reference", "-r")
{
    Description = "Reference (ideal) WAV/MP3 to compare against.",
    Required = true,
};
var evalCandidate = new Option<string>("--candidate", "-c")
{
    Description = "Candidate (generated) WAV/MP3 to score.",
    Required = true,
};
var ttsEvalCommand = new Command("eval", "Score generated audio against a reference (SNR / correlation / mel + DTW).")
{
    evalReference,
    evalCandidate,
};
ttsEvalCommand.SetAction(parseResult => Commands.TtsEval(
    parseResult.GetValue(evalReference)!,
    parseResult.GetValue(evalCandidate)!));
ttsCommand.Subcommands.Add(ttsEvalCommand);

// ── voice: manage enrolled voices (enroll requires consent). ──
var enrollId = new Argument<string>("id")
{
    Description = "Id to enroll the voice under.",
};
var enrollSample = new Option<string>("--sample")
{
    Description = "Reference audio clip (WAV/MP3) of the voice.",
    Required = true,
};
var enrollLanguage = new Option<string>("--language")
{
    Description = "Primary language of the voice (e.g. pl, en).",
    DefaultValueFactory = _ => "pl",
};
var enrollConsent = new Option<bool>("--consent")
{
    Description = "Required: confirm you OWN this voice or have explicit permission to use it.",
};
var voiceEnrollCommand = new Command("enroll", "Enroll a voice from a reference clip (requires --consent).")
{
    enrollId,
    enrollSample,
    enrollLanguage,
    enrollConsent,
};
voiceEnrollCommand.SetAction(parseResult => Commands.VoiceEnroll(
    parseResult.GetValue(enrollId)!,
    parseResult.GetValue(enrollSample)!,
    parseResult.GetValue(enrollLanguage)!,
    parseResult.GetValue(enrollConsent)));

var voiceListCommand = new Command("list", "List enrolled voices.");
voiceListCommand.SetAction(_ => Commands.VoiceList());

var voiceCommand = new Command("voice", "Manage enrolled voices for TTS.")
{
    voiceEnrollCommand,
    voiceListCommand,
};

var rootCommand = new RootCommand("Overfit — run local LLMs, RAG and agents in pure .NET. No Python, no native runtime.")
{
    pullCommand,
    listCommand,
    chatCommand,
    serveCommand,
    ttsCommand,
    voiceCommand,
};

return rootCommand.Parse(args).Invoke();
