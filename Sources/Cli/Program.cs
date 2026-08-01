// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.CommandLine;
using DevOnBike.Overfit.Cli;
using DevOnBike.Overfit.Exceptions;

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
var chatTemp = new Option<float>("--temp", "-t")
{
    Description = "Sampling temperature. 0 (default) = greedy/deterministic; 0.2–0.4 factual; 0.7–1.0 creative.",
};
var chatTopK = new Option<int>("--top-k") { Description = "Keep only the K highest-probability tokens (0 = off)." };
var chatTopP = new Option<float>("--top-p") { Description = "Nucleus sampling: keep the smallest set with cumulative probability ≥ P (1 = off)." };
var chatMinP = new Option<float>("--min-p") { Description = "Min-P: keep tokens with probability ≥ minP × P(top) (0 = off). Scale-adaptive tail-trim." };
var chatTopNSigma = new Option<float>("--top-n-sigma") { Description = "Top-nσ: keep tokens with logit ≥ max − n·σ (0 = off). Strong anti-hallucination tail-trim." };
var chatTypicalP = new Option<float>("--typical-p") { Description = "Locally typical sampling: keep tokens near the entropy up to cumulative prob P (1 = off)." };
var chatCommand = new Command("chat", "Chat with a local model interactively.")
{
    chatModel,
    chatTemp,
    chatTopK,
    chatTopP,
    chatMinP,
    chatTopNSigma,
    chatTypicalP,
};
chatCommand.SetAction(parseResult => Commands.Chat(
    parseResult.GetValue(chatModel)!,
    parseResult.GetValue(chatTemp),
    parseResult.GetValue(chatTopK),
    parseResult.GetValue(chatTopP),
    parseResult.GetValue(chatMinP),
    parseResult.GetValue(chatTopNSigma),
    parseResult.GetValue(chatTypicalP)));

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
var serveTtsModel = new Option<string?>("--tts-model")
{
    Description = "Optional Orpheus GGUF (cache name or path) — enables POST /v1/audio/speech (text-to-speech), "
        + "in-process, no data egress. Requires --tts-snac.",
};
var serveTtsSnac = new Option<string?>("--tts-snac")
{
    Description = "SNAC decoder weights directory for TTS (snac_24khz.safetensors). Default: $OVERFIT_SNAC_DIR or "
        + "~/.overfit/snac.",
};
var serveSessions = new Option<int>("--sessions")
{
    Description = "Number of concurrent chat sessions (one KV cache each; weights shared via mmap). Default 1 "
        + "(serialized, like llama.cpp). N>1 decodes N chats at once at the cost of N× KV-cache RAM.",
    DefaultValueFactory = _ => 1,
};
var serveCommand = new Command("serve", "Start an OpenAI-compatible HTTP server for a model.")
{
    serveModel,
    serveHost,
    servePort,
    serveEmbedModel,
    serveTtsModel,
    serveTtsSnac,
    serveSessions,
};
serveCommand.SetAction(parseResult => Commands.Serve(
    parseResult.GetValue(serveModel)!,
    parseResult.GetValue(serveHost)!,
    parseResult.GetValue(servePort),
    parseResult.GetValue(serveEmbedModel),
    parseResult.GetValue(serveTtsModel),
    parseResult.GetValue(serveTtsSnac),
    parseResult.GetValue(serveSessions)));

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

// ── mcp: MCP (Model Context Protocol) stdio server — plug local AI tools into Claude Code / Desktop / IDEs. ──
var mcpModel = new Argument<string>("model")
{
    Description = "A model name in the local store, or a path to a .gguf file (powers the 'ask' and 'rag_query' tools).",
};
var mcpRagDir = new Option<string?>("--rag-dir")
{
    Description = "Optional folder of .txt/.md documents to index at startup — enables the 'rag_query' tool "
        + "(grounded answers with citations; embeddings come from the chat model itself, multilingual).",
};
var mcpWhisperModel = new Option<string?>("--whisper-model")
{
    Description = "Optional whisper.cpp ggml file (e.g. ggml-tiny.bin) — enables the 'transcribe' tool "
        + "(WAV/MP3 → text, loaded lazily on first use).",
};
var mcpCommand = new Command("mcp",
    "Start an MCP (Model Context Protocol) stdio server exposing local, zero-egress AI tools "
    + "(ask, rag_query, transcribe) to hosts like Claude Code:  claude mcp add overfit -- overfit mcp <model>")
{
    mcpModel,
    mcpRagDir,
    mcpWhisperModel,
};
mcpCommand.SetAction(parseResult => Commands.Mcp(
    parseResult.GetValue(mcpModel)!,
    parseResult.GetValue(mcpRagDir),
    parseResult.GetValue(mcpWhisperModel)));

// ── bench: concurrent serving load test of any OpenAI-compatible streaming endpoint. Measures TTFT/
//    ITL/throughput/goodput under N concurrent users and folds them into one holistic score, the same
//    metric shape the GPU-serving world uses — so a pure-.NET CPU server can be compared apples-to-apples. ──
var benchUrl = new Option<string>("--url")
{
    Description = "Base URL of the OpenAI-compatible API (the part before /chat/completions).",
    DefaultValueFactory = _ => "http://127.0.0.1:11434/v1",
};
var benchModel = new Option<string>("--model")
{
    Description = "Model name to request (sent as the 'model' field).",
    DefaultValueFactory = _ => "local",
};
var benchUsers = new Option<int>("--users", "-u")
{
    Description = "Number of concurrent virtual users.",
    DefaultValueFactory = _ => 8,
};
var benchRequests = new Option<int>("--requests", "-n")
{
    Description = "Total requests to send in the measured window (clamped to at least --users).",
    DefaultValueFactory = _ => 64,
};
var benchMaxTokens = new Option<int>("--max-tokens")
{
    Description = "max_tokens per request — caps decode length so ITL is measured over a real stream.",
    DefaultValueFactory = _ => 128,
};
var benchPrompt = new Option<string>("--prompt")
{
    Description = "The user prompt every request sends.",
    DefaultValueFactory = _ => "Explain, in a short paragraph, why running language models locally can be useful.",
};
var benchWarmup = new Option<int>("--warmup")
{
    Description = "Warm-up requests sent before the measured window (not scored) to reach steady state.",
    DefaultValueFactory = _ => 8,
};
var benchCost = new Option<double>("--cost-units")
{
    Description = "Cost units in the score denominator (e.g. CPU cores or server count); default 1.",
    DefaultValueFactory = _ => 1.0,
};
var benchCommand = new Command("bench",
    "Load-test an OpenAI-compatible streaming endpoint (concurrent users → TTFT/ITL/throughput + a holistic score).")
{
    benchUrl,
    benchModel,
    benchUsers,
    benchRequests,
    benchMaxTokens,
    benchPrompt,
    benchWarmup,
    benchCost,
};
benchCommand.SetAction(parseResult => ServingBenchmark.Run(
    parseResult.GetValue(benchUrl)!,
    parseResult.GetValue(benchModel)!,
    parseResult.GetValue(benchUsers),
    parseResult.GetValue(benchRequests),
    parseResult.GetValue(benchMaxTokens),
    parseResult.GetValue(benchPrompt)!,
    parseResult.GetValue(benchWarmup),
    parseResult.GetValue(benchCost)));

var doctorModel = new Argument<string>("model")
{
    Description = "A model name in the local store, or a path to a .gguf file.",
};
var doctorCommand = new Command("doctor",
    "Inspect a GGUF — architecture, quant, tokenizer, chat template, context, support, recommended flags + warnings.")
{
    doctorModel,
};
doctorCommand.SetAction(parseResult => Commands.Doctor(parseResult.GetValue(doctorModel)!));

// ── repack: offline pre-repack Q4_K weights to a sidecar → faster, RAM-free prefill by default. ──
var repackModel = new Argument<string>("model")
{
    Description = "Model alias or path to a .gguf. Its repackable Q4_K matmul weights are converted to block_q4_Kx8.",
};
var repackOutput = new Option<string?>("--output", "-o")
{
    Description = "Sidecar output path; default is <model>.repack next to the GGUF (where the loader auto-discovers it).",
};
var repackCommand = new Command("repack",
    "Pre-repack a Q4_K model to a memory-mapped sidecar — enables the faster register-tiled prefill by default at no extra RAM.")
{
    repackModel,
    repackOutput,
};
repackCommand.SetAction(parseResult => Commands.Repack(
    parseResult.GetValue(repackModel)!, parseResult.GetValue(repackOutput)));

// ── score: run a trained XGBoost model (JSON) over a CSV of feature rows, pure-managed, zero-egress. ──
var scoreModel = new Argument<string>("model")
{
    Description = "Path to an XGBoost model saved as JSON (booster.save_model(\"model.json\")).",
};
var scoreInput = new Option<string>("--input", "-i")
{
    Description = "CSV of feature rows (one row per line, NumFeatures columns; an optional header row is "
        + "auto-detected and skipped; an empty cell or 'nan'/'na'/'?' is a missing value).",
    Required = true,
};
var scoreOutput = new Option<string?>("--output", "-o")
{
    Description = "Write predictions here as CSV; default writes to stdout.",
};
var scoreMargin = new Option<bool>("--margin")
{
    Description = "Emit raw pre-transform margins (XGBoost output_margin=True) instead of probabilities.",
};
var scoreCommand = new Command("score",
    "Score a CSV with a trained XGBoost model (JSON) — pure-managed, zero-allocation, in-process, no Python.")
{
    scoreModel,
    scoreInput,
    scoreOutput,
    scoreMargin,
};
scoreCommand.SetAction(parseResult => Commands.Score(
    parseResult.GetValue(scoreModel)!,
    parseResult.GetValue(scoreInput)!,
    parseResult.GetValue(scoreOutput),
    parseResult.GetValue(scoreMargin)));

// ── gateway: LLM egress firewall — OpenAI-compatible proxy redacting outbound PII/secrets, zero data leaves raw. ──
var gwUpstream = new Option<string?>("--upstream")
{
    Description = "Upstream OpenAI-compatible base URL to forward to, e.g. https://api.openai.com/v1. "
        + "Required unless set in --config; the CLI value overrides the config.",
};
var gwConfig = new Option<string?>("--config")
{
    Description = "Optional JSON config (rules, policy, upstream) so a deployment configures the gateway without "
        + "recompiling. See docs; CLI flags override config values.",
};
var gwKeyEnv = new Option<string>("--upstream-key-env")
{
    Description = "Name of the env var holding the upstream API key. The gateway injects it as the upstream "
        + "Authorization — clients authenticate to the gateway and never see the real key.",
    DefaultValueFactory = _ => "OPENAI_API_KEY",
};
var gwHost = new Option<string>("--host")
{
    Description = "Bind host. Default 127.0.0.1 (local only); use 0.0.0.0 to expose on the network.",
    DefaultValueFactory = _ => "127.0.0.1",
};
var gwPort = new Option<int>("--port", "-p")
{
    Description = "TCP port to listen on.",
    DefaultValueFactory = _ => 8080,
};
var gwAudit = new Option<string>("--audit")
{
    Description = "JSON-lines audit log path (records per-category redaction counts, never the values).",
    DefaultValueFactory = _ => "redaction-audit.jsonl",
};
var gwClientKeysEnv = new Option<string>("--client-keys-env")
{
    Description = "Name of the env var holding gateway client key(s) (comma/space separated). When set, callers "
        + "must present one as 'Authorization: Bearer <key>'. Unset = open gateway (warns at startup).",
    DefaultValueFactory = _ => "OVERFIT_GATEWAY_KEYS",
};
var gwInsecure = new Option<bool>("--insecure")
{
    Description = "Allow binding plaintext HTTP to a non-loopback host. Without this, the gateway refuses to expose "
        + "an unencrypted port (client→gateway traffic carries the PII it protects). Use only when the hop to "
        + "clients is already encrypted (TLS-terminating proxy/sidecar, service-mesh mTLS, private TLS load balancer).",
};
var gwScanResponses = new Option<bool>("--scan-responses")
{
    Description = "Also scan model responses and mask any secrets/PII the model itself produced (leaked system "
        + "prompt, RAG-echoed document). Applies to non-streaming responses. The caller's own redacted values are "
        + "still restored — only genuinely model-generated content is masked. Default off.",
};
var gatewayCommand = new Command("gateway",
    "LLM egress firewall: an OpenAI-compatible proxy that redacts outbound PII/secrets before forwarding to an "
    + "upstream LLM (the gateway holds the key, clients never see it), restores the response, and audits. "
    + "Point your OpenAI client's base_url at it — change one URL.")
{
    gwUpstream,
    gwConfig,
    gwKeyEnv,
    gwHost,
    gwPort,
    gwAudit,
    gwClientKeysEnv,
    gwInsecure,
    gwScanResponses,
};
gatewayCommand.SetAction(parseResult => Commands.Gateway(
    parseResult.GetValue(gwUpstream),
    parseResult.GetValue(gwKeyEnv)!,
    parseResult.GetValue(gwHost)!,
    parseResult.GetValue(gwPort),
    parseResult.GetValue(gwAudit)!,
    parseResult.GetValue(gwConfig),
    parseResult.GetValue(gwClientKeysEnv)!,
    parseResult.GetValue(gwInsecure),
    parseResult.GetValue(gwScanResponses)));

// ---- anomaly-guard: the deployable guard loop, shadow by default ----
var guardConfig = new Option<string>("--config", "-c")
{
    Description = "Path to the JSON configuration: Prometheus URL, namespace, pod regex, metric map, thresholds.",
    Required = true,
};

var guardState = new Option<string?>("--state")
{
    Description = "File holding open incidents across restarts. Without it a restart reopens every incident "
                + "that was running, so a rollout of this process pages an operator for problems they were "
                + "already told about.",
};

var guardCadence = new Option<int>("--cadence-seconds")
{
    Description = "How often to evaluate. Default 300; consecutive windows overlap, so far below the window "
                + "length just re-decides what it already decided.",
};

var guardWindow = new Option<int>("--window-minutes")
{
    Description = "How much history each cycle evaluates. Default 20. Longer is NOT safer: measured on a "
                + "healthy population, 20 min gave 234 false incidents a day, 60 gave 93 and 240 gave 2583.",
};

var anomalyGuardCommand = new Command(
    "anomaly-guard",
    "Watch a deployment's Prometheus metrics and report incidents. Shadow by default: it counts, explains "
    + "and wakes nobody.")
{
    guardConfig,
    guardState,
    guardCadence,
    guardWindow,
};

anomalyGuardCommand.SetAction((parseResult, ct) => AnomalyGuardCommand.RunAsync(
    parseResult.GetValue(guardConfig)!,
    parseResult.GetValue(guardState),
    parseResult.GetValue(guardCadence),
    parseResult.GetValue(guardWindow),
    ct));

var rootCommand = new RootCommand("Overfit — run local LLMs, RAG and agents in pure .NET. No Python, no native runtime.")
{
    pullCommand,
    listCommand,
    chatCommand,
    serveCommand,
    doctorCommand,
    repackCommand,
    mcpCommand,
    ttsCommand,
    voiceCommand,
    benchCommand,
    scoreCommand,
    gatewayCommand,
    anomalyGuardCommand,
};

// Safety net for anything that escapes a command's own handler. Only OverfitException is caught: every one of
// those is a message we wrote for the user (gated repo, corrupt download, unsupported operator), so a stack trace
// adds noise and hides the point. Anything else is a bug in Overfit and is deliberately left to crash with its
// full trace — swallowing those would turn a reportable defect into a silent exit code.
try
{
    return rootCommand.Parse(args).Invoke();
}
catch (OverfitException ex)
{
    Console.Error.WriteLine($"error: {ex.Message}");

    // Inner exceptions usually carry the actionable detail (the HTTP failure under a load error, etc.).
    for (var inner = ex.InnerException; inner is not null; inner = inner.InnerException)
    {
        Console.Error.WriteLine($"  caused by: {inner.Message}");
    }

    return 1;
}
catch (OperationCanceledException)
{
    // Ctrl+C during a long pull or a chat session is a normal exit, not a failure to report.
    Console.Error.WriteLine("cancelled.");
    return 130;
}
