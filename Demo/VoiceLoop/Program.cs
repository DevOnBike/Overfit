// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Audio;
using DevOnBike.Overfit.Audio.Tts;
using DevOnBike.Overfit.Audio.Tts.Orpheus;
using DevOnBike.Overfit.Demo.VoiceLoop;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Whisper;

// VoiceLoop — the full local voice agent in one .NET process, on the CPU, no cloud, no Python:
//   microphone → Whisper (speech→text) → LLM (think) → Orpheus + SNAC (text→speech) → speaker.
//
// Usage:
//   VoiceLoop --chat <chat.gguf> --orpheus <orpheus.gguf> --snac <dir> [--whisper <ggml>] [options]
//   VoiceLoop --wav <input.wav> ...     # drive one turn from a file instead of the mic (testable)
// Options: --seconds <n> (mic record length, default 5) | --voice <tara> | --once | --no-play | --save <out.wav>

var opts = ParseArgs(args);

var whisperPath = opts.Get("whisper", EnvDir("OVERFIT_WHISPER_DIR", @"c:\whisper", "ggml-tiny.bin"));
var chatPath = opts.Get("chat", null);
var orpheusPath = opts.Get("orpheus", EnvDir("OVERFIT_ORPHEUS_DIR", @"c:\orpheus", "orpheus-3b-0.1-ft-q4_k_m.gguf"));
var snacDir = opts.Get("snac", Environment.GetEnvironmentVariable("OVERFIT_SNAC_DIR") ?? @"c:\snac");
var voice = opts.Get("voice", OrpheusPrompt.DefaultVoice)!;
var seconds = int.TryParse(opts.Get("seconds", "5"), out var s) ? s : 5;
var wavInput = opts.Get("wav", null);
var savePath = opts.Get("save", null);
var once = opts.Has("once") || wavInput is not null;
var noPlay = opts.Has("no-play");

if (chatPath is null || !File.Exists(chatPath))
{
    Console.Error.WriteLine("Need a chat model: --chat <path-to-.gguf> (a small instruct GGUF works well).");
    return 1;
}
foreach (var (label, path) in new[] { ("Whisper", whisperPath), ("Orpheus", orpheusPath) })
{
    if (!File.Exists(path))
    {
        Console.Error.WriteLine($"{label} model not found: {path}");
        return 1;
    }
}

Console.WriteLine("Loading models (Whisper + chat LLM + Orpheus/SNAC)…");
var whisper = WhisperTranscriber.Load(whisperPath!);
using var chat = OverfitClient.LoadGguf(chatPath, maxContextLength: 4096);
chat.AddSystem("You are a friendly local voice assistant. Reply in one or two short, natural spoken sentences. "
    + "No markdown, no lists, no emoji.");
using var tts = OrpheusVoiceEngine.Load(orpheusPath!, snacDir);

Console.WriteLine("Ready. The whole loop runs on this CPU — nothing leaves the machine.\n");

do
{
    float[] micSamples;
    if (wavInput is not null)
    {
        var raw = AudioFile.ReadMono(wavInput, out var rate);
        micSamples = rate == MicCapture.SampleRate ? raw : AudioResampler.Resample(raw, rate, MicCapture.SampleRate);
        Console.WriteLine($"[input] {Path.GetFileName(wavInput)} ({micSamples.Length / (double)MicCapture.SampleRate:F1}s)");
    }
    else
    {
        Console.Write($"Press Enter to record {seconds}s (or type 'q' to quit): ");
        if (string.Equals(Console.ReadLine()?.Trim(), "q", StringComparison.OrdinalIgnoreCase))
        {
            break;
        }
        Console.WriteLine("🎙  recording…");
        micSamples = MicCapture.Record(seconds);
    }

    var heard = whisper.Transcribe(micSamples, "en").Trim();
    Console.WriteLine($"You:       {heard}");
    if (heard.Length == 0)
    {
        Console.WriteLine("(heard nothing — try again)\n");
        continue;
    }
    if (IsQuit(heard))
    {
        break;
    }

    var reply = chat.Send(heard).Trim();
    Console.WriteLine($"Assistant: {reply}");

    var audio = tts.Synthesize(reply, voice);
    if (savePath is not null)
    {
        WavWriter.WriteMono(savePath, audio, tts.SampleRate, WavSampleFormat.Pcm16,
            SyntheticSpeechMetadata.ForNow(voice).ToInfoComment());
        Console.WriteLine($"(saved {savePath}, {audio.Length / (double)tts.SampleRate:F2}s)");
    }
    if (!noPlay)
    {
        AudioPlayer.Play(audio, tts.SampleRate);
    }
    Console.WriteLine();
}
while (!once);

Console.WriteLine("Bye.");
return 0;

static bool IsQuit(string text)
{
    var t = text.ToLowerInvariant();
    return t is "quit." or "quit" or "goodbye." or "goodbye" or "bye." or "bye";
}

static string EnvDir(string envVar, string fallbackDir, string file)
    => Path.Combine(Environment.GetEnvironmentVariable(envVar) ?? fallbackDir, file);

static Options ParseArgs(string[] argv)
{
    var map = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
    for (var i = 0; i < argv.Length; i++)
    {
        if (!argv[i].StartsWith("--", StringComparison.Ordinal))
        {
            continue;
        }
        var key = argv[i][2..];
        if (i + 1 < argv.Length && !argv[i + 1].StartsWith("--", StringComparison.Ordinal))
        {
            map[key] = argv[++i];
        }
        else
        {
            map[key] = null; // flag
        }
    }
    return new Options(map);
}

internal sealed class Options(Dictionary<string, string?> map)
{
    public string? Get(string key, string? fallback) => map.TryGetValue(key, out var v) && v is not null ? v : fallback;

    public bool Has(string key) => map.ContainsKey(key);
}
