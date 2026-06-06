// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using DevOnBike.Overfit.Audio;
using DevOnBike.Overfit.Audio.Tts;
using DevOnBike.Overfit.Audio.Tts.Orpheus;
using DevOnBike.Overfit.Audio.Tts.Snac;
using DevOnBike.Overfit.LanguageModels.Contracts;
using DevOnBike.Overfit.LanguageModels.LoRA;
using DevOnBike.Overfit.LanguageModels.Tokenizers;

// VoiceClone — fine-tune Orpheus on a target voice, pure .NET. Build a dataset from one recording + transcript,
// QLoRA-train an adapter, save it. Use --dry-run first to confirm the recording splits into the right segments.
//
//   VoiceClone --recording my.wav --transcript transcript.txt --orpheus orpheus.gguf --snac C:\snac --voice myvoice
//              [--out myvoice.adapter] [--dry-run] [--epochs 3] [--max-seq 1700]
//              [--threshold 0.03] [--min-silence 0.5]

var a = ParseArgs(args);
var recording = a.Get("recording");
var transcript = a.Get("transcript");
var orpheus = a.Get("orpheus");
var snacDir = a.Get("snac");
var voice = a.Get("voice") ?? "myvoice";
var outPath = a.Get("out") ?? "voice.adapter";
var dryRun = a.Has("dry-run");
var epochs = int.TryParse(a.Get("epochs"), out var e) ? e : 3;
var maxSeq = int.TryParse(a.Get("max-seq"), out var ms) ? ms : 1700;
var threshold = float.TryParse(a.Get("threshold"), System.Globalization.CultureInfo.InvariantCulture, out var th) ? th : 0.03f;
var minSilence = float.TryParse(a.Get("min-silence"), System.Globalization.CultureInfo.InvariantCulture, out var msl) ? msl : 0.5f;

var testText = a.Get("test");
var testOut = a.Get("test-out") ?? "voice_test.wav";
var adapterIn = a.Get("adapter");
var synthOnly = adapterIn is not null;
var maxNew = int.TryParse(a.Get("max-new"), out var mn) ? mn : 1200;

if (snacDir is null || !PathExists(snacDir))
{
    Console.Error.WriteLine($"Missing --snac (path not found: {snacDir ?? "<null>"}).");
    return 1;
}
if (!synthOnly)
{
    foreach (var (label, path) in new[] { ("recording", recording), ("transcript", transcript) })
    {
        if (path is null || !PathExists(path))
        {
            Console.Error.WriteLine($"Missing --{label} (path not found: {path ?? "<null>"}).");
            return 1;
        }
    }
}
if (!dryRun && (orpheus is null || !File.Exists(orpheus)))
{
    Console.Error.WriteLine("Training needs --orpheus <orpheus.gguf> (omit it only with --dry-run).");
    return 1;
}

var snac = Snac.Load(snacDir);

// Synth-only: load an existing adapter and speak --test in the cloned voice (no training).
if (adapterIn is not null)
{
    if (orpheus is null || !File.Exists(orpheus) || !File.Exists(adapterIn) || testText is null)
    {
        Console.Error.WriteLine("--adapter mode needs --orpheus <gguf>, an existing --adapter <file>, and --test \"sentence\".");
        return 1;
    }
    Console.WriteLine("Loading Orpheus + adapter…");
    using var synthTrainer = new VoiceCloneTrainer(orpheus, maxSeq, new QLoRAOptions());
    synthTrainer.LoadAdapter(adapterIn);
    Synthesize(synthTrainer, snac, testText, voice, testOut);
    return 0;
}

List<OrpheusTrainingExample> BuildWith(DevOnBike.Overfit.LanguageModels.Contracts.ITokenizer tok)
{
    var builder = new VoiceCloneDatasetBuilder(snac, tok, tok.EndOfTextTokenId);
    var audio = DevOnBike.Overfit.Audio.AudioFile.ReadMono(recording!, out var rate);
    var lines = ReadTranscript(transcript!);
    Console.WriteLine($"recording: {audio.Length / (double)rate:F1}s @ {rate} Hz | transcript lines: {lines.Count}");
    return builder.BuildFromRecording(audio, rate, lines, voice, minSilence, threshold);
}

if (dryRun)
{
    var tok = new GgufEmbeddedTokenizer(GgufTokenizer.Load(orpheus ?? throw new ArgumentException("--orpheus required even for --dry-run (for the tokenizer).")));
    var examples = BuildWith(tok);
    Report(examples);
    Console.WriteLine("\nDry run OK — re-run without --dry-run to train.");
    return 0;
}

Console.WriteLine("Loading Orpheus (this is the trainable base)…");
using var trainer = new VoiceCloneTrainer(orpheus!, maxSeq, new QLoRAOptions { Epochs = epochs });
var ex = BuildWith(trainer.Tokenizer);
Report(ex);

var longest = 0;
foreach (var t in ex)
{
    longest = Math.Max(longest, t.InputIds.Length);
}
if (longest > maxSeq)
{
    Console.Error.WriteLine($"Longest example is {longest} tokens > --max-seq {maxSeq}. Raise --max-seq (more RAM) "
        + "or record shorter sentences.");
    return 1;
}

Console.WriteLine($"\nTraining {ex.Count} examples × {epochs} epochs (maxSeq {maxSeq})…");
trainer.Train(ex, (epoch, step, loss) =>
{
    if (step % 5 == 0 || step < 3)
    {
        Console.WriteLine($"  epoch {epoch}  step {step}  loss {loss:F4}");
    }
});

trainer.SaveAdapter(outPath);
Console.WriteLine($"\nSaved adapter → {outPath}");

if (testText is not null)
{
    Synthesize(trainer, snac, testText, voice, testOut);
}
return 0;

// Generate audio tokens with the trained model, decode through SNAC, save a watermarked WAV.
void Synthesize(VoiceCloneTrainer t, Snac s, string text, string vc, string outWav)
{
    Console.WriteLine($"Synthesizing \"{text}\" in voice '{vc}'…");
    var promptIds = TokenizeWith(t.Tokenizer, OrpheusPrompt.Format(text, vc));
    var audioBase = ResolveAudioBase(t.Tokenizer);
    var generated = t.Generate(promptIds, maxNewTokens: maxNew, eosTokenId: t.EndOfTextTokenId);

    var codes = new List<int>();
    foreach (var tok in generated)
    {
        if (tok == t.EndOfTextTokenId)
        {
            break;
        }
        var code = OrpheusSnacBridge.DecodeCustomToken(tok - audioBase, codes.Count);
        if (code is >= 0 and < 4096)
        {
            codes.Add(code);
        }
    }

    if (codes.Count < OrpheusSnacBridge.FrameStride)
    {
        Console.Error.WriteLine($"Model produced only {codes.Count} audio codes — needs more training / epochs.");
        return;
    }

    var levels = OrpheusSnacBridge.Redistribute(CollectionsMarshal.AsSpan(codes));
    var audio = AudioPostProcessing.TrimSilence(s.Decode(levels));
    WavWriter.WriteMono(outWav, audio, s.SampleRate, WavSampleFormat.Pcm16,
        SyntheticSpeechMetadata.ForNow(vc).ToInfoComment());
    Console.WriteLine($"Wrote {outWav}  ({audio.Length / (double)s.SampleRate:F2}s, {codes.Count} codes, watermarked).");
}

static int[] TokenizeWith(ITokenizer tok, string text)
{
    var buf = new int[tok.CountTokens(text)];
    var n = tok.Encode(text, buf);
    return n == buf.Length ? buf : buf[..n];
}

static int ResolveAudioBase(ITokenizer tok)
{
    Span<int> ids = stackalloc int[8];
    var n = tok.Encode("<custom_token_0>", ids);
    return n == 1 ? ids[0] : throw new InvalidOperationException("Not an Orpheus tokenizer.");
}

static void Report(List<OrpheusTrainingExample> ex)
{
    var min = int.MaxValue;
    var max = 0;
    long sum = 0;
    foreach (var t in ex)
    {
        min = Math.Min(min, t.InputIds.Length);
        max = Math.Max(max, t.InputIds.Length);
        sum += t.InputIds.Length;
    }
    Console.WriteLine($"built {ex.Count} examples | token length min {min} / avg {sum / Math.Max(ex.Count, 1)} / max {max}");
}

static List<string> ReadTranscript(string path)
{
    var lines = new List<string>();
    foreach (var raw in File.ReadAllLines(path))
    {
        var s = raw.Trim();
        if (s.Length > 0)
        {
            lines.Add(s);
        }
    }
    return lines;
}

static bool PathExists(string p) => File.Exists(p) || Directory.Exists(p);

static Args ParseArgs(string[] argv)
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
            map[key] = null;
        }
    }
    return new Args(map);
}

internal sealed class Args(Dictionary<string, string?> map)
{
    public string? Get(string key) => map.TryGetValue(key, out var v) ? v : null;

    public bool Has(string key) => map.ContainsKey(key);
}
