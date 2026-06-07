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
using DevOnBike.Overfit.LanguageModels.Whisper;

namespace DevOnBike.Overfit.Demo.VoiceClone
{
    // VoiceClone — fine-tune Orpheus on a target voice, pure .NET. Build a dataset from one recording + transcript,
    // QLoRA-train an adapter, save it. Use --dry-run first to confirm the recording splits into the right segments.
    //
    //   VoiceClone --recording my.wav --transcript transcript.txt --orpheus orpheus.gguf --snac C:\snac --voice myvoice
    //              [--out myvoice.adapter] [--dry-run] [--epochs 3] [--max-seq 1700]
    //              [--threshold 0.03] [--min-silence 0.5]
    //   Per-file dataset (NN.wav + sibling NN.txt, auto normalize + silence-trim per clip):
    //   VoiceClone --folder C:\myvoice --orpheus orpheus.gguf --snac C:\snac --voice myvoice --out myvoice.adapter
    internal static class Program
    {
        private static int Main(string[] args)
        {
            var a = ParseArgs(args);
            var recording = a.Get("recording");
            // Per-file dataset: a folder of NN.wav + sibling NN.txt (one sentence per clip). Each clip is
            // peak-normalized and silence-trimmed before SNAC encode (real takes carry a lead-in pause).
            var folder = a.Get("folder");
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
            var temperature = float.TryParse(a.Get("temp"), System.Globalization.CultureInfo.InvariantCulture, out var tp) ? tp : 0.6f;
            var topP = float.TryParse(a.Get("top-p"), System.Globalization.CultureInfo.InvariantCulture, out var pp) ? pp : 0.9f;
            var repeatPenalty = float.TryParse(a.Get("repeat-penalty"), System.Globalization.CultureInfo.InvariantCulture, out var rp) ? rp : 1.1f;
            var seed = int.TryParse(a.Get("seed"), out var sd) ? sd : 1;

            // Optional Whisper → auto-transcribe each segment (robust: can't misalign like order-pairing a transcript).
            var whisperArg = a.Get("whisper");
            var whisper = whisperArg is not null && File.Exists(whisperArg) ? WhisperTranscriber.Load(whisperArg) : null;

            // Diagnostic: split the recording and transcribe each segment with Whisper, alongside the expected line
            // — reveals transcript/segment misalignment and badly-recorded sentences. Needs --whisper.
            if (a.Has("check"))
            {
                return RunCheck(recording, transcript, a.Get("whisper"), minSilence, threshold);
            }

            if (snacDir is null || !PathExists(snacDir))
            {
                Console.Error.WriteLine($"Missing --snac (path not found: {snacDir ?? "<null>"}).");
                return 1;
            }
            if (!synthOnly && folder is not null)
            {
                if (!PathExists(folder))
                {
                    Console.Error.WriteLine($"Missing --folder (path not found: {folder}).");
                    return 1;
                }
            }
            else if (!synthOnly)
            {
                if (recording is null || !PathExists(recording))
                {
                    Console.Error.WriteLine($"Missing --recording (path not found: {recording ?? "<null>"}).");
                    return 1;
                }
                // Either a transcript (paired by order) or --whisper (auto-transcribe each segment).
                if (whisper is null && (transcript is null || !PathExists(transcript)))
                {
                    Console.Error.WriteLine("Need --transcript <txt> (paired by order) or --whisper <ggml> (auto-transcribe per segment).");
                    return 1;
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

            if (dryRun)
            {
                var tok = new GgufEmbeddedTokenizer(GgufTokenizer.Load(orpheus ?? throw new ArgumentException("--orpheus required even for --dry-run (for the tokenizer).")));
                var examples = BuildWith(tok);
                Report(examples);
                Console.WriteLine("\nDry run OK — re-run without --dry-run to train.");
                return 0;
            }

            Console.WriteLine("Loading Orpheus (this is the trainable base)…");
            // Gradient checkpointing stays ON: for this 28-layer / 3B model the full activation graph otherwise OOMs
            // (measured). --no-checkpoint exists only for small models whose whole graph fits.
            using var trainer = new VoiceCloneTrainer(orpheus!, maxSeq, new QLoRAOptions { Epochs = epochs },
                useCheckpoint: !a.Has("no-checkpoint"), restrictToAudioVocab: !a.Has("full-vocab"));
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

            // ── local functions (capture the parsed options above) ──

            List<OrpheusTrainingExample> BuildWith(ITokenizer tok)
            {
                var builder = new VoiceCloneDatasetBuilder(snac, tok, tok.EndOfTextTokenId);

                // Per-file folder: NN.wav + sibling NN.txt. Normalizes + trims lead/trail silence per clip.
                if (folder is not null)
                {
                    Console.WriteLine($"folder: {folder} | per-file clips (.wav + sibling .txt), normalize + silence-trim");
                    return builder.BuildFromFolder(folder, voice, whisper);
                }

                var audio = AudioFile.ReadMono(recording!, out var rate);
                if (whisper is not null)
                {
                    Console.WriteLine($"recording: {audio.Length / (double)rate:F1}s @ {rate} Hz | auto-transcribing each segment with Whisper");
                    return builder.BuildFromRecording(audio, rate, voice, whisper, minSilence, threshold);
                }
                var lines = ReadTranscript(transcript!);
                Console.WriteLine($"recording: {audio.Length / (double)rate:F1}s @ {rate} Hz | transcript lines: {lines.Count}");
                return builder.BuildFromRecording(audio, rate, lines, voice, minSilence, threshold);
            }

            // Generate audio tokens with the trained model, decode through SNAC, save a watermarked WAV.
            void Synthesize(VoiceCloneTrainer t, Snac s, string text, string vc, string outWav)
            {
                Console.WriteLine($"Synthesizing \"{text}\" in voice '{vc}'…");
                var promptIds = TokenizeWith(t.Tokenizer, OrpheusPrompt.Format(text, vc));
                var audioBase = ResolveAudioBase(t.Tokenizer);
                var generated = t.Generate(promptIds, maxNew, t.EndOfTextTokenId,
                    temperature: temperature, topP: topP, repeatPenalty: repeatPenalty, seed: seed);

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
        }

        // Splits the recording the same way training does, transcribes each segment with Whisper, and prints it
        // next to the expected transcript line — so misalignment / bad takes are visible.
        private static int RunCheck(string? recording, string? transcript, string? whisperArg, float minSilence, float threshold)
        {
            if (recording is null || !File.Exists(recording) || transcript is null || !File.Exists(transcript))
            {
                Console.Error.WriteLine("--check needs --recording <wav> and --transcript <txt>.");
                return 1;
            }
            var whisperPath = whisperArg
                ?? Path.Combine(Environment.GetEnvironmentVariable("OVERFIT_WHISPER_DIR") ?? @"c:\whisper", "ggml-tiny.bin");
            if (!File.Exists(whisperPath))
            {
                Console.Error.WriteLine($"Whisper model not found: {whisperPath} (pass --whisper <ggml-*.bin>).");
                return 1;
            }

            var whisper = WhisperTranscriber.Load(whisperPath);
            var audio = AudioFile.ReadMono(recording, out var rate);
            var normalized = AudioPostProcessing.PeakNormalize(audio);
            var segments = AudioSegmenter.SplitOnSilence(normalized, rate, amplitudeThreshold: threshold, minSilenceSeconds: minSilence);
            var lines = ReadTranscript(transcript);

            Console.WriteLine($"segments {segments.Count} vs transcript lines {lines.Count}\n");
            for (var i = 0; i < segments.Count; i++)
            {
                var (start, end) = segments[i];
                var seg = normalized.AsSpan(start, end - start);
                var seg16 = rate == 16000 ? seg.ToArray() : AudioResampler.Resample(seg, rate, 16000);
                var heard = whisper.Transcribe(seg16, "en").Trim();
                var expected = i < lines.Count ? lines[i] : "(no line)";
                Console.WriteLine($"[{i + 1,2}] {(end - start) / (double)rate:F1}s");
                Console.WriteLine($"     exp: {expected}");
                Console.WriteLine($"     got: {heard}");
            }
            return 0;
        }

        private static int[] TokenizeWith(ITokenizer tok, string text)
        {
            var buf = new int[tok.CountTokens(text)];
            var n = tok.Encode(text, buf);
            return n == buf.Length ? buf : buf[..n];
        }

        private static int ResolveAudioBase(ITokenizer tok)
        {
            Span<int> ids = stackalloc int[8];
            var n = tok.Encode("<custom_token_0>", ids);
            return n == 1 ? ids[0] : throw new InvalidOperationException("Not an Orpheus tokenizer.");
        }

        private static void Report(List<OrpheusTrainingExample> ex)
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

        private static List<string> ReadTranscript(string path)
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

        private static bool PathExists(string p) => File.Exists(p) || Directory.Exists(p);

        private static Args ParseArgs(string[] argv)
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
    }

    internal sealed class Args(Dictionary<string, string?> map)
    {
        public string? Get(string key) => map.TryGetValue(key, out var v) ? v : null;

        public bool Has(string key) => map.ContainsKey(key);
    }
}
