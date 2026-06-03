# MicDemo — live microphone speech-to-text (Windows)

Record a few seconds from the microphone and transcribe with Whisper — pure .NET on the CPU. A
record-then-transcribe loop: press Enter, speak, get text, repeat.

Microphone capture uses the built-in Windows **winmm** `waveIn` API via P/Invoke — **no NuGet, no
native binary shipped**. The core engine stays platform-neutral; only this demo touches the OS audio
API, so it is Windows-only.

## Quick start

```bat
download-materials.cmd     :: fetches ggml-tiny.bin (~77 MB) into ..\materials
run.cmd                    :: English, 5 s per round
run.cmd pl 5               :: Polish, 5 s per round
```

Or directly:

```bat
dotnet run -c Release --project . -- <model.ggml.bin> [language=en] [seconds=5]
```

## How it works

`Press Enter → record N s → Whisper transcribes → prints text`. Type `q` + Enter to quit. The
transcriber reuses its buffers, so each round is allocation-stable (~2 KB).

## Notes

- Windows only (winmm). On Linux/macOS use [../WhisperDemo](../WhisperDemo) on a file instead.
- Uses the default input device (Settings → System → Sound → Input). Microphone access must be allowed
  for desktop apps (Settings → Privacy → Microphone), else `waveInOpen` fails.
- Lower latency: pass fewer seconds (e.g. `run.cmd en 3`) — shorter window = faster transcription.
- whisper-tiny transcribes a round in well under a second on CPU; swap in `ggml-base.bin` for quality.
