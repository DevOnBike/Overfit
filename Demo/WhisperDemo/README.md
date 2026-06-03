# WhisperDemo — speech-to-text (pure .NET, CPU)

Transcribes a **WAV or MP3** with a whisper.cpp ggml model, entirely in C# on the CPU — no GPU, no
Python, no native binary. Any sample rate (resampled to 16 kHz), mono or stereo (downmixed).

## Quick start

```bat
download-materials.cmd     :: fetches ggml-tiny.bin (~77 MB) + jfk.wav into ..\materials
run.cmd                    :: transcribes jfk.wav (English)
run.cmd pl                 :: transcribe in Polish (works on a Polish recording)
```

Or call the demo directly:

```bat
dotnet run -c Release --project . -- <model.ggml.bin> <audio.wav|audio.mp3> [language=en]
```

## What it shows

- Pure-C# Whisper pipeline: log-mel (Bluestein FFT) → multi-threaded encoder → KV-cache greedy decode.
- Loads a whisper.cpp `ggml-*.bin` model directly (no conversion).
- Decodes WAV and MP3 in pure C# (the MP3 decoder is dependency-free — see [../Mp3Demo](../Mp3Demo)).

## Speed

On a dev CPU, whisper-tiny runs a 30 s window in ~0.5 s (**~60× real-time**); base ~36×. The MP3
decoder alone runs at ~160× real-time. The transcriber reuses its buffers, so repeated calls are
allocation-stable (~2 KB).

## Notes

- Bigger model = better quality, slower: swap `ggml-tiny.bin` for `ggml-base.bin` / `ggml-medium.bin`
  (download from the same Hugging Face repo, `ggerganov/whisper.cpp`).
- For live microphone input see [../MicDemo](../MicDemo).
