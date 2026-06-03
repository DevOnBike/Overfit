# Mp3Demo — pure-C# MP3 decoder

Decodes an **MPEG-1/2/2.5 Layer III (MP3)** file to mono PCM and writes a 16 kHz 16-bit WAV next to it,
so you can hear that the from-scratch decoder works. **No native binaries, no external libraries, no
Python.** Zero per-frame allocation.

## Quick start

```bat
download-materials.cmd     :: fetches a sample MP3 (~9 MB) into ..\materials
run.cmd                    :: decodes ..\materials\sample.mp3 -> sample.wav
run.cmd C:\music\song.mp3  :: decode your own file
```

Or directly:

```bat
dotnet run -c Release --project . -- <input.mp3> [output.wav]
```

## What it shows

The whole Layer III pipeline implemented in C#: bit reservoir → side info → scalefactors (incl. MPEG-2
LSF) → Huffman → requantize → reorder → stereo → antialias → IMDCT → polyphase subband synthesis.
Prints sample rate / duration / RMS and writes a playable WAV.

## Guarantees

- **Formats:** MPEG-1, MPEG-2 (LSF), MPEG-2.5 Layer III, all sample rates, mono &amp; stereo (downmixed).
- **Zero per-frame allocation** — all working buffers are pre-allocated; only the output buffer is
  allocated, once.
- **Speed:** ~160× real-time on a dev CPU (4.5 s of audio in ~28 ms).
- **Validated:** the same recording decoded from MP3 vs WAV transcribes identically through Whisper.

See [docs/mp3-decoding.md](../../docs/mp3-decoding.md) for the design. MP3 also feeds Whisper directly —
see [../WhisperDemo](../WhisperDemo).
