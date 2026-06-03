# MP3 decoding (pure C#)

Overfit ships a from-scratch **MPEG-1 / MPEG-2 / MPEG-2.5 Layer III (MP3)** decoder. It is pure C#
with **no native binaries, no external libraries, and no Python** — consistent with the engine's
zero-dependency, Native-AOT-friendly identity. Its main job is feeding audio to the
[Whisper speech-to-text](../Sources/Main/LanguageModels/Whisper) runtime, but it is usable standalone.

## Usage

```csharp
using DevOnBike.Overfit.Audio.Mp3;

// Decode to mono 32-bit float PCM in [-1, 1].
float[] samples = Mp3Reader.ReadMono(@"C:\audio\song.mp3", out int sampleRate);

// ...or from any stream.
using var fs = File.OpenRead(path);
float[] s = Mp3Reader.ReadMono(fs, out int sr);
```

Format dispatch goes through `IAudioDecoder` — WAV and MP3 are built in, and `AudioFile.ReadMono`
resolves a path to the right decoder by extension. Consumers stay format-agnostic; new sources plug in
without touching them:

```csharp
float[] pcm = AudioFile.ReadMono(@"C:\audio\clip.mp3", out int sampleRate); // or .wav
```

Whisper consumes any registered format directly — `WhisperTranscriber.TranscribeFile` resolves the
decoder and resamples to 16 kHz (mono):

```csharp
var w = WhisperTranscriber.Load(@"C:\whisper\ggml-tiny.bin");
string text = w.TranscribeFile(@"C:\whisper\pl.mp3", "pl");
```

### Speed / live use

On a dev CPU (best-of-3, warm): MP3 decode runs at **~160× real-time** (4.56 s of audio in ~28 ms),
and full speech-to-text on whisper-tiny runs at **~6× real-time** (a 30 s window in ~5 s; base ~3.6×).
So the pipeline keeps up with a live stream. The transcriber reuses its mel / encoder / KV-cache buffers,
so **repeated transcriptions allocate only a few KB** (the result) — see `Demo/MicDemo` for live mic
capture. For low *latency*, transcribe shorter windows (encoder cost scales with window length); true
streaming with overlap + voice-activity detection is a follow-on.

### CLI demo

```
dotnet run -c Release --project Demo/Mp3Demo -- <input.mp3> [output.wav]
```

Decodes the MP3, prints sample rate / duration / RMS, and writes a 16 kHz 16-bit WAV you can play:

```
Decoding pl.mp3 ...
  24000 Hz mono, 109 440 samples, 4.56 s, RMS 0.0673
  decoded in 41 ms (pure C#, zero per-frame allocations)
Wrote pl_decoded.wav (16000 Hz mono 16-bit, 72 960 samples).
```

## Guarantees

- **Formats:** MPEG-1, MPEG-2 (LSF) and MPEG-2.5 Layer III, all sample rates (8–48 kHz), mono &
  stereo (downmixed to mono on read). Layers I/II and free-format are not supported.
- **Zero per-frame allocation.** All working buffers (bit reservoir, scalefactors, requantized lines,
  IMDCT/overlap-add store, the polyphase `V` buffer, scratch) live in pre-allocated instance fields.
  The only allocation in a decode is the single output buffer, sized once from a header probe —
  measured at **24 bytes of overhead beyond the output buffer across an entire 190-frame file**.
- **Validated end-to-end.** Decoding the same recording from MP3 (`pl.mp3`, 24 kHz MPEG-2) and from
  WAV (`polish.wav`, 16 kHz) yields **identical** Whisper transcriptions
  (`Tests/LanguageModels/Whisper/WhisperMp3E2ETests`).

## Design

```text
Sources/Main/Audio/
  WavReader.cs           16-bit PCM / 32-bit float WAV → mono float
  AudioResampler.cs      linear resample (e.g. 24 kHz → 16 kHz for Whisper)
  Mp3/
    Mp3BitReader.cs      MSB-first bit reader (struct, zero-alloc)
    Mp3FrameHeader.cs    sync / version / bitrate+samplerate / frame-size / side-info length
    Mp3Reader.cs         public facade: Probe (frame walk) + ReadMono
    Mp3Huffman.cs        packed binary-tree Huffman walk (big_values + count1)
    Mp3HuffmanData.cs    the ISO Table B.7 codeword tables (generated from the public-domain pdmp3 reference)
    Mp3SynthWindowData.cs the ISO D[512] synthesis window (likewise)
    Mp3Tables.cs         scalefactor bands (9 rates), IMDCT windows + cosine tables, synthesis cosine matrix
    Mp3Decoder.cs        the DSP pipeline
```

The DSP pipeline per granule/channel:

```text
bit reservoir (main_data_begin back-pointer)
  → side info            (MPEG-1 vs MPEG-2/2.5 LSF field layout)
  → scalefactors         (MPEG-1 slen1/slen2 + scfsi sharing; MPEG-2 LSF partitioning)
  → Huffman              (big_values regions + count1 quadruples, linbits ESC, signs)
  → requantize           (global_gain / scalefactor / subblock_gain, x^(4/3))
  → reorder              (short blocks)
  → stereo               (MS + intensity)
  → antialias            (8 butterflies per subband boundary)
  → IMDCT + windowing    (18-pt long / 3×6-pt short, 4 window types) + overlap-add
  → frequency inversion
  → polyphase subband synthesis (32→64 cosine matrix + D[512] window)  → PCM
```

The MPEG-1 path is a faithful port of the public-domain **pdmp3** reference (Krister Lagerström);
the MPEG-2/2.5 LSF side-info and scalefactor scheme were added from the ISO/IEC 13818-3 spec. The
large constant tables (Huffman codewords, the D[512] window) are standardized ISO values transcribed
verbatim from the reference — data, not a code dependency.

## Notes / limits

- Linear resampling is adequate for speech (Whisper is robust); a higher-quality resampler can be
  added later if needed.
- Loading is one-directional: Overfit decodes MP3 → PCM but does not encode MP3.
