# `LanguageModels/Whisper` — speech to text

A complete Whisper implementation in C#: log-mel front end, `WhisperEncoder`, `WhisperDecoder`,
`WhisperTokenizer`, and `WhisperGgmlLoader` for `ggml-*.bin` weights. `WhisperTranscriber` is the
facade:

```csharp
var text = WhisperTranscriber.Load(ggmlPath).TranscribeFile(wavPath, "en");
```

**Validated end to end on real weights** — `ggml-tiny` against the JFK sample transcribes correctly,
pure .NET on CPU, with no Python and no native runtime. That is the claim this directory supports; it
is not a sketch.

Audio decoding, resampling and the mel filterbank live in `../../Audio`, so this directory is the model
and nothing else.

## Known limits, stated rather than implied

- No KV-cache on the decoder yet — correctness first, and the speed pass has not been made.
- Single 30-second window; longer audio needs the sliding-window pass that is not written.
- WAV in. MP3 decoding exists in `../../Audio/Mp3` but is not wired to this entry point.
- English is the exercised path; other languages are reachable via the language token but unverified.
