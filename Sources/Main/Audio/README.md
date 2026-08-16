# `Audio` — decoding, resampling, spectrograms and speech synthesis

Everything sound-related, kept out of the model directories so `../LanguageModels/Whisper` is a model
and nothing else.

## Front end

`WavReader` / `WavWriter` / `AudioFile` handle WAV; `Mp3/` is a **complete hand-written MP3 decoder**
(bit reader, Huffman tables, synthesis window) so an MP3 needs no external dependency.
`AudioResampler` normalises sample rates — Whisper wants 16 kHz and real files are not — and
`MelSpectrogram` produces the log-mel front end the speech models consume.

`AudioSimilarity` / `AudioQualityAssert` exist for testing: comparing generated audio to a reference by
listening does not scale, so a similarity report and an assertion do.

## `Tts/` — speech synthesis

`ITextToSpeechEngine` with text normalisation in front (`TtsTextNormalizer`, `SentenceSplitter`,
`EnglishNumberToWords` — "3.5" has to become words before it can be spoken) and `IAudioSink` behind.

`Snac/` is the neural audio codec that turns acoustic tokens back into a waveform; `Orpheus/` is the
model that produces those tokens, with `OrpheusSnacBridge` between them and a LoRA-based voice-cloning
path (`VoiceCloneDatasetBuilder`, `VoiceCloneTrainer`).

**`SyntheticSpeechMetadata` is the intended policy and is NOT currently enforced.** Generated speech is
meant to carry a marker identifying it as synthetic — a voice-cloning path that can produce unmarked audio
of a real person is not something to ship.

An earlier version of this paragraph claimed the marker was "enforced at the engine rather than left to the
caller". **That was false**, and it was found by a review on 2026-08-01 rather than by anything failing:

- `Orpheus/OrpheusVoiceEngine` does not reference `SyntheticSpeechMetadata` at all. Nothing in the synthesis
  path applies it.
- It is applied by callers — two sites in `Sources/Cli/Commands.cs` and one in
  `Sources/Server/OpenAi/SpeechExchange.cs` — so a new caller gets unmarked audio by default.
- `WavAudioSink` takes it as `SyntheticSpeechMetadata? metadata = null`. Omitting it writes an unmarked file
  and nothing objects.
- The server's `response_format=pcm` path emits raw samples with no container, so it has nowhere to put the
  marker and carries none.

The policy stands; the enforcement does not exist yet. Until it does, **treat unmarked output as reachable**
— see the open-defect entry in `ROADMAP.md`. A safety property that lives only in a document is a documented
intention, not a guarantee, and a document asserting otherwise is worse than no document at all.
