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

**`SyntheticSpeechMetadata` is mandatory, not optional.** Generated speech carries a marker identifying
it as synthetic. This is a deliberate constraint on the feature — a voice-cloning path that can produce
unmarked audio of a real person is not something to ship, and the marker is enforced at the engine
rather than left to the caller.
