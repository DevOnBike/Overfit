# Overfit TTS — initiative tracker

The single source of truth for the text-to-speech work: **what we're building, why, where we are, and what's
next.** Update the status table as stages land.

---

## Goal

Synthesize speech from text **entirely in managed .NET, on the CPU** — no Python, no GPU, no native codec — so
Overfit closes the **voice loop**:

```
hear:  Whisper (speech → text)   ✅ shipped
think: Bielik / Qwen (LLM)       ✅ shipped  (RAG / tools / JSON)
speak: TTS (text → speech)       ◀ this initiative
```

A local voice agent that **hears, thinks and speaks — in one .NET process, on-prem, no data egress.** That's the
"wow" demo that pairs with the enterprise story.

## Why (and why now)

- **Strategic fit, not a detour.** Modern TTS (2024–2026) is *LLM + neural audio codec*. We already run the LLM
  half; the new work is one small conv-based codec decoder + glue — the same op family as the Whisper conv stem.
- **Marketing value is high** ("type text → it speaks, locally"), enterprise value is lower → sequence it **after**
  the buyable lines (M.E.AI / OpenAI API / RAG), as the viral finale.
- **Honest caveat:** quality is subjective (no byte-parity gate) and *voice cloning* carries real legal/abuse risk.
  So we split the work into a safe **preset-voice** track first and a gated **voice-cloning** Phase 2 (below).

## Status at a glance

| # | Stage | Deliverable | Status |
|---|---|---|:--:|
| **S0** | **Scaffolding** | contracts + WAV-out + watermark, model-free | ✅ **Done — 2026-06-06** |
| **S1** | **CLI + consent + enrollment** | `overfit tts` over a stub engine, consent gate, `VoiceProfile` persistence | ✅ **Done — 2026-06-06** |
| S2 | **SNAC codec decoder** | `Snac.Decode(codes) → PCM@24k` (the bulk) | ✅ **Done — 2026-06-06 (121.5 dB vs PyTorch)** |
| S3 | Orpheus LM glue | text → SNAC audio tokens (de-interleave) | ⬜ |
| S4 | End-to-end preset-voice TTS | `TextToSpeech.Synthesize(text, voice) → WAV` — first real speech | ⬜ |
| S5 | Quality & robustness | Polish text normalization, multiple voices, long text, sampling | ⬜ |
| S6 | Voice-loop demo + docs | `Demo/VoiceDemo`, mic → Whisper → LLM → TTS → speaker | ⬜ |
| **P2** | **Voice cloning** (zero-shot from a clip) | speaker enrollment → "my voice" — **gated on quality + legal** | ⬜ Phase 2 |

**Effort:** ~3–4 weeks for preset-voice end-to-end (S2 dominates). Voice cloning (P2) is materially harder — see
the model trade-off and the cloning note.

**Cross-cutting tooling — objective quality evaluator ✅ DONE 2026-06-06.** `Sources/Main/Audio/AudioSimilarity`
scores a candidate waveform against a reference ("ideal") one — the audio counterpart of the RAG stability harness,
turning "does it sound right?" into a measurable gate. Two views: *waveform domain* (SNR dB / Pearson correlation /
RMSE — sample-aligned, the right gate for a **deterministic codec decode vs. a reference decode**, i.e. the S2 gate)
and *spectral domain* (RMS log-mel distance + **DTW-aligned** mel distance — timing/length-robust, the right gate for
**generated speech vs. a reference clip**, i.e. S4). `AudioQualityAssert.Matches(...)` throws `AudioQualityException`
naming the breached metric (CI guard). CLI: `overfit tts eval --reference ideal.wav --candidate gen.wav`. Pure
managed, model-free, reuses `MelSpectrogram` + `AudioResampler`. 10 deterministic tests. **This retires the "no
objective CI gate" risk below** — we can now *measure* how close to ideal each decode is.

---

## Strategy: two tracks (decided)

1. **Track A — preset-voice local TTS (do first).** *"Your agent talks back, in .NET, no cloud."* A good fixed
   voice is already a killer demo and the voice loop, with **no licensing or deepfake landmine**. This is S1–S6.
2. **Track B — voice cloning ("my voice", P2).** Zero-shot from a short clip. The highest-wow, but: hardest port,
   license-encumbered models, and a hard **subjective quality bar + legal exposure**. Gate it behind a quality
   review and the consent/watermark posture — do **not** make "my voice" a product claim until it genuinely sounds
   right.

## Architecture

```
text ──► [TTS LM: Llama/Qwen arch] ──► audio-codec tokens ──► [neural-codec DECODER] ──► PCM ──► WAV
            (Overfit already runs this)        (de-interleave)      (the real new work)        (WavWriter ✅)
```

### Reuse map — what's already there vs new

| Need | Status |
|---|---|
| Transformer inference for the TTS LM (Llama-3.2 arch) | ✅ ships (`OverfitClient` / GGUF loader) |
| Tokenizer incl. the model's special audio-token vocab | ✅ GGUF embedded tokenizer handles extra vocab |
| Greedy / sampled decode loop with a custom stop token | ✅ exists; needs an audio-EOS stop + raw-token access |
| WAV writer @ any rate + streaming sink + watermark | ✅ **`WavWriter` / `WavAudioSink`** (S0) |
| Conv1d / transposed-conv / residual blocks | 🔶 have Conv (Whisper stem); need transposed-conv + the decoder graph |
| Vector-quantizer codebook lookup / dequantize | 🔶 new (small — embedding-table gather per level) |
| **SNAC decoder weights load + graph** | ✅ **done (S2) — `Snac.Decode`, 121.5 dB vs PyTorch** |
| Orpheus token format ↔ SNAC frame de-interleave | 🔴 new (documented offset scheme, S3) |

## Model choice (trade-off)

No single model has all three of *great "my voice" cloning · easy pure-C# port · permissive license*. Pick per
track:

| Model + codec | Zero-shot "my voice" (PL) | Pure-C# port | License |
|---|:--:|:--:|:--:|
| **Orpheus (Llama-3.2-3B) + SNAC** ← Track A | ❌ weak zero-shot | ✅ easiest (LM already loads; SNAC small conv) | ✅ permissive |
| **XTTS-v2** ← candidate for P2 | ✅ 6–30 s clip, Polish | ❌ heavy (DVAE + GPT + HiFi-GAN) | ❌ CPML (non-commercial) |
| **F5-TTS** ← candidate for P2 | 🟡 | ❌ flow-matching/diffusion (iterative, slow on CPU) | 🟡 verify |
| Piper / Kokoro | ❌ fixed voices | 🟡 (VITS / StyleTTS2 — different arch) | ✅ MIT / Apache |

**Decision:** Track A on **Orpheus + SNAC** (best architectural fit — the LM is already supported, the codec is a
small pure-managed conv decoder, permissive license). Revisit XTTS-v2 / F5 only for P2 cloning, accepting the
heavier port and the license caveat (or "you bring the model" posture).

## Decisions log

- **2026-06-06 — preset-voice first, cloning is Phase 2.** Quality + legal risk of cloning make it a gated
  follow-on, not the entry. Build the loop with a good preset voice first.
- **2026-06-06 — Track A = Orpheus + SNAC.** Reuses the existing engine; SNAC is a small, permissive, conv-only
  decoder.
- **2026-06-06 — pure-managed only.** No ONNX Runtime in the product path (it breaks the "no native binary" claim).
  An ONNX-RT prototype, if ever used, is explicitly a throwaway, not a shipped backend.
- **2026-06-06 — watermark is mandatory.** Every synthesized file carries a synthetic-speech provenance marker
  (`SyntheticSpeechMetadata`), per EU-AI-Act-style disclosure rules.

---

## Stage detail

### S0 — scaffolding ✅ **DONE 2026-06-06** (backend-agnostic, pure-managed, no model)
Model-independent surface so the CLI, enrollment, demos and the voice loop can be built/tested before any model is
ported:
- `Sources/Main/Audio/` — **`WavWriter`** (mono PCM16 / Float32; optional `LIST/INFO` provenance chunk) +
  `WavSampleFormat`.
- `Sources/Main/Audio/Tts/` — `ITextToSpeechEngine`, `IAudioSink` + **`WavAudioSink`** (streamed PCM → WAV),
  `VoiceProfile` (preset vs cloned, `IsCloned`), `TtsOptions`, **`SyntheticSpeechMetadata`** (the watermark).
- **8 model-free tests** green: PCM16 round-trips within quantization, Float32 exact, the marker embeds without
  breaking the audio, streamed chunks concatenate, a fake engine drives the contract end-to-end.

### S1 — CLI + consent + enrollment ✅ **DONE 2026-06-06** *(cheap, on-brand, no model)*
- `overfit tts --text "…" --out out.wav [--voice <id>] [--language pl]` — drives `PlaceholderTtsEngine` (a tone
  stand-in until S2–S4) → a **watermarked** WAV via `WavAudioSink`. Resolves an enrolled voice or falls back to a
  preset.
- `overfit voice enroll <id> --sample <wav> --language pl --consent` — validates the clip decodes, persists the
  `VoiceProfile` (via `VoiceProfileStore`, reflection-free manifest + embedding blob); the embedding is computed
  once the cloning backend (P2) lands. `overfit voice list` shows enrolled voices.
- **Consent gate:** enrollment requires `--consent` (own the voice / have permission) — refuses otherwise.
- **Validated live:** `tts` wrote a 24 kHz watermarked WAV; enroll-without-consent failed (exit 1); enroll-with
  consent persisted `maciej (pl, preset)` and `voice list` showed it. New: `PlaceholderTtsEngine`,
  `VoiceProfileStore`, CLI `tts`/`voice {enroll,list}`. 4 store + 2 engine tests green (suite 1151/0).

### S2 — SNAC decoder: codes → waveform ✅ **DONE 2026-06-06 — matches PyTorch to 121.5 dB SNR**
**The codec decoder works end-to-end in pure managed .NET.** Real `hubertsiuzdak/snac_24khz` weights decode real
codes to a waveform bit-identical (to float32 rounding: **SNR 121.5 dB, correlation 1.000, RMSE 0.0000**) with the
PyTorch reference noise-off decode. The whole graph is validated: codebook gather → `out_proj` → repeat-interleave +
cross-level sum → depthwise stem → transposed-conv upsampling → dilated residual units → output conv → `tanh`.
- **Offline convert** (`Scripts/convert_snac.py`, the only Python step): folds `weight_norm` into plain conv weights,
  exports decode-path tensors → `snac_24khz.safetensors` (canonical names) + a deterministic noise-off reference
  fixture (`codes.bin` + `reference_noiseoff.f32`) into `c:\snac`.
- **Native C#** (`Sources/Main/Audio/Tts/Snac/`): `Snac.Load(dir).Decode(int[][] codes) → float[]@24k`. Kernels
  `SnacConv` (grouped/depthwise `Conv1d` + `ConvTranspose1d`), `SnacActivations.Snake1d`, `SnacResidualVq`
  (codebook gather + repeat-interleave); `SnacConfig`/`SnacWeights`/`SnacDecoder` wire the graph. Reflection-free,
  loads via the existing native `SafetensorsReader`.
- **Tests:** 30 model-free unit tests (kernels exact vs scatter/naive references, Snake closed form, VQ ops) +
  `SnacDecoderParityTests` [LongFact] = the 121.5 dB end-to-end gate on `c:\snac`.
- **Noise:** the parity gate runs the deterministic (noise-off) path; SNAC's stochastic NoiseBlock is wired behind
  `Decode(codes, addNoise: true)` for real generation (its own deterministic Box–Muller RNG, not reproducible vs
  torch by design).
- **Building blocks** (all grounded verbatim in the real SNAC source, not assumed): `SnacConv` —
  `ConvTranspose1d` (gather formulation → race-free parallel; 13 tests incl. fast kernel = canonical scatter
  bit-for-bit) + grouped/depthwise `Conv1d`; `SnacActivations.Snake1d` (`x+(α+1e-9)⁻¹·sin(αx)²`); `SnacResidualVq`
  (`DecodeCodebook` + `RepeatInterleaveTime`). 30 model-free unit tests.
- **Gate met:** known codes → PCM matching the reference decode, measured by `AudioSimilarity` at **121.5 dB SNR**.
  Sound comes out of codes; the LM glue (S3) is next.

### S3 — Orpheus LM glue: text → audio tokens ⬜
- Load Orpheus GGUF (Llama-3.2-3B → loads today). Build the prompt format; decode the audio-token stream; stop at
  audio-EOS; de-interleave the 7-tokens-per-frame into SNAC's 3 levels.
- **Gate:** tokens in valid range, per-frame structure correct, generation terminates.

### S4 — End-to-end preset-voice TTS ⬜
- `TextToSpeech.Synthesize(text, voice) → WAV`: Orpheus → tokens → de-interleave → `SnacDecoder` → PCM → WAV.
- **Gate (subjective):** a sentence on the real model → intelligible speech. Keep a small reference-clip set.

### S5 — Quality & robustness ⬜
- Polish text normalization (numbers → words, abbreviations `np.`/`itd.`/`zł`, dates, punctuation, prosody),
  multiple voices, long text via sentence chunking + concatenation, silence trim, sampling vs greedy.
- **Gate:** several sentences across voices sound natural and stable.

### S6 — Voice loop demo + docs ⬜
- `Demo/VoiceDemo`: mic → Whisper (STT) → LLM → **TTS** → speaker, all in one .NET process. README / `docs/tts.md`.
- Optional: streaming codec decode (emit audio as frames arrive — low latency).

### P2 — Voice cloning ⬜ *(gated)*
- Speaker enrollment (clip → embedding) + a cloning-capable model (XTTS-v2 / F5). **Gate on:** a quality review
  ("does it actually sound like the target?") **and** the legal posture (consent record + watermark + jurisdiction
  check). Only then can "my voice" become a demo/claim.

---

## Risks & open questions

- **Subjective quality / uncanny valley.** A clone that's "85% you" is anti-viral. *Partly mitigated:* the
  `AudioSimilarity` evaluator (above) gives an **objective** gate — SNR/correlation for deterministic decodes, a
  DTW mel distance vs. reference clips for generated speech. It can't predict MOS (reference-free naturalness)
  without a model, so the final naturalness call stays a listen — but regressions are now caught by a number.
- **Codec weight format & tensor-name mapping** (S2) — straightforward but must be exact; an off-by-one in the
  residual-VQ level mapping = noise.
- **License & deepfake (P2).** Cloning models are often non-commercial (XTTS CPML); voice cloning without consent
  is increasingly regulated. Posture: *"Overfit provides the local TTS runtime; you are responsible for the model
  license and the voice rights."* Always watermark.
- **Polish quality (S5).** Text normalization + prosody matter as much as the model; without it, PL output trails
  ElevenLabs.

## Definition of done (preset-voice PoC)

`overfit tts --text "Cześć, tu Overfit. Ten głos powstał lokalnie w .NET." --out demo.wav` produces an
intelligible, watermarked 24 kHz WAV from the real Orpheus + SNAC stack, in pure managed .NET, with the full voice
loop (`Demo/VoiceDemo`) running mic → STT → LLM → TTS → speaker on one CPU box. Voice cloning remains explicitly
Phase 2.

## Out of scope (PoC)

Voice cloning beyond the gated P2; sub-200 ms real-time targets; non-LLM TTS (VITS / Tacotron / diffusion
vocoders); training a TTS model (we load a pre-trained one).
