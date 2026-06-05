# Text-to-speech in pure .NET — PoC plan

**Goal.** Synthesize speech from text entirely in managed .NET on the CPU — no Python, no GPU, no native codec —
so Overfit closes the **voice loop**: Whisper (speech→text) + an LLM + **TTS (text→speech)**, all in one process.

**Why it's tractable.** Modern TTS (2024–2026) is *LLM + neural audio codec*, and Overfit already has most of both
halves. A transformer LM (which we run) predicts discrete **audio-codec tokens** from text; a small conv-based
**codec decoder** (which is the same op family as our Whisper conv stem) turns those tokens into a waveform.

```
text ──► [TTS LM: Llama/Qwen arch] ──► audio-codec tokens ──► [neural-codec DECODER] ──► PCM ──► WAV
            (Overfit already runs this)        (de-interleave)      (the real new work)        (we have WavWriter)
```

## Recommended path: Orpheus (Llama-3.2-3B) + SNAC

| Why this pair | |
|---|---|
| **Orpheus TTS** | a Llama-3.2-3B fine-tune that emits SNAC tokens — **Overfit loads `llama` arch today**, so the LM half is mostly reuse + glue. |
| **SNAC** (24 kHz) | a *small* (~19M-param) **conv-only** multi-scale residual-VQ decoder — no transformer in the decoder → pure-C# friendly, reuses our Conv/residual kernels. |

**Alternatives** (pick one model+codec pair and commit): **OuteTTS 0.3** (Qwen2.5/Llama) + **WavTokenizer** (single
codebook — simpler hierarchy than SNAC); **Llasa** (Llama) + XCodec. Avoid classic **VITS / Tacotron / Kokoro**
(StyleTTS2/ISTFT-net) — different architecture, little reuse of our engine.

## Reuse map — what's already there vs new

| Need | Status |
|---|---|
| Transformer inference for the TTS LM (Llama-3.2 arch) | ✅ ships (`OverfitClient` / GGUF loader) |
| Tokenizer incl. the model's special audio-token vocab | ✅ GGUF embedded tokenizer handles extra vocab |
| Greedy / sampled decode loop with custom stop token | ✅ exists; needs an audio-EOS stop + raw-token access |
| Conv1d / transposed-conv / residual blocks | ✅ have Conv (Whisper stem); need transposed-conv + the SNAC decoder graph |
| Vector-quantizer codebook lookup / dequantize | 🔶 new (small — embedding-table gather per level) |
| WAV writer @ 24 kHz | ✅ `WavReader`/writer in `Sources/Main/Audio/` |
| **SNAC decoder weights load + graph** | 🔴 **new — the bulk of the work** |
| Orpheus token format ↔ SNAC frame de-interleave | 🔴 new (well-documented offset scheme) |

---

## Session ladder

Each session ends with a concrete deliverable and a validation gate. Confirm exact tensor names / token offsets
against the model card + source at the start of each stage (TTS repos evolve).

### S1 — SNAC decoder: codes → waveform *(the foundation, biggest chunk)*
- Load SNAC weights (`model.safetensors` / the codec's checkpoint), map tensor names.
- Build the decoder graph: per-level **codebook dequantize** → **transposed-conv upsampling** + **dilated residual
  units** → final conv → `tanh` → 24 kHz mono PCM (the standard SNAC decoder topology).
- New kernels as needed: `ConvTranspose1d`, dilated `Conv1d` residual unit (reuse the Conv core).
- **Gate:** feed a *known* set of SNAC codes (from a reference encode of a test clip, or a published test vector)
  → decoded PCM matches the reference decode within tolerance / is clean audible audio. **Get sound out of codes
  before touching the LM.**
- Deliverable: `SnacDecoder.Decode(codes) -> float[] (PCM@24k)` + WAV out.

### S2 — Orpheus LM glue: text → audio tokens
- Load Orpheus GGUF (Llama-3.2-3B → loads today). Build the prompt format (voice id / style + text + control tokens).
- Decode the audio-token stream greedily; stop at the audio-EOS token.
- De-interleave the **7 tokens-per-frame** into SNAC's 3 hierarchical levels (the `<custom_token_…>` offset map).
- **Gate:** emitted tokens are in valid range, the per-frame structure is correct, generation terminates.
- Deliverable: `text -> int[][] snacCodesPerLevel`.

### S3 — End-to-end + facade
- `TextToSpeech.Synthesize(text, voice) -> WAV`: Orpheus LM → audio tokens → de-interleave → `SnacDecoder` → PCM → WAV.
- **Gate (subjective):** synthesize a sentence on the real model → **intelligible speech**. (No byte-parity here —
  TTS quality is judged by listening; keep a small set of reference clips.)
- Deliverable: one-call TTS facade, validated on a real model.

### S4 — Quality & robustness
- Multiple voices; punctuation / numbers / abbreviations; long text via sentence chunking + concatenation;
  leading/trailing-silence trim; sampling (temperature/top-p) for naturalness vs greedy flatness.
- **Gate:** a handful of sentences across voices sound natural and stable (no dropouts / runaway).

### S5 — *(optional)* Streaming + demo + the full voice loop
- **Streaming codec decode** — emit audio as SNAC frames arrive (low latency for voice agents).
- `Demo/TtsDemo` CLI; README / `docs/tts.md`.
- **Full local voice loop:** mic → Whisper (STT) → LLM → **TTS** → speaker, entirely in one .NET process.

**Total:** ~3–4 weeks (≈ 6–9 sessions). S1 (SNAC decoder) dominates; the LM half is largely existing inference.

---

## Risks & open questions

- **Quality validation is subjective.** Unlike text (byte-parity) or retrieval (deterministic), "does it sound
  right" needs listening. Mitigate with a fixed set of reference sentences + spectrogram diffs against a reference
  decode; accept that the CI gate is weaker here.
- **Codec weight format & tensor-name mapping** (SNAC ships PyTorch/safetensors) — straightforward but must be exact.
- **Token offset scheme** (Orpheus' audio-token base offsets + 7-per-frame interleave) — documented, but confirm
  against the current model release.
- **Residual-VQ correctness** — each of SNAC's levels dequantizes from its own codebook at its own rate; an off-by-
  one in level mapping = noise.
- **Streaming** (S5) adds real-time buffering complexity — defer past the batch PoC.

## Out of scope (for the PoC)

- Voice cloning / zero-shot speaker adaptation (a later capability on top).
- Real-time / sub-200 ms latency targets (S5 streaming is a first step, not a guarantee).
- Non-LLM TTS architectures (VITS, Tacotron, diffusion vocoders).
- Training a TTS model — we load and run a pre-trained one.

## Strategic note

TTS is a **vertical** (call-center transcription+response, accessibility, voice assistants, on-prem voice agents),
not a core-adoption feature — consistent with Overfit's positioning. Its unique value is the same column as the
rest of the product: **the whole voice loop in-process in .NET, on the CPU, with no Python and no data egress.**
Build it when a customer pulls for it; the Orpheus + SNAC path keeps the cost to "implement one small conv decoder
+ glue", because the LM half is already shipped.
