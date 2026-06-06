# VoiceLoop — local voice agent in one .NET process

`hear → think → speak`, entirely on the CPU, no cloud, no Python, no GPU:

```
microphone ─► Whisper (speech→text) ─► LLM (think) ─► Orpheus + SNAC (text→speech) ─► speaker
```

Everything runs in-process on your machine — **no audio or text ever leaves the box**. Mic capture and playback
use the built-in Windows `winmm` API via P/Invoke (no NuGet, no native binary shipped); Windows-only demo.

## Run

```powershell
VoiceLoop --chat <chat.gguf> --orpheus <orpheus.gguf> --snac <snac-dir> [--whisper <ggml-tiny.bin>]
```

Live, push-to-talk: press Enter to record, speak, and it answers out loud. Say "bye" to quit.

Defaults: `--whisper` → `%OVERFIT_WHISPER_DIR%`/`c:\whisper\ggml-tiny.bin`; `--orpheus` →
`%OVERFIT_ORPHEUS_DIR%`/`c:\orpheus\…`; `--snac` → `%OVERFIT_SNAC_DIR%`/`c:\snac`.

### Options

| Flag | Meaning |
|---|---|
| `--wav <file>` | Drive one turn from an audio file instead of the mic (testable, no hardware needed). |
| `--seconds <n>` | Mic record length per turn (default 5). |
| `--voice <id>` | Orpheus preset voice (tara / leah / jess / leo / dan / mia / zac / zoe). |
| `--once` | One turn then exit. |
| `--no-play` | Don't play the reply (e.g. headless). |
| `--save <out.wav>` | Save the spoken reply to a WAV (watermarked). |

## Notes

- **Models:** any small instruct GGUF for `--chat`; the Orpheus 3B GGUF + converted SNAC weights for speech
  (see `docs/tts-poc-plan.md` and `Scripts/convert_snac.py`); a whisper.cpp `ggml-*.bin` for STT.
- **Latency:** this is the **offline/batch** build — TTS runs at ~8–9× real-time on CPU, so replies are spoken
  after a short wait, not streamed live. Every spoken file carries a synthetic-speech provenance marker.
