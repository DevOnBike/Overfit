# Bielik — private Polish agent in .NET (setup & run)

Run [Bielik](https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct-GGUF) (a Polish LLM) as a private agent inside ASP.NET — Polish chat, RAG over Polish documents, C# tool calling and guaranteed JSON — **entirely in .NET, no Python, no Ollama, no model server, no data egress.**

The model and your documents never leave your machine.

---

## Prerequisites

- **.NET 10 SDK** (`dotnet --version` ≥ 10).
- **~4 GB free disk** (the Bielik Q4_K_M GGUF is ~2.7 GB; the MiniLM embedder ~90 MB) and **~4 GB free RAM** while running.
- **Python with `huggingface_hub`** — only to *download* the models (`pip install huggingface_hub`). It is not used at runtime.

No HuggingFace token is needed: the Bielik **GGUF** repo and the MiniLM repo are both public. (The Bielik *safetensors* repo is gated, but the demo doesn't use it.)

---

## 1. Download the two models

From the current folder (the scripts default to `C:\bielik` and `C:\minilm`; pass a path to override):

```powershell
.\download-bielik.cmd        # the LLM      — Bielik-4.5B-v3.0-Instruct Q4_K_M GGUF (~2.7 GB → C:\bielik)
.\download-embedder.cmd      # the embedder — all-MiniLM-L6-v2 (~90 MB → C:\minilm), used by RAG
```

The demo loads **two** models: the LLM answers, the embedder retrieves. You need both for the RAG endpoints; `/chat` and `/agent` work with just the LLM.

---

## 2. Run with the Bielik preset

```powershell
dotnet run -c Release --project Demo/LocalAgentAspNetDemo --launch-profile bielik
```

or run with powershell

```powershell
.\run-bielik.cmd
```

The `bielik` profile sets `ASPNETCORE_ENVIRONMENT=Bielik`, which loads [`appsettings.Bielik.json`](appsettings.Bielik.json):

- `ModelPath` → `C:\bielik\Bielik-4.5B-v3.0-Instruct-Q4_K_M.gguf` (tokenizer + ChatML template are read **from the GGUF** — no sibling files needed; Q4_K_M decodes ~1.24× faster than Q8_0),
- `EmbeddingModelPath` → `C:\minilm`,
- `DataPath` → `Data-pl` (the Polish documents: regulamin, polityka reklamacji, RODO, FAQ),
- a Polish system prompt.

If you put the models elsewhere, edit those paths in `appsettings.Bielik.json` (or set `OVERFIT_MODEL_DIR` / `OVERFIT_EMBEDDING_DIR`).

Wait for `Now listening on http://localhost:5234`. Open **`http://localhost:5234/swagger`** to try everything from the browser (request bodies are pre-filled), or use curl below.

---

## 3. Try it (Polish)

```powershell
# Health — confirms which model loaded.
curl http://localhost:5234/health

# Chat (Polish).
curl -X POST http://localhost:5234/chat -H "Content-Type: application/json" -d '{ "message": "Wyjaśnij krótko, czym jest rękojmia." }'

# Index the Polish documents (chunks + embeds into the in-process vector store).
curl -X POST http://localhost:5234/documents/index

# RAG over the Polish documents — grounded, with cited sources.
curl -X POST http://localhost:5234/rag/query -H "Content-Type: application/json" -d '{ "question": "Ile dni ma klient z UE na odstąpienie od umowy?", "topK": 4 }'
# → "Klient z UE ma 14 dni na odstąpienie od umowy, liczone od daty zakupu."

# Tool calling — a Polish request becomes a constrained C# tool call.
curl -X POST http://localhost:5234/agent -H "Content-Type: application/json" -d '{ "message": "Załóż zgłoszenie reklamacyjne o wysokim priorytecie dla klienta anna@firma.pl." }'
# → toolName: create_ticket, args { customerEmail, subject, priority }, dispatched + result

# Guaranteed JSON object.
curl -X POST http://localhost:5234/chat/json -H "Content-Type: application/json" -d '{ "message": "Zwróć obiekt JSON z polami imie i miasto dla Anny z Krakowa." }'
# → { "imie": "Anna", "miasto": "Kraków" }

# A business decision as guaranteed, typed JSON (field names English, reason in Polish).
curl -X POST http://localhost:5234/decision/refund -H "Content-Type: application/json" -d '{ "message": "Klient z UE kupił produkt 10 dni temu, nie był używany, chce odstąpić od umowy." }'
# → { "eligible": true, "reason": "Klient ma prawo do odstąpienia w ciągu 14 dni...", "requiredAction": "accept_refund", "confidence": 0.95 }

# Prometheus metrics (zero-allocation decode is visible here: overfit_allocated_bytes_total = 0).
curl http://localhost:5234/metrics
```

> **PowerShell note:** the backtick `` ` `` continues a line. In `cmd.exe` use `^` instead, and escape the inner quotes (`-d "{ \"message\": \"...\" }"`).

---

## What to expect

- **Coherent Polish** answers, with `allocatedBytes: 0` on every decode (zero-allocation hot path).
- **~11 tok/s** on a typical desktop CPU (Q4_K_M, 4.5B; Q8_0 is ~9 tok/s but larger). A good speed/quality balance for the demo.
- Tool calls carry **exactly** the right argument keys (constrained decoding), and JSON is **well-formed by construction**.

## Known limits (model, not bugs)

- **Borderline temporal questions** ("zwrot *po* 10 dniach") sit at the edge of a 4.5B's reasoning and can flip run-to-run — lead a live demo with clear factual questions ("Ile dni ma klient na odstąpienie?").
- **Retrieval ranking** uses MiniLM, which is English-centric and ranks Polish passages only roughly. Fine for the demo; a Polish/multilingual embedder would rank better.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not locate a model` at startup | Run `download-bielik.cmd`, or fix `ModelPath` in `appsettings.Bielik.json`. |
| `/documents/index` or `/rag/query` returns 400 about the embedding model | Run `download-embedder.cmd`, or set `EmbeddingModelPath` / `OVERFIT_EMBEDDING_DIR`. |
| `No module named huggingface_hub` | `pip install huggingface_hub`. |
| Very slow / high RAM | Q4_K_M needs ~3 GB resident; close other apps, or try a smaller quant (Q4_K_S / Q3_K_M). |
| Answers come out in English | The `bielik` profile sets a Polish system prompt — make sure you launched with `--launch-profile bielik`. |

---

## How it differs from the default demo

Everything (endpoints, metrics, code) is identical to the default English demo — the preset only swaps **the model, the documents, and the system prompt** via `appsettings.Bielik.json`. The same `OverfitClient.LoadGguf` path loads Bielik; the tokenizer is reconstructed from the GGUF's embedded vocabulary, and the constrained tool-calling / JSON paths work unchanged on its SentencePiece tokenizer.
