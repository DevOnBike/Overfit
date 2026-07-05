# LinkedIn — OverThink ("we go the other way")

Short, contrarian post centered on OverThink (an LLM running on your phone). Drop-in copy.

---

**No cloud. No GPU. No internet. Just a language model running on your phone.**

The whole industry is racing in one direction: more GPUs, more VRAM, bigger clusters, everything in the cloud.

We went the other way.

**OverThink** is a chat app that runs a real language model **entirely on your Android phone**. No server. No cloud. No API key. Turn on airplane mode — it still answers. Nothing you type ever leaves the device.

Under the hood it's **Overfit** — our language-model engine written in **pure C# / .NET**. No Python runtime, no native binary, no model server. It memory-maps a quantized GGUF model and decodes it on the CPU with near-zero allocation — the same engine whether it's a datacenter or the phone in your pocket.

Is it as fast as a datacenter GPU? Of course not — and that's not the point. The point is: your data stays yours, it works offline, and it costs nothing to run.

The most private AI isn't the one with the best security policy. It's the one that physically *can't* phone home.

Sometimes the interesting direction is the unfashionable one.

#dotnet #csharp #ai #llm #ondevice #edgeai #privacy

---

## Notes / variants
- **Viral title options** (swap the bold headline for any of these):
  1. *No cloud. No GPU. No internet. Just a language model running on your phone.* ← current
  2. *Everyone's buying bigger GPUs. I put a real AI on a phone — then turned the internet off.*
  3. *I asked an AI a question with my phone in airplane mode. It answered.*
  4. *The whole industry is renting GPUs. My AI runs on a phone with the Wi-Fi off.*
  5. *The most private AI is the one that physically can't phone home. So I built it.*
  6. *They said you need a datacenter to run AI. I fit one in my pocket.*
- **Proof line to add if you want numbers:** "~0.5B params, 4-bit, streaming tokens, fully offline — pure managed .NET on the CPU."
- **CTA options:** link the GitHub repo / a 10-second screen-recording of airplane-mode chat (the strongest asset — the plane icon in the status bar is the whole argument).
- Pair with the `store/linkedin/02-pitch.png` About-dialog screenshot, or a short clip of tokens streaming with Wi-Fi off.
