# Artykuł na LinkedIn (wersja PL, wiralowa) — czerwiec 2026

---

## Wyrzuciłem Pythona z AI. Został czysty C#. I wiecie co? Działa.

Dwa lata temu usłyszałem zdanie, które słyszy każdy .NET-owiec:

**"Chcesz AI? To stawiaj Pythona."**

Sidecar z FastAPI. Ollama na boku. ONNX Runtime z natywnymi bibliotekami. Dane wyjeżdżające do API w chmurze. Cztery technologie, trzy języki, dwa zespoły — żeby aplikacja odpowiedziała na pytanie o własne dokumenty.

Postanowiłem sprawdzić, czy to naprawdę konieczne.

Tak powstał **Overfit** — silnik AI w 100% w C#. Nie wrapper na llama.cpp. Nie bindingi. Każda linijka — od parsera GGUF, przez kernele SIMD, po dekoder MP3 — napisana od zera w .NET.

### Co dziś potrafi (od najnowszych rzeczy):

🧩 **Serwer MCP** — jedna komenda i Claude Code rozmawia z Twoim prywatnym modelem, Twoimi dokumentami i Twoim audio. Korpus nie opuszcza maszyny.

🧪 **Lokalny LLM-sędzia** — Microsoft.Extensions.AI.Evaluation ocenia jakość odpowiedzi Twoim własnym modelem. Bez Azure, bez klucza, bez wysyłania czegokolwiek.

🧊 **Serwowanie, które śpi** — kontener między requestami siedzi na ~0% CPU. Pełna prędkość, gdy leci token.

🤖 **Qwen3, Phi-4 14B, Gemma 2** — wczytujesz GGUF-a z HuggingFace i rozmawiasz. W C#.

🗣️ **Pełna pętla głosowa** — Whisper (mowa→tekst, także po polsku) i synteza z klonowaniem głosu. Jeden proces, jeden CPU.

🎯 **Fine-tuning na laptopie** — QLoRA douczy prawdziwy, skwantyzowany model 3B Twoich prywatnych danych. Nauczyłem model zmyślonego faktu — wyrecytował go z pamięci. Bez GPU.

🛠️ **JSON, który nie może być zepsuty** — constrained decoding sprawia, że model fizycznie nie jest w stanie wylosować niepoprawnego JSON-a ani złej nazwy narzędzia.

🇵🇱 **Bielik w ASP.NET** — polski model, polski RAG z cytatami, polskie tool-calle. On-prem, w banku, za firewallem.

### Liczby (zmierzone, nie obiecane):

- **0 bajtów** alokacji na token — garbage collector się nie budzi
- **~24 tok/s** — model 3B na desktopowym CPU
- **34 MB** — cały serwer OpenAI-compatible jako obraz Dockera
- **~3 GB RAM** — fine-tuning modelu 3B
- **~60× szybciej niż czas rzeczywisty** — transkrypcja Whisperem
- **1260+ testów**, parzystość embeddingów z PyTorchem na poziomie cosine ≈ 1.0

### A teraz część, której nie znajdziecie w żadnym pitchu:

**llama.cpp dalej jest ~1,2× szybsze w surowym dekodowaniu. I my to piszemy we własnym README.**

Bo Overfit nie wygrywa na tokenach na sekundę. Wygrywa na wszystkim innym: Twoje dane nie opuszczają procesu, deploy to jeden binarek, stack to jeden język, a audytor dostaje odpowiedź "co dokładnie wykonuje ten kod" zamiast "no… tu jest 200 MB skompilowanego C++".

W regulowanych branżach — banki, ubezpieczenia, medycyna, obronność — to nie jest nice-to-have. To jest warunek wejścia.

### Po co o tym piszę?

Bo przez dwa lata słyszałem, że "się nie da". Że bez Pythona nie ma AI. Że bez GPU nie ma fine-tuningu. Że bez natywnych bibliotek nie ma wydajności.

Da się. Mam testy na dowód.

Jeśli budujesz .NET-owy produkt i potrzebujesz AI, które nie wyjdzie poza Twoją infrastrukturę — zajrzyj:

⭐ github.com/DevOnBike/Overfit
🌐 overfit-ml.com

`dotnet add package DevOnBike.Overfit`

A jeśli uważasz, że czegoś się "nie da" w .NET — napisz w komentarzu. Lubię takie listy.

#dotnet #csharp #AI #LLM #onprem #privacy #opensource

---

## Notki publikacyjne

- Hook = pierwsze zdanie nagłówka; LinkedIn ucina podgląd po ~2 liniach — "Wyrzuciłem Pythona z AI" musi być widoczne przed "zobacz więcej".
- Wiralowe dźwignie: (1) kontrarianizm wobec "AI = Python", (2) szczerość o llama.cpp jako pattern-interrupt, (3) CTA prowokujące komentarze ("napisz, czego się nie da") — komentarze >> lajki dla zasięgu.
- Najlepsze godziny PL: wt–czw 8:00–10:00.
- Pierwsze 60 min: odpowiadać na KAŻDY komentarz (algorytm).
- Grafika: infografika "We kept shipping" (spec w rozmowie) jako obraz artykułu.
- Wersja EN dostępna na życzenie — ten sam szkielet, hook: "I threw Python out of AI. Pure C# remained. It works."
