# Skrypt pokazu — prywatne AI w .NET (demo dla banku)

Czas: **~15 minut** + Q&A. Wszystko działa **lokalnie na laptopie prelegenta** — w trakcie pokazu nic nie
wychodzi do internetu (to jest puenta, nie ograniczenie). Język demo: **polski** (Bielik).

**Teza otwierająca (1 zdanie):** *„Pokażę, jak w istniejącym stacku .NET uruchomić prywatne AI — czat, RAG nad
waszymi dokumentami i audytowalne decyzje JSON — bez Pythona, bez GPU, bez chmury i bez wysyłania
czegokolwiek poza maszynę."*

---

## Checklista PRZED pokazem

**Dzień wcześniej:**
- [ ] **Merge `11sounds` → `main` + przepublikowany obraz** (Actions → „Docker Hub" → Run). ⚠️ Obraz sprzed
  merge'a generuje bełkot na Bieliku — bez tego kroku Akt 1 musi używać `overfit:local`.
- [ ] Release `demo-models` na GH opublikowany (Bielik-1.5B + Qwen3-0.6B + ggml-tiny + minilm).
- [ ] Na laptopie: Docker Desktop, .NET 10 SDK, repo sklonowane, modele w `C:\bielik`, `C:\qwen3-06b`,
  `C:\whisper`, `C:\minilm`.
- [ ] `docker pull devonbikeit/overfit:latest` (cache obrazu — pokaz nie zależy od Wi-Fi).
- [ ] Przejdź CAŁY skrypt raz na sucho na tym laptopie.

**30 minut przed:**
- [ ] `docker info` (daemon żyje), porty 8080/5234 wolne, zasilanie podpięte (CPU boost!).
- [ ] Zbuduj ASP.NET demo z wyprzedzeniem: `dotnet build -c Release Demo/LocalAgentAspNetDemo` (pierwszy
  build trwa — nie rób tego na oczach widowni).
- [ ] Terminal z dużą czcionką; drugi terminal na requesty.

---

## AKT 1 — „Cały serwer AI w 34 MB" (Docker, ~4 min)

> Narracja: *„To jest kompletny serwer LLM zgodny z API OpenAI. Obraz ma 34 megabajty — nie ma w nim
> Pythona ani runtime'u .NET, bo to jeden natywny binarek (Native AOT). Model montujemy z dysku."*

```powershell
# 1. Pokaż rozmiar obrazu (wow #1)
docker images devonbikeit/overfit

# 2. Odpal serwer z polskim modelem (Bielik-1.5B, 928 MB na dysku)
docker run --rm -p 8080:8080 -v C:\bielik:/models devonbikeit/overfit:latest /models/bielik-1.5b-v3.0-instruct-q4_k_m-imat.gguf
```

W drugim terminalu — pytanie po polsku, **zawsze `temperature: 0`** (deterministycznie = czysto na 1.5B):

```powershell
Invoke-RestMethod http://localhost:8080/v1/chat/completions -Method Post -ContentType 'application/json' `
  -Body '{"model":"bielik","messages":[{"role":"system","content":"Jestes pomocnym asystentem banku. Odpowiadaj krotko po polsku."},{"role":"user","content":"Wymien trzy korzysci z uruchamiania modeli AI lokalnie, na wlasnym sprzecie."}],"max_tokens":200,"temperature":0}' |
  ConvertTo-Json -Depth 10
```

> Puenta aktu: *„Każdy klient OpenAI — LangChain, Semantic Kernel, wasze istniejące integracje — działa z tym
> po zmianie jednego URL-a. A teraz proszę zauważyć: mogę wyłączyć Wi-Fi i nic się nie zmieni."* (Jeśli masz
> odwagę — wyłącz i powtórz request. Działa, bo wszystko jest lokalne.)

Zatrzymaj kontener (Ctrl+C) przed Aktem 2 (zwalnia RAM).

---

## AKT 2 — „RAG nad dokumentami + audytowalny JSON" (ASP.NET, ~7 min)

> Narracja: *„Teraz to samo jako biblioteka WEWNĄTRZ aplikacji ASP.NET — bez osobnego serwera modeli.
> Zaindeksujemy polskie dokumenty (regulamin, polityka reklamacji, procedura RODO) i będziemy zadawać
> pytania z cytowaniem źródeł."*

```powershell
# Bielik-1.5B zamiast domyślnego 4.5B z presetu:
$env:ModelPath = "C:\bielik\bielik-1.5b-v3.0-instruct-q4_k_m-imat.gguf"
cd Demo\LocalAgentAspNetDemo
.\run-bielik.cmd          # wstaje na http://localhost:5234, Swagger pod /swagger
```

**2a. Zaindeksuj dokumenty** (pokaż wcześniej pliki w `Data-pl\` — zwykłe markdowny):
```powershell
Invoke-RestMethod http://localhost:5234/documents/index -Method Post
```

**2b. Pytanie RAG po polsku — z cytowanymi źródłami** (RODO = język banku):
```powershell
Invoke-RestMethod http://localhost:5234/rag/query -Method Post -ContentType 'application/json' `
  -Body '{"question":"Gdzie domyslnie hostowane sa dane klientow i czy sa uzywane do trenowania modeli?"}' |
  ConvertTo-Json -Depth 10
```
> Pokaż w odpowiedzi pole ze źródłami: *„Model nie zmyśla — odpowiada z waszych dokumentów i mówi, z których."*

**2c. Gwarantowany JSON — decyzja biznesowa** (wow dla compliance). Para przykładów: ta sama struktura,
przeciwne (poprawne!) decyzje — regulamin daje 14 dni zwrotu na plan roczny:
```powershell
# w oknie 14 dni → eligible: true / accept_refund   (zweryfikowane: confidence 0.9)
Invoke-RestMethod http://localhost:5234/decision/refund -Method Post -ContentType 'application/json' `
  -Body '{"message":"Klient kupil ROCZNY plan Business 12 dni temu i prosi o pelny zwrot zgodnie z regulaminem."}' |
  ConvertTo-Json -Depth 10

# po oknie → eligible: false / reject   (zweryfikowane: confidence 0.9)
Invoke-RestMethod http://localhost:5234/decision/refund -Method Post -ContentType 'application/json' `
  -Body '{"message":"Klient kupil roczny plan Business 40 dni temu i prosi o pelny zwrot."}' |
  ConvertTo-Json -Depth 10
```
> ⚠️ Trzymaj się słowa „ROCZNY" w przykładzie — regulaminowe okno 14 dni dotyczy planów rocznych;
> nieprecyzyjny opis (bez typu planu) daje modelowi 1.5B dwuznaczność i niestabilną decyzję.
> Narracja: *„To nie jest «model zwykle zwraca JSON». Gramatyka jest wymuszana na poziomie tokenów — model
> FIZYCZNIE nie może wygenerować niepoprawnego JSON-a ani wymyślić pola. Każda decyzja ma stały schemat —
> do audytu, do logów, do systemu downstream."*

**2d. (opcja) Tool-calling:** `POST /agent` — model wywołuje funkcję C#.

> Puenta aktu: *„Jeden proces .NET. Jeden deploy. Jedna powierzchnia bezpieczeństwa. Zero egress."*

---

## AKT 3 (opcjonalny, ~2 min) — „Bank, który słyszy po polsku" (Whisper)

> Narracja: *„Rozpoznawanie mowy — też w czystym C#, też lokalnie, też po polsku."*

```powershell
dotnet run -c Release --project Demo/WhisperDemo -- C:\whisper\ggml-tiny.bin C:\whisper\polish.wav pl
```

Transkrypcja polskiego nagrania w ~1-2 s. *„Mowa → tekst → Bielik → odpowiedź: cała pętla głosowa w jednym
procesie, bez chmury."*

---

## Zamknięcie (slajd / 30 sekund)

- **Zero egress**: prompty, dokumenty i odpowiedzi nie opuszczają maszyny/serwerowni — z definicji, nie z polityki.
- **Audytowalność**: `temperature 0` = odpowiedzi powtarzalne bit-w-bit; JSON wymuszony schematem; cytowane źródła.
- **Wdrożenie**: NuGet (`DevOnBike.Overfit`), narzędzie (`dotnet tool install -g DevOnBike.Overfit.Cli`),
  kontener (`devonbikeit/overfit`, 34 MB) — on-prem / air-gapped.
- **Sprzęt**: ten pokaz działał na CPU laptopa. Bez GPU, bez Pythona, bez serwera modeli.
- Licencja: AGPL-3.0 / komercyjna (COMMERCIAL.md).

## Q&A — przewidywane pytania (uczciwe odpowiedzi)

| Pytanie | Odpowiedź |
|---|---|
| „Jak szybkie to jest?" | Na tym laptopie ~X tok/s (zmierz na sucho!). llama.cpp jest ~1.2× szybszy w surowym dekodzie — nasza przewaga to czysty .NET w procesie, −36% RAM, 0 B/token, trening/LoRA na CPU. |
| „Większe modele?" | Tak — 7B/14B na mocniejszym serwerze CPU (pokazywaliśmy Phi-4 14B); skala = RAM + rdzenie. |
| „Jakość 1.5B?" | Do RAG-Q&A i decyzji JSON wystarcza (co widzieliście). Do złożonych analiz — Bielik-11B na serwerze, ta sama ścieżka. |
| „Fine-tuning na naszych danych?" | QLoRA na CPU, w czystym .NET — bez GPU i bez wysyłania danych (docs/qlora-finetuning.md). |
| „Skąd model?" | GGUF z HuggingFace (Bielik = polski model Apache-2.0 od SpeakLeash) albo własny — plik na dysku, koniec zależności. |

## Fallbacki

- **Docker nie wstaje** → Akt 1 przez global tool: `overfit serve C:\bielik\bielik-1.5b-... --port 8080`.
- **Coś z obrazem z Huba** → `Sources\Cli\docker-build.cmd` dzień wcześniej i użyj `overfit:local`.
- **RAG odpowiada słabo na pytanie z sali** → wróć do przygotowanych pytań (1.5B + MiniLM-EN embedder ma
  granice na polszczyźnie; w produkcji: embeddingi Bielika albo większy model).
- **Brak czasu** → tnij Akt 3, nigdy 2c (JSON to najmocniejszy punkt dla banku).
