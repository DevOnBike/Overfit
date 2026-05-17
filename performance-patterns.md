# Best practices wydajnościowe Microsoftu — z dekompilacji VS 2022

Źródło: dekompilacja `.NET` DLL z `C:\Program Files\Microsoft Visual Studio\2022\Community\` narzędziem `ilspycmd` (ICSharpCode.Decompiler 10.0.1). Wyniki zsyntetyzowane z 10 kluczowych assembly:

- `Microsoft.CodeAnalysis.dll` / `Microsoft.CodeAnalysis.CSharp.dll` (Roslyn)
- `Microsoft.VisualStudio.Threading.dll` / `Microsoft.VisualStudio.Validation.dll`
- `Microsoft.VisualStudio.Text.Data.dll` / `Microsoft.VisualStudio.Text.Logic.dll`
- `System.Collections.Immutable.dll` / `System.Memory.dll`
- `Newtonsoft.Json.dll` / `StreamJsonRpc.dll`

Łącznie ~3690 plików `.cs` zdekompilowanych do `C:\Temp\vsdecomp\`.

---

## 1. Object pooling — fundament (740+ trafień)

### Kanoniczny `ObjectPool<T>` (Roslyn)

Źródło: `Microsoft.CodeAnalysis.PooledObjects.ObjectPool.cs`

- **"First item" fast path** — oddzielne pole `_firstItem` + tablica zapasowa `_items[ProcessorCount * 2 - 1]`.
- **Lock-free** przez `Interlocked.CompareExchange` — nigdy `lock`.
- **Hot/Slow split** — `Allocate` (inline-able) wywołuje `AllocateSlow` przy missie. JIT może wciągnąć tylko gorącą ścieżkę.
- `[Conditional("DEBUG")]` na walidacji → zerowy IL w Release.

```csharp
internal T Allocate() {
    T val = _firstItem;
    if (val == null || val != Interlocked.CompareExchange(ref _firstItem, null, val))
        val = AllocateSlow();
    return val;
}
```

### Pooled wrappers — wzorzec do skopiowania

Źródło: `PooledStringBuilder.cs`, `PooledHashSet.cs`, `PooledDictionary.cs`, `ArrayBuilder.cs`

- `GetInstance()` / `ToStringAndFree()` — idiom wymuszający zwrot.
- **Discard-when-too-big** — `if (Builder.Capacity <= 1024) { Clear(); Free(); }` żeby duże instancje nie blokowały pamięci w puli.
- `[Obsolete("Consider calling ToStringAndFree instead.")]` na `ToString()` żeby developer nie zapomniał zwrócić.

---

## 2. `readonly struct` jako default (217 trafień / 147 plików)

Praktycznie każdy typ wartościowy w Roslyn/VS jest `readonly struct`. Eliminuje to obronne kopie kompilatora. Przykłady: `SyntaxToken`, `SyntaxTrivia`, `TextSpan`, `LinePosition`, `TypeWithAnnotations`.

**Rule of thumb**: jeśli piszesz nowy value type, zacznij od `readonly struct` i zmień tylko gdy kompilacja się nie powiedzie.

---

## 3. `[MethodImpl(AggressiveInlining)]` — celowo, nie wszędzie (250 trafień / 40 plików)

Skupione w:
- `System.Memory` (Span/ReadOnlySpan acessory, MemoryMarshal, BinaryPrimitives).
- `System.Collections.Frozen` (table lookups).
- Indeksery `SegmentedArray`.

**Lekcja**: stosować na geterach/indekserach/operatorach matematycznych — NIE na wieloliniowych metodach (JIT i tak je rozważy).

---

## 4. Two-tier hash table (`StringTable.cs` w Roslynie)

Dedupe stringów z dwoma poziomami:

| Tier | Rozmiar | Zakres | Synchronizacja |
|------|---------|--------|---|
| `_localTable` | 2048 | per-instancja (pooled) | brak (thread-local) |
| `s_sharedTable` | 65536 (bucket=16) | statyczna | lock-free CAS |

- Wszystkie rozmiary potęgi 2 (`& mask` zamiast `%`).
- `Environment.TickCount` jako początkowy random dla wyboru bucketu — unika patologicznych chain.

---

## 5. Segmented arrays — unikanie LOH

Źródło: `Microsoft.CodeAnalysis.Collections.SegmentedArray<T>`

`T[][]` (jagged) zamiast `T[]`. Każdy segment < 85kB → omija Large Object Heap. Używane przez `SegmentedDictionary`, `SegmentedHashSet` dla dużych kolekcji w Roslynie.

**Why:** alokacje > 85kB trafiają na LOH, który: (a) nie jest kompaktowany domyślnie, (b) generuje fragmentację, (c) długo żyje.

---

## 6. Frozen collections — kod-generacja per kształt klucza

`System.Collections.Frozen` zawiera kilkadziesiąt specjalizowanych podklas:

- `OrdinalStringFrozenDictionary_LeftJustifiedSingleChar`
- `OrdinalStringFrozenDictionary_LeftJustifiedSubstring`
- `OrdinalStringFrozenDictionary_RightJustifiedCaseInsensitiveAsciiSubstring`
- `Int32FrozenDictionary`, `SmallValueTypeDefaultComparerFrozenDictionary`...

Przy `ToFrozenDictionary()` analizowane są dane i wybierany najszybszy wariant — koszt build'u jednorazowy, read O(1) z minimalnymi instrukcjami.

---

## 7. Lock-free synchronizacja (738 trafień / 57 plików)

- **`Interlocked.CompareExchange`** dominuje na hot path.
- **`Volatile.Read/Write`** zamiast `lock` przy single-reader.
- **`ImmutableInterlocked`** — atomicowe operacje na immutable structures.
- VS Threading ma **własne**: `AsyncReaderWriterLock`, `AsyncSemaphore`, `ReentrantSemaphore`, `JoinableTaskFactory` (vs BCL — bo BCL nie radzi sobie z UI thread coordination).
- `ConfigureAwait(false)` użyte 75 razy w samym `Microsoft.VisualStudio.Threading`.

---

## 8. Span/stackalloc tylko punktowo (10 trafień)

Wbrew hype'owi — **Span jest sparingly używany w wyższych warstwach**. Pojawia się w:
- Serializacji JSON-RPC (`TraceParent`, `HeaderDelimitedMessageHandler`).
- Parserach (`BoundStackAllocArrayCreation`, `SyntaxFacts`).

**Wniosek**: `Span<T>` ma sens gdy faktycznie eliminuje alokację w gorącej pętli — nie jest cudowną zamianą `string`/`array` wszędzie.

---

## 9. Drobne ale klasyczne

- **`[Conditional("DEBUG")]`** na walidatorach (Validate, AssertInvariants) — 0 IL w Release.
- **Boxing avoidance** — `Microsoft.CodeAnalysis.Boxes` z `static readonly object BoxedTrue/BoxedFalse`.
- **`SharedStopwatch`** — jeden statyczny `Stopwatch`, nie alokuje per pomiar.
- **`ReaderWriterLockSlimExtensions`** — RAII disposable helper, eliminuje `try/finally` przy lock'u.
- **`EmptyStruct`** — placeholder dla generic'ów z zerowym footprint.
- **`WeakKeyDictionary`** — cache który nie trzyma przy życiu.

---

## TL;DR — adopcja

1. **Pool wszystko co alokujesz w pętli** — StringBuilder, List, Dictionary. Wzór z `PooledStringBuilder` + `ObjectPool` to template do skopiowania.
2. **`readonly struct` jako default** dla nowych value types.
3. **`Interlocked.CompareExchange` na "first item"** zamiast lock'u w puli/cache.
4. **Discard pooled object po przekroczeniu progu rozmiaru** — inaczej leak.
5. **Hot path → oddzielna metoda `*Slow`** — pomaga JIT-owi inline'ować szybką ścieżkę.
6. **Tablice > LOH? → segmented (jagged)**.
7. **`[Conditional("DEBUG")]` na assertach** zamiast `if (DEBUG)`.
8. **Frozen collections** dla read-only lookup table'i ładowanych raz na start.

---

---

# Część 2 — druga pula 14 DLL

Dodatkowe assembly: `Microsoft.VisualStudio.Composition`, `Microsoft.VisualStudio.Telemetry` (821 plików), `Microsoft.VisualStudio.RpcContracts`, `Microsoft.VisualStudio.Utilities` (452 pliki), `Microsoft.VisualStudio.Shell.Framework` (373 pliki), `Microsoft.VisualStudio.Text.UI` (447 plików), `Microsoft.VisualStudio.LanguageServer.Protocol`, `Microsoft.VisualStudio.Imaging`, `Microsoft.ServiceHub.Client`, `Microsoft.Bcl.AsyncInterfaces`, `Nerdbank.Streams`, `MessagePack`, `System.IO.Pipelines`, `System.Text.Json`.

## 10. `ArrayPool<byte>.Shared` jako standard dla buforów I/O

Źródło: `System.Text.Json.PooledByteBufferWriter.cs`

Wzorzec idealny do skopiowania:

```csharp
public PooledByteBufferWriter(int initialCapacity) {
    _rentedBuffer = ArrayPool<byte>.Shared.Rent(initialCapacity);
    _index = 0;
}

public void Dispose() {
    if (_rentedBuffer != null) {
        _rentedBuffer.AsSpan(0, _index).Clear();   // czyść przed zwrotem (sensitive data + GC roots)
        ArrayPool<byte>.Shared.Return(_rentedBuffer);
        _rentedBuffer = null;
    }
}

private void CheckAndResizeBuffer(int sizeHint) {
    // strategia growth: doubling, ale z guard'em na 2GB
    int num4 = num + Math.Max(sizeHint, num);
    ...
    _rentedBuffer = ArrayPool<byte>.Shared.Rent(num4);   // znowu z poola
    // skopiuj stare → zwróć stare do poola → kontynuuj na nowym
}
```

**Kluczowe**: Clear() przed Return() — w przeciwnym razie GC może trzymać przy życiu obiekty referencjowane przez bufor.

## 11. `Lazy<T>` jako fundament MEF/Composition (43 trafień)

`Microsoft.VisualStudio.Composition.LazyServices` + `Roslyn.Utilities.RoslynLazyInitializer` używają `Lazy<T>` z `LazyThreadSafetyMode.PublicationOnly` masowo — komponenty MEF są **leniwie inicjalizowane**.

**Lesson**: VS startuje szybko bo praktycznie nic nie ładuje na starcie. Każdy serwis to lambda w `Lazy<>` lub `ExportFactory<>`. Pierwsze użycie funkcji = pierwsza inicjalizacja zależności. Aplikuj to do każdego "dużego" obiektu w swoim systemie.

## 12. `ConcurrentDictionary.GetOrAdd` jako default cache (700+ trafień)

Najczęściej widziany pattern cache'owania:

```csharp
private readonly ConcurrentDictionary<Type, IFormatter> _cache = new();

public IFormatter GetFormatter(Type t) =>
    _cache.GetOrAdd(t, static type => CreateFormatter(type));
```

Konkretne lokalizacje:
- `MessagePack.Resolvers.CachingFormatterResolver` — cache formatterów per typ.
- `MessagePack.Internal.ThreadsafeTypeKeyHashTable` — własna, lepsza od ConcurrentDictionary dla `Type` jako klucz (custom hash).
- `System.Text.Json.ReflectionEmitCachingMemberAccessor` — IL-emit raz, cache delegate na zawsze.
- `Roslyn.Utilities.ConcurrentDictionaryExtensions` — własne rozszerzenia GetOrAdd.

**Tip z MessagePack**: gdy klucz jest `Type`, własna hashtable z `RuntimeHelpers.GetHashCode` bije ConcurrentDictionary.

## 13. `ConcurrentLruCache<K,V>` — gdy GetOrAdd nie wystarcza

Źródło: `Microsoft.CodeAnalysis.InternalUtilities.ConcurrentLruCache.cs`

Gdy potrzebujesz **ograniczonego** cache'a (eviction LRU), Roslyn używa **Dictionary + LinkedList + lock** zamiast Concurrent. Dlaczego nie `ConcurrentDictionary`?
- LRU wymaga atomowej operacji na DWÓCH strukturach (mapa + lista).
- `lock` z krótkimi sekcjami krytycznymi jest tu szybszy niż double-CAS.

```csharp
private readonly Dictionary<K, CacheValue> _cache;
private readonly LinkedList<K> _nodeList;
private readonly object _lockObject = new object();
```

## 14. `PipeAwaitable` — custom struct-based awaiter (zero allocation)

Źródło: `System.IO.Pipelines.PipeAwaitable.cs`

Jeden z najfajniejszych wzorców niskopoziomowych:

- **`internal struct PipeAwaitable`** — NIE klasa, NIE Task. Sam awaiter to value type.
- `[Flags] AwaitableState { None, Completed=1, Running=2, Canceled=4, UseSynchronizationContext=8 }` — 4 stany w jednym intie, bit operations.
- `[MethodImpl(AggressiveInlining)]` na **każdej** metodzie zmieniającej stan.
- `ValueTaskSourceOnCompletedFlags` zamiast pełnego Task — `ValueTask` nie alokuje.
- `cancellationToken.UnsafeRegister(...)` (zamiast `Register`) — pomija przechwyt `ExecutionContext` (10x szybsze).

**To pokazuje**: gdy faktycznie potrzebujesz zero-allocation async (millions ops/sec), implementujesz własny `IValueTaskSource` na strukturze.

## 15. `ValueTask` w hot paths (85 trafień / 48 plików)

`ValueTask` zamiast `Task` gdy:
- Operacja często kończy się synchronicznie (dane już w buforze).
- Hot path z wieloma wywołaniami na sekundę.

Klasyk: `PipeReader.ReadAsync` zwraca `ValueTask<ReadResult>` — przy danych w buforze nie ma alokacji `Task`.

**Roslyn ma `ValueTaskFactory`** — prebuilt `ValueTask` dla typowych przypadków (TrueResult, FalseResult, EmptyResult).

## 16. `IAsyncDisposable` / `DisposeAsync` (152 trafień / 55 plików)

Wszystko co alokuje async resources (streams, pipes, connections, lock'i) implementuje `IAsyncDisposable`. Stary `IDisposable.Dispose()` byłby blokujący — `DisposeAsync` pozwala na czysty cleanup bez block'owania wątku.

VS Threading wprowadza **własny** `Microsoft.VisualStudio.Threading.IAsyncDisposable` (przed standardem) — pokazuje jak ważny był to wzorzec.

## 17. `CancellationToken` propagacja przez IPC

Źródło: `StreamJsonRpc.StandardCancellationStrategy`, `Microsoft.VisualStudio.Threading.CancellableJoinComputation`

Cancel w RPC nie ma natywnego wsparcia w JSON-RPC. Microsoft implementuje:
- Każde wywołanie dostaje correlation ID.
- Klient anuluje → wysyła `$/cancelRequest` z tym ID.
- Server tłumaczy to na lokalny `CancellationToken`.

**Key insight**: nie ignoruj `CancellationToken` na granicy procesu — przekaż go.

## 18. `cancellationToken.UnsafeRegister` zamiast `Register` (167 trafień na CancellationTokenSource)

`UnsafeRegister` pomija przechwyt `ExecutionContext` w `CallbackNode`. Używać gdy callback nie polega na `AsyncLocal`/`CallContext` — czyli **prawie zawsze** w infrastruktrze. Roslyn, Pipelines, StreamJsonRpc, Nerdbank — wszystkie go używają.

## 19. `MultiplexingStream` — wiele logicznych strumieni w jednym fizycznym

Źródło: `Nerdbank.Streams.MultiplexingStream.cs` (41 trafień PipeReader)

Jak VS multipleksuje wiele kanałów RPC przez jeden named pipe / socket:
- Każdy logical channel ma 8-bajtowy header (ID + flags + length).
- Wewnętrznie `PipeReader`/`PipeWriter` per channel + frame demuxer.
- Backpressure działa per-channel — wolny channel nie blokuje innych.

**Tym sposobem VS odpala dziesiątki out-of-proc serwisów (Roslyn server, LSP servers) na 1-2 fizycznych połączeniach**.

## 20. Batching telemetrii — `TaskTimer` + `PersistenceTransmitter`

Źródło: `Microsoft.VisualStudio.Telemetry`

Zamiast wysyłać każde event natychmiast:
- `TelemetryCollector` agreguje w buforze.
- `TaskTimer` (custom timer ze `CancellationTokenSource.CancelAfter`) — flush co N sek.
- `PersistenceTransmitter` — jeśli sieć nie działa, pisz na dysk, retry później.

**Pattern**: nigdy nie blokuj hot path I/O. Batch + delay + persist + retry.

## 21. `Channel<T>` — nieobecne, Microsoft idzie w `Pipe`/`PipeReader`

Co ciekawe: **0 trafień `Channel<T>`** w 24 DLL. Microsoft Visual Studio używa **`System.IO.Pipelines.Pipe`** (binary streaming) zamiast `System.Threading.Channels.Channel<T>` (typed messaging).

Powód: dla komunikacji binarnej IPC `Pipe` jest niżej i bardziej kontrolowalny (rozmiar bufora, backpressure, zero-copy).

`Channel<T>` ma sens dla in-process producer/consumer z typami .NET. Pipe ma sens dla bytes-over-wire.

---

## TL;DR część 2 — dodatkowe wzorce

| # | Wzorzec | Kiedy |
|---|---------|-------|
| 10 | `ArrayPool<byte>.Shared.Rent` + Clear-on-return | Każdy bufor > 1KB w hot path |
| 11 | `Lazy<T>` z `PublicationOnly` | Każdy "ciężki" singleton/serwis |
| 12 | `ConcurrentDictionary.GetOrAdd` static lambda | Default cache (Type → Formatter etc.) |
| 13 | Custom LRU z `lock` | Bounded cache z eviction |
| 14 | `IValueTaskSource` na strukturze | Hot async path bez Task alloc |
| 15 | `ValueTask` zamiast `Task` | Sync-completing async methods |
| 16 | `IAsyncDisposable` | Resources wymagające async cleanup |
| 17 | RPC correlation ID + reverse-cancel | Cancel cross-process |
| 18 | `UnsafeRegister` na CT | Gdy nie potrzebujesz ExecutionContext |
| 19 | Frame-multiplexing | Wiele logical channels nad 1 socket |
| 20 | Batch + timer + persist | I/O telemetry, logging, analytics |
| 21 | `Pipe` > `Channel<T>` | Binary IPC; `Channel<T>` dla in-proc typed |

---

*Część 2 dopisana 2026-05-15. Łącznie 24 DLL, ~7300 plików `.cs` w `C:\Temp\vsdecomp\`. Brak `Roslyn.Utilities.dll` jako osobny plik — zawartość siedzi wewnątrz `Microsoft.CodeAnalysis.dll`.*

---

# Część 3 — masowy scan 1000 DLL (252 039 plików `.cs`)

Po przeskanowaniu 1000 największych managed Microsoft/System DLL z VS 2022 (BCL, ASP.NET Core, Kestrel, Roslyn full stack, ML.NET, Build, MessagePack, Cosmos, etc.) wyłaniają się **nowe** klasy wzorców, niewidoczne w pierwszej puli 24 assembly.

## 22. `[SkipLocalsInit]` — pomijanie zerowania lokalnych (1068 trafień / 134 plików)

Najczęstszy "ukryty" trick perf w nowym kodzie Microsoftu. CIL ma flagę `init` wymuszającą zerowanie wszystkich lokalnych zmiennych. `[SkipLocalsInit]` ją wyłącza → JIT generuje mniej instrukcji.

**Critical**: nie używaj jeśli czytasz lokalne przed zapisem (UB). W praktyce bezpieczne dla:
- Metod intensywnie używających `stackalloc` (bez tego stackalloc zeruje się gratis).
- Hot paths z prostymi value-type locals zapisywanymi przed odczytem.

**Praktyka Microsoftu**: applied at **assembly level** w `AssemblyInfo.cs`. Każdy z VS Copilot, CoreUtility, Extensibility, LanguageServer, SolutionPersistence, ProjectSystem, Completions ma własny polyfill `SkipLocalsInitAttribute.cs` (bo atrybut wymaga .NET 5+, więc downlevel projekty go sobie deklarują wewnętrznie — wystarczy, kompilator C# rozpoznaje po nazwie).

## 23. `[InlineArray(N)]` (.NET 8) — fixed-size buffer w strukcie

47 trafień w 46 plikach. Pliki `__InlineArray2.cs`, `__InlineArray8.cs`, `__InlineArray38.cs` — to **kompilator C# emituje** te typy gdy widzi `Span<T> stackalloc[N]` patterns lub `[InlineArray]` ręcznie.

**Konkretne użycie**: `Microsoft.AspNetCore.Razor.Utilities.Checksum` przechowuje 32-byte hash jako `InlineArray<32, byte>` w strukcie — fixed-size content addressing bez heap alloc. Roslyn LSP Protocol, Razor, Workspaces, Features, Copilot — wszystkie używają.

**Lekcja**: dla fixed-size data (hash, UUID, fixed packets) `[InlineArray]` zastępuje `unsafe fixed byte[N]` z pełnym typesystem support.

## 24. `[ModuleInitializer]` — run-once code na load assembly (32 trafień)

Zastępuje statyczny konstruktor klasy gdy chcesz kod uruchamiany **raz, gdy assembly się ładuje**, niezależnie od pierwszego użycia jakiejkolwiek klasy.

```csharp
internal static class StartupInitializer {
    [ModuleInitializer]
    public static void Init() {
        // np. rejestracja DI, codec registration, SQLite native dll preload
    }
}
```

Konkretni użytkownicy: Copilot (Roslyn, CodeMappers, Common, Service, Vsix, UI.Core), Extensibility Framework, SolutionPersistence, ProjectSystem, **Microsoft.VisualStudio.Cache.SqliteInitializer** (preload native sqlite.dll), `Microsoft.Windows.SDK.NET.ProjectionInitializer` (WinRT projections).

## 25. SIMD intrinsics — gdzie naprawdę żyją (391 trafień / 32 plików)

SIMD jest **bardzo selektywnie** używane. Praktycznie tylko:

| Plik | Co robi przez SIMD |
|------|---|
| `System.Private.CoreLib\System.Text\Ascii.cs` (38) | walidacja/konwersja ASCII bytes |
| `System.Private.CoreLib\System.Text.Unicode\Utf8Utility.cs` (23) | UTF-8 validation/transcoding |
| `System.Private.CoreLib\System.Buffers\IndexOfAnyAsciiSearcher.cs` (6) | wielo-bajtowe IndexOfAny |
| `System.Private.CoreLib\System.Buffers\ProbabilisticMap.cs` (10) | string search |
| `System.Private.CoreLib\System.Buffers.Text\Base64.cs` (14) | Base64 encode/decode |
| `System.Private.CoreLib\System\HexConverter.cs` (9) | hex encoding |
| **`Kestrel.StringUtilities.cs` (25)** | hex-encoding HTTP connection IDs przez `Ssse3.Shuffle` |
| `System.Text.Encodings.Web\OptimizedInboxTextEncoder.cs` (46) | HTML/JS escape |
| `System.Numerics.Tensors\TensorPrimitives.cs` (57) | ML/AI vector math |

Wzór z Kestrel (`StringUtilities.cs:61-68`):
```csharp
if (Ssse3.IsSupported) {
    Vector128<byte> vector = Ssse3.Shuffle(
        Vector128.CreateScalarUnsafe(item3).AsByte(),
        Vector128.Create(15,15,3,15,...).AsByte()); // permutacja nibbli
    // ...mask, shuffle table "0123456789ABCDEF"... 
    Unsafe.WriteUnaligned(ref ..., left);  // 16 bajtów w jednej instrukcji
}
else { /* scalar fallback */ }
```

**Pattern**: SIMD ścieżka + scalar fallback, gated przez `Ssse3.IsSupported`/`Avx2.IsSupported`/`AdvSimd.IsSupported`. **Microsoft nie używa SIMD w VS code** — tylko w BCL i Kestrel (request parsing).

## 26. `ValueStringBuilder` — `ref struct` zamiast `StringBuilder`

Stack-allocated string builder z fallbackiem na `ArrayPool` przy growth:

```csharp
internal ref struct ValueStringBuilder {
    private char[] _arrayToReturnToPool;
    private Span<char> _chars;
    private int _pos;

    public ValueStringBuilder(Span<char> initialBuffer) {  // typowo: stackalloc char[256]
        _arrayToReturnToPool = null;
        _chars = initialBuffer;
        _pos = 0;
    }
    // Grow() → ArrayPool.Rent → CopyTo → Return stary
    public override string ToString() {
        string result = _chars.Slice(0, _pos).ToString();
        Dispose();  // zwraca rented bufor
        return result;
    }
}
```

**Microsoft `polyfilluje` ten typ w masie projektów**: ASP.NET HttpLogging, Http.Extensions, Razor, Roslyn (jako `PooledStringBuilder` lub `StringBuilderPool`). To **standardowy zamiennik** dla `new StringBuilder()` w hot path.

**Idiom użycia**:
```csharp
Span<char> initialBuffer = stackalloc char[256];
var sb = new ValueStringBuilder(initialBuffer);
sb.Append(...);
return sb.ToString();  // alloc tylko gdy <= 256 nie wystarczy
```

## 27. `AdaptiveCapacityDictionary` — array dla małego N, Dictionary po promocji

Źródło: `Microsoft.AspNetCore.Internal.AdaptiveCapacityDictionary`

```csharp
private const int DefaultArrayThreshold = 10;
internal KeyValuePair<TKey, TValue>[]? _arrayStorage;
private int _count;
// gdy _count > 10 → promocja do Dictionary<TKey, TValue>
```

Linear scan po `_arrayStorage` dla małej liczby kluczy bije Dictionary (hashCode overhead + indirection). Dopiero powyżej 10 elementów przechodzi na hash.

**Użytkowane głównie w**: HTTP routing (zwykle 3-5 route values), HTTP headers (zwykle 10-15), MVC binders. Wszędzie tam gdzie `N` typowo jest małe ale czasem bywa duże.

## 28. `RoslynParallel` — własny wrapper na `Parallel.ForEach`

273 trafień `Parallel.*` / 122 plików, ale w VS-specific kod używa **własnego** `RoslynParallel` (4-6 hits per copy w `Microsoft.CodeAnalysis`, `Workspaces`, `MSBuild.BuildHost`).

Co wnoszą custom wrappery (vs stock `Parallel.ForEach`):
- Cancellation propagation z `OperationCanceledException` → `TaskCanceledException`.
- Aggregate exception flattening.
- `JoinableTaskFactory` integration (nie blokuje UI thread).
- `IAsyncEnumerable` wsparcie (stock `Parallel` dostał `ForEachAsync` dopiero w .NET 6).

**ML.NET** ma jeszcze własny `ParallelUtilities`, `KISSParallel`, `ParallelAsync` (Azure Cosmos), `EnumerableParallelizationExtensions`.

## 29. `EventSource` (ETW) — własny per komponent

`HostingEventSource`, `KestrelEventSource`, `MessagePackEventSource`, `HttpConnectionsEventSource` — każdy duży Microsoft komponent ma własny EventSource. Niski koszt (gdy nie ma listenera, `WriteEvent` to no-op po IsEnabled check), strukturalne eventy przez ETW/EventPipe.

```csharp
[EventSource(Name = "Microsoft-AspNetCore-Hosting")]
internal sealed class HostingEventSource : EventSource {
    public static readonly HostingEventSource Log = new HostingEventSource();
    [Event(1, Level = EventLevel.Informational)]
    public void HostStart() { if (IsEnabled()) WriteEvent(1); }
}
```

**Lekcja**: dla bibliotek `EventSource` > `ILogger.LogTrace` — jest zero-cost gdy nikt nie słucha, a `ILogger` zawsze allokuje argv array.

## 30. `UnmanagedBufferAllocator` + `MemoryPoolBlock` (Kestrel)

Kestrel idzie *poniżej* GC — używa **unmanaged native memory** dla HTTP buffers:
- `UnmanagedBufferAllocator` — alokuje przez `NativeMemory.Alloc`.
- `MemoryPoolBlock` — reusable bloki native pamięci.

**Why**: GC nie skanuje, nie kompaktuje, brak pinning. Dla serwera obsługującego miliony requestów to różnica między pojawiającymi się pauzami a płaską latencją.

**Dla 99% aplikacji**: `ArrayPool<byte>.Shared` wystarczy. To technika dla serwerów high-throughput.

## 31. `MessagePack.UnsafeMemory32` / `UnsafeMemory64` — wybór per architecture

Dwie kopie tego samego kodu — jedna używa 32-bit pointer arithmetic, druga 64-bit:
```csharp
if (IntPtr.Size == 8) UnsafeMemory64.WriteRaw1(...); 
else UnsafeMemory32.WriteRaw1(...);
```

Każda używa `Unsafe.WriteUnaligned` + `Unsafe.As<byte, X>`. To 31 trafień w jednym pliku. **Brutalny** unsafe code dla maksymalnej szybkości serializacji.

## 32. `[FieldOffset]` / `LayoutKind.Explicit` — union view

Masowe użycie do "union" w stylu C:
- `MessagePack.GuidBits` — Guid jako 16 bajtów dla quick encode.
- `Microsoft.Azure.Cosmos.HybridRow.Float128`, `UnixDateTime`, `HybridRowHeader`, `MongoDbObjectId` — packed binary formats.
- `Microsoft.AspNetCore.HttpSys.Internal.HttpApiTypes` — interop z natywnym http.sys.
- `VSPerfReader._LARGE_INTEGER`, `_EVENTBLOB`, `_SAMPLEEVENTBLOB` — VS profiler binary format.

```csharp
[StructLayout(LayoutKind.Explicit)]
internal struct GuidBits {
    [FieldOffset(0)] public Guid Value;
    [FieldOffset(0)] public ulong Low;
    [FieldOffset(8)] public ulong High;
}
```

Zero kopii, atomicowy odczyt dwóch ulongów zamiast 16 bajtów.

## 33. `ConditionalWeakTable<TKey, TValue>` — przyczepianie danych bez retencji

Pozwala dorzucić "extra" dane do obiektu jakby były polem, ale **nie trzyma obiektu przy życiu**. Microsoft używa do:
- `Microsoft.Cci.TrivialHashtableUsingWeakReferences`, `WeakValuesEnumerator` — FxCop analyzer state.
- `System.Management.Automation.WeakReferenceDictionary` — PowerShell engine.
- VS Debugger `WeakEventDelegate` — unsubscribable event handlers.

**Use case**: cache czy state attached do user-supplied object gdzie nie kontrolujesz lifetime.

## 34. `StackObjectPool` (ASP.NET Components) — thread-local pool

Stack-based pool LIFO zamiast bag/queue. Idea: ostatnio zwrócony obiekt prawdopodobnie jest "ciepły" w cache CPU.

```csharp
[ThreadStatic] private static Stack<T>? t_pool;
public T Rent() => (t_pool?.Count > 0) ? t_pool.Pop() : new T();
public void Return(T item) { (t_pool ??= new Stack<T>()).Push(item); }
```

Używane w `RenderTreeBuilder` dla bazillion-times-per-render elementów. Brak kontencji (thread-local), cache-friendly.

## 35. `PerformanceSensitiveAttribute` — analyzer hint

Roslyn ma własny attribute `[PerformanceSensitive(...)]` którym oznacza metody gdzie alokacja byłaby zła. Analyzer ostrzega.

```csharp
[PerformanceSensitive("https://...", AllowCaptures = false)]
private void HotPath() { ... }
```

**Lekcja**: rozważ własny analyzer dla hot path methods w twojej bazie — Roslyn pokazuje że to się skaluje.

---

## TL;DR część 3 — co nowego

| # | Wzorzec | Adopcja |
|---|---------|---|
| 22 | `[SkipLocalsInit]` assembly-wide | Każdy nowy Microsoft projekt |
| 23 | `[InlineArray(N)]` (.NET 8) | Fixed-size hashe, packets, UUID-like |
| 24 | `[ModuleInitializer]` | DI rejestracja, native preload na load assembly |
| 25 | SIMD `Vector128`/`Avx2`/`Ssse3` | Tylko BCL + Kestrel string ops + ML tensors |
| 26 | `ValueStringBuilder` (`ref struct`) | Zamiast `new StringBuilder()` w hot paths |
| 27 | `AdaptiveCapacityDictionary` | Routing, headers — małe N typowo |
| 28 | Custom Parallel wrappers | Cancellation + JoinableTask integration |
| 29 | Per-komponent `EventSource` | Zamiast `ILogger.LogTrace` w bibliotekach |
| 30 | `UnmanagedBufferAllocator` | High-throughput servers (Kestrel) |
| 31 | 32-bit / 64-bit kod-fork | Brutalna szybkość unsafe serializacji |
| 32 | `[FieldOffset]` union | Binary protocols, interop, packed data |
| 33 | `ConditionalWeakTable` | Attach state do user objects |
| 34 | `[ThreadStatic] Stack<T>` pool | Cache-warm, no contention |
| 35 | `[PerformanceSensitive]` analyzer | Egzekwowanie no-alloc kontraktu |

---

## Meta-obserwacje z 1000 DLL

1. **VS-specific kod (`Microsoft.VisualStudio.*`) używa głównie wysokopoziomowych wzorców** — pooling, `JoinableTaskFactory`, `Lazy<T>`, immutable. Mało SIMD, mało unsafe.

2. **ASP.NET Core / Kestrel używa kompletnie innego poziomu agresji** — SIMD, unmanaged memory, własne PipelineSchedulers. Inny constraint (serwer 100K req/s vs IDE).

3. **System.Private.CoreLib (BCL)** to laboratorium SIMD i `[Intrinsic]`. Patrzysz tam jak się robi `IndexOf` na 1 cykl per 16 bajtów.

4. **Roslyn jest najbogatszym źródłem ogólnych wzorców** (`ObjectPool`, `ArrayBuilder`, `SegmentedArray`, `StringTable`, `ConcurrentLruCache`). Można skopiować praktycznie jak template.

5. **MessagePack i StreamJsonRpc** to wzorce **IPC perf** — `IValueTaskSource`, `Pipe`, multiplexing.

6. **Każdy serious komponent ma własny `EventSource`** — ETW jest tańsze niż logging w bibliotekach.

7. **`SkipLocalsInit` to ukryty trick** — 1068 trafień, nikt o nim głośno nie mówi, ale Microsoft go używa wszędzie.

8. **Brak `Channel<T>` nawet w 1000 DLL** w VS-specific code dla streamingu. To naprawdę pipeline-first stack.

---

*Część 3 dopisana 2026-05-15. Łącznie 1024 DLL zdekompilowane do `C:\Temp\vsdecomp\` (~3.5 GB, 252 039 plików `.cs`).*

---

# Część 4 — domknięcie do 2064 DLL (100% managed)

Dodatkowe 1040 DLL: pełny WPF (`PresentationCore`, `PresentationFramework`, `WindowsBase`), `Microsoft.Extensions.DependencyInjection`, EntityFramework, F# Compiler Service, ML.NET Data, Xamarin, Azure SDK, BuildXL. Dorzucają **nową** klasę wzorców z innych światów niż VS-internal/Roslyn/ASP.NET.

## 36. WPF DependencyProperty — sparse storage struct (`EffectiveValueEntry`)

Źródło: `WindowsBase\System.Windows\EffectiveValueEntry.cs`, `DependencyObject.cs:69` hits

WPF musi obsłużyć tysiące właściwości na elementach UI, ale typowy element ustawia ~5-10. Zamiast tablicy `object[N]` dla wszystkich możliwych właściwości — **rzadka tablica struktur 12-bajtowych**:

```csharp
internal struct EffectiveValueEntry {
    private object _value;
    private short _propertyIndex;        // 2 B — index, nie pełna referencja na DependencyProperty
    private FullValueSource _source;     // 4 B bitfield (Local|Style|Inherited|Animated...)
}
```

- `short` (2 bajty) zamiast pełnej referencji na `DependencyProperty` (8B) — zakłada <32k DP per type.
- `FullValueSource` bitfield trzyma 8 priorytetów (Local, Style, Trigger, Inherited...) w jednym intie.
- Tablica jest sortowana po `_propertyIndex` → binary search.
- `DTypeMap` (PresentationCore) — fast type→handler dispatch.

**Pattern do skopiowania**: gdy masz "obiekt z opcjonalnie setkami atrybutów ale typowo zestaw kilku" — sparse `Entry[]` sorted po int index bije `Dictionary<Key, Value>` na małym N.

## 37. WPF Freezable — opt-in immutability

Brushes, Geometries, Animations w WPF dziedziczą po `Freezable`. Domyślnie **mutable** (developer może zmienić kolor pędzla), ale po wywołaniu `.Freeze()`:
- Staje się **thread-safe** (nie potrzebuje sync).
- Wszystkie change-notification subskrypcje są odrzucane.
- Można współdzielić między wątkami / oknami.

```csharp
var brush = new SolidColorBrush(Colors.Red);
brush.Freeze();  // teraz immutable, shareable, no notifications
```

**Korzyść perf**: jeden `Brushes.Red` (frozen) reused thousands of times. Bez Freezable każdy element UI musiałby mieć własną kopię lub subskrybować change events.

`AbstractFreezable` widoczne też w `ICSharpCode.Decompiler` (build → freeze → use forever).

## 38. `RBTree<T>` + `LiveShapingList` (WPF Data) — incremental sort

`MS.Internal.Data.RBTree` — Red-Black tree dla kolekcji widoków. `LiveShapingList` / `LiveShapingTree` aktualizuje filter/sort/group **przyrostowo** gdy źródło się zmienia, zamiast od zera.

**Pattern**: dla "always-up-to-date sorted view of changing collection", tree z O(log n) insert > resort O(n log n).

## 39. `Expression.Compile` — runtime IL emit cache

Dziesiątki bibliotek (CsvHelper, EntityFramework, MessagePack, F#, MVC ModelBinder, MAPI parserów, Azure SDK) **kompilują wyrażenia raz**, cache'ują delegate forever:

```csharp
private static readonly ConcurrentDictionary<Type, Func<object>> s_factories = new();
Func<object> factory = s_factories.GetOrAdd(t, type => {
    var ctor = type.GetConstructor(Type.EmptyTypes);
    return Expression.Lambda<Func<object>>(Expression.New(ctor)).Compile();
});
```

**Reflection wywoływany raz na typ, potem JIT'd delegate.** 10-100x szybsze niż `Activator.CreateInstance(t)` na hot path.

## 40. DI `ActivatorUtilities.CreateFactory` — pre-compile constructor

`Microsoft.Extensions.DependencyInjection.ActivatorUtilities`:
- `CreateFactory(Type, Type[])` — emituje IL z wybranym konstruktorem i lookupami z `IServiceProvider`.
- Zwraca `ObjectFactory` delegate cached.

Zamiast reflection per request: jedna kompilacja na typ na cały lifetime aplikacji.

`Microsoft.Extensions.DependencyInjection.ServiceLookup` (CallSite engine) buduje graf dependency w postaci wyrażeń, compile do delegate, ponownie używa.

## 41. Runtime IL emit przez `DynamicMethod` / `ILGenerator`

Tam gdzie `Expression.Compile` nie wystarcza (struktury, ref params), Microsoft schodzi do gołego IL:

| Konsument | Co emituje |
|---|---|
| `MessagePack.DynamicObjectTypeBuilder` (96 hits) | Per-type serializer/deserializer (full IL) |
| `MessagePack.ILGeneratorExtensions` (81 hits) | DSL na ILGenerator dla typowych operacji |
| `EntityFramework.EntityProxyFactory` (9 hits) | Proxy classes dla lazy-loading entities |
| `EntityFramework.IPocoImplementor` (113 hits) | Property getter/setter override |
| `F# Compiler.ILDynamicAssemblyWriter` (**262 hits**) | F# kompiluje całe assembly do pamięci |
| `dotnet-svcutil.CodeGenerator` (72 hits) | WCF service contract emission |
| `SignalR.TypedClientBuilder` (23 hits) | Strongly-typed hub proxies |
| `Roslyn.MessagePack` (in workspaces) | Code analysis serialization |

**Wniosek**: Runtime IL emit nie jest niszą — używają go największe biblioteki .NET. Pattern: **codegen at first-use, cache result, run forever**.

## 42. Segmented sort — `SegmentedArraySortHelper`

Skoro Roslyn ma `SegmentedArray<T>` (jagged → omija LOH), to `Array.Sort` nie zadziała. Microsoft napisał własny `SegmentedGenericArraySortHelper` (port klasyka introsort/heapsort z BCL `ArraySortHelper`) działający na `T[][]`.

Ten plik widziany w 7 różnych DLLs (`Microsoft.CodeAnalysis.Workspaces`, `MSBuild.BuildHost`, `Microsoft.VisualStudio.CoreUtility`, `Microsoft.Build.Framework`, `InteractiveHost`) — kod skopiowany identycznie. **Tym jak Microsoft dzieli kod**: copy-paste między assembly zamiast tworzyć dependency.

## 43. F# pamięta o **persistent collections** w warstwie sourcu

`FSharp.Core` używa swoich własnych immutable map/set zaimplementowanych jako persistent AVL trees (w `FSharp.Collections`). Każda "zmiana" zwraca nową strukturę dzielącą >90% pamięci ze starą.

To **inny model** niż System.Collections.Immutable (które używa B-tree dla `ImmutableSortedDictionary` ale AVL dla `ImmutableSortedSet`). Microsoft utrzymuje OBIE biblioteki bo F# ma odrębne ograniczenia (functional purity, structural equality).

## 44. WPF `InsertionSortMap` — sortowanie małych zbiorów

`WindowsBase\MS.Utility\InsertionSortMap.cs` — insertion sort jako dedykowany helper dla map/list o N < ~16. WPF zna swój domain: większość kolekcji jest mała → insertion sort O(n²) ale **stała mała**, bez complications quicksort'a.

**Pattern**: dla "wiem że N zwykle <16" insertion sort bije quicksort/Array.Sort overhead.

## 45. `[CallerArgumentExpression]` polyfill + assert helpers

Widoczne we wszystkich nowszych projektach Microsoftu. `ArgumentNullException.ThrowIfNull(x, nameof(x))` używa `[CallerArgumentExpression]` żeby kompilator dostarczył napis automatycznie.

`Microsoft.VisualStudio.Validation.Requires` + `Verify` używają tego patternu od dawna — teraz BCL go zaadoptował.

---

## TL;DR część 4

| # | Wzorzec | Skąd |
|---|---------|---|
| 36 | Sparse `Entry[]` struct dla obiektów z setkami opcjonalnych pól | WPF `DependencyObject` |
| 37 | Mutable → `.Freeze()` → immutable shareable | WPF Brushes/Animations |
| 38 | RB-tree dla "live sorted view" zmieniającej się kolekcji | WPF `LiveShapingList` |
| 39 | `Expression.Compile` + cache per typ | DI, EF, MessagePack, ASP.NET MVC |
| 40 | `ActivatorUtilities.CreateFactory` | Microsoft.Extensions.DI |
| 41 | `DynamicMethod` + `ILGenerator` z cache | MessagePack, EF proxy, F# compiler |
| 42 | Własny sort na segmented arrays | Roslyn (kopiowany 7×) |
| 43 | Persistent AVL trees zamiast immutable B-tree | F#.Core |
| 44 | Insertion sort dla N<16 | WPF |
| 45 | `[CallerArgumentExpression]` w guardach | Wszędzie w nowym kodzie |

---

## Końcowe meta-obserwacje (po 2064 DLL)

1. **Trzy "szkoły" perf w Microsoft**:
   - **VS-internal** (Roslyn, VS Threading) — pooling, immutable, async-first, brak SIMD/unsafe
   - **High-throughput server** (Kestrel, ASP.NET Core) — SIMD, unmanaged memory, custom awaiters
   - **BCL** (`System.Private.CoreLib`) — laboratorium `[Intrinsic]`, AVX/SSE, RyuJIT-aware code

2. **WPF jest osobnym światem** — opracowano go gdy `Span<T>`/`ArrayPool` nie istniały. Patterns (sparse storage, freezable, RBTree) są wciąż doskonałym wzorcem dla **innych** problemów niż UI.

3. **Codegen jest standardem** — `Expression.Compile`/`DynamicMethod` wszędzie gdzie liczy się szybkość reflection. Roslyn analyzers wręcz wymuszają tę technikę (`[PerformanceSensitive]`).

4. **Microsoft prefers copy-paste over shared dependency** dla niskopoziomowych utilities (`SegmentedArraySortHelper`, `SkipLocalsInitAttribute`, `ValueStringBuilder`). Większy binary, ale brak version conflicts.

5. **F# i C# stacks rozdzielne** — F# nie używa Roslyn ani BCL collections, ma równoległą implementację compilera + persistent collections.

6. **Pooling > SIMD pod względem ROI** w aplikacjach IDE/biznesowych. SIMD opłaca się gdy parsujesz miliony bajtów/s.

---

*Część 4 dopisana 2026-05-15. **Łącznie 2064 managed DLL zdekompilowanych** w `C:\Temp\vsdecomp\` (~5 GB, 370 860 plików `.cs`). 100% pokrycia managed Microsoft+System+3rd-party assembly z VS 2022 install (>=32KB, niefiltrowanych jako native runtime stubs).*

---

# Część 5 — cross-reference z dotnet-optimization-cheatsheet (Nikou Usalp)

Źródło: <https://github.com/nikouu/dotnet-optimization-cheatsheet> — 40 technik perf zebranych z blogów (Stephen Toub, NDepend, Adam Sitnik et al.).

## Co cheatsheet potwierdza z mojego scanu (cross-validation)

Praktycznie 1:1 z moim wynikiem — `ArrayPool`, `ObjectPool`, `ValueTask`, `Span<T>`, `stackalloc`, `[MethodImpl(AggressiveInlining)]`, `[SkipLocalsInit]`, `[InlineArray]`, `MemoryMarshal`, `Unsafe.*`, custom struct awaiters, SIMD. To, co Microsoft pisze w blogach, **rzeczywiście robi w kodzie produkcyjnym**. Niewielu firm tak robi.

## 46. Co dodaje cheatsheet (nowe vs mój scan)

### A. Techniki których brakowało mi w analizie

**`[SuppressGCTransition]` na P/Invoke** (#36 cheatsheet) — atrybut na `[DllImport]` który pomija przełączenie wątku w "preemptive GC mode" na czas natywnego wywołania. Oszczędza ~10 ns per call. Tylko dla **bardzo szybkich** native calls (jak `GetTickCount`) — dla długich blokujących to się wywali (GC stuck).

**`GC.TryStartNoGCRegion(size)` + `GC.EndNoGCRegion()`** (#37) — wyłącza GC w sekcji krytycznej. Używane w real-time / low-latency (HFT, audio processing). W moim scanie tego nie szukałem — sprawdziłem teraz, faktycznie się pojawia w paru komponentach (zaraz dopiszę gdzie).

**`CollectionsMarshal.AsSpan(List<T>)`** (#15) — zwraca `Span<T>` na wewnętrzną tablicę listy, **bez kopiowania**. Iteracja indeksowa po spanie bije `foreach` po `List<T>`. Trzeba uważać: nie wolno modyfikować listy w trakcie iteracji.

**`string.Create<TState>(int length, TState state, SpanAction<char, TState>)`** (#5) — buduje string znanej długości jednym `Span`-write, bez `StringBuilder` ani intermediate buffers. Idealny do formatowania (UUID, hex, czasów).

**`MemoryMarshal.GetArrayDataReference()` + `Unsafe.IsAddressLessThan()`** (#31) — "fastest loops" — iteracja przez pointer arithmetic bez bounds check. Używane w BCL (`Ascii.cs`, `Utf8Utility.cs` — mój scan widział te pliki ale nie nazwałem techniki).

**`[UnscopedRef]`** (#28) — pozwala ref-field w struct "uciec" poza scope metody, gdy normalnie kompilator by zablokował. Wymagane dla niektórych zaawansowanych ref-struct patternów (C# 11+).

**Default `GetHashCode`/`Equals` dla `struct` używa reflection** (#29) — jeden z najmniej znanych łapaczy. Jeśli używasz struktury jako klucza `Dictionary`, **musisz** nadpisać `Equals`/`GetHashCode` ręcznie albo dostaniesz reflection-based wolny path. Dla `record struct` kompilator generuje sam.

**Static lambdas** (#32) — `Dict.GetOrAdd(key, static k => Create(k))` zamiast `Dict.GetOrAdd(key, k => Create(k))`. `static` keyword wymusza brak closure capture → zero alokacji delegate'a. W moim scanie używane jest implicitnie ale warte wymienienia.

### B. Techniki spoza skanu (config / build-level, nie widać w IL)

- **Server GC** — `<ServerGarbageCollection>true</ServerGarbageCollection>` w csproj. Multi-thread mark/sweep, lepsze dla serwerów (gorsze dla desktopa). VS odpala w trybie Workstation GC.
- **Native AOT** — `<PublishAot>true</PublishAot>`. Kompilacja natywna, brak JIT, brak reflection runtime. Bardzo szybki start, mały binary. Trade-off: dynamic code (Expression.Compile, EF) odpada.

### C. Domain-specific tricks z cheatsheeta

- **`HttpClient.GetStreamAsync` / `GetFromJsonAsync<T>`** zamiast `GetAsync().ReadAsStringAsync()` — pomija intermediate buffer.
- **`EF.CompileQuery(...)`** — pre-kompiluje LINQ query do delegate'a. Microsoft Extensions DI używa tego samego patternu (część 4 #40).
- **`IsSuccessStatusCode` zamiast try/catch na HttpResponse** — exception throw kosztuje 10-100 μs.
- **Nie używaj `int?`/`bool?` w hot path** — unboxing + null check per access.
- **`String.Compare(a, b, StringComparison.OrdinalIgnoreCase)`** zamiast `a.ToLower() == b.ToLower()` — żadnych alokacji.
- **`RecyclableMemoryStream`** (`Microsoft.IO.RecyclableMemoryStream` NuGet) — pooled `MemoryStream` z resizing przez chunki. Zamiennik `new MemoryStream()` w hot path serializacji.

### D. `ReadOnlySequence<T>.Slice` jest wolne (#40)

Niespodzianka cheatsheeta: `ReadOnlySequence<T>.Slice` ma narzut. Lepiej wziąć `.First.Span` jeśli wystarczy pierwszy segment. Wpływa na każdy kod parserujący Pipelines.

## Co MOJE scan dodaje czego nie ma w cheatsheecie

To ważne — cheatsheet zbiera techniki "ogólne", ale **nie pokazuje architektur**. Mój scan pokazał wzorce **konstrukcyjne**:

- **`SegmentedArray<T>` / `SegmentedDictionary` / `SegmentedHashSet`** (Roslyn, kopiowane 7×) — unikanie LOH przez jagged arrays.
- **Two-tier hash table** (Roslyn `StringTable` — local + shared).
- **Sparse struct storage** (WPF `EffectiveValueEntry`).
- **Frame-multiplexing** (Nerdbank `MultiplexingStream`).
- **RPC correlation-ID cancel propagation** (StreamJsonRpc).
- **Per-shape frozen dictionaries** (BCL `OrdinalStringFrozenDictionary_LeftJustifiedSingleChar` etc.).
- **`Freezable` opt-in immutability** (WPF).
- **Custom LRU z `lock`** (nie `ConcurrentDictionary`, bo eviction wymaga atomic op na 2 strukturach).
- **`ValueStringBuilder` polyfill w setkach projektów**.
- **`UnsafeMemory32` / `UnsafeMemory64` dual code path** (MessagePack).
- **`AdaptiveCapacityDictionary`** (array dla N≤10, promocja).
- **Hot/Slow split metod** (`Allocate` / `AllocateSlow`).
- **`[ThreadStatic]` `Stack<T>` pool** (ASP.NET RenderTree).
- **`PerformanceSensitive` analyzer attribute** w Roslyn.
- **Per-komponent `EventSource`** zamiast `ILogger`.
- **F# `ILDynamicAssemblyWriter`** (262 emit calls) — równoległy compiler stack do Roslyn.

Cheatsheet to **lista narzędzi**. Mój scan to **lista architektur**. Razem to dopiero całość.

---

## TL;DR część 5 — co dopisać do toolboxa

| # | Technika | Co dodaje |
|---|----------|---|
| 46a | `[SuppressGCTransition]` na P/Invoke | ~10 ns/call dla quick native calls |
| 46b | `GC.TryStartNoGCRegion` | No-GC critical sections (real-time) |
| 46c | `CollectionsMarshal.AsSpan(list)` | Span view bez kopiowania List<T> |
| 46d | `string.Create<TState>` | Known-length string bez intermediate |
| 46e | `MemoryMarshal.GetArrayDataReference` + `Unsafe.IsAddressLessThan` | Fastest loops (pointer arith, no bounds check) |
| 46f | `[UnscopedRef]` | Ref fields w struct mogą "uciec" |
| 46g | Override `GetHashCode`/`Equals` dla struct keys | Bo default = reflection |
| 46h | `static` lambdas | Zero closure allocation |
| 46i | Server GC config | Wielowątkowe app, nie desktop |
| 46j | Native AOT | Startup + binary size (kosztem reflection) |
| 46k | `HttpClient.GetStreamAsync` / `GetFromJsonAsync<T>` | Bez intermediate string |
| 46l | `EF.CompileQuery` | Pre-compiled LINQ |
| 46m | `IsSuccessStatusCode` vs throw | Wyjątek = 10-100 μs |
| 46n | Unikaj `int?`/`bool?` w hot path | Boxing + null check overhead |
| 46o | `String.Compare(..., StringComparison.X)` | Bez ToLower() alloc |
| 46p | `RecyclableMemoryStream` | Pooled MemoryStream |
| 46q | `ReadOnlySequence<T>.Slice` jest wolne | Użyj `.First.Span` |

---

*Część 5 dopisana 2026-05-15. Łącznie **63 wzorce** (45 z mojego scanu + 17 nowych z cheatsheetu + 1 cross-reference). Plik staje się referencyjną kartą perf .NET.*

---

# Część 6 — cross-reference z awesome-dot-net-performance (Adam Sitnik)

Źródło: <https://github.com/adamsitnik/awesome-dot-net-performance> — kuratowana lista tooli, talków, książek od Adama Sitnika (.NET runtime team, autor `Span<T>`/`ArrayPool` perf improvements, BenchmarkDotNet contributor).

To **inna oś** niż moja analiza i cheatsheet Nikou: nie wzorce kodu, lecz **meta-wiedza** — narzędzia, ludzie, koncepcje JIT których w decompiled IL **nie zobaczysz**.

## 47. JIT-level mechanizmy (niewidoczne w IL, kluczowe dla perf)

To są techniki które dzieją się **podczas wykonania kodu**, nie w samym kodzie:

### Tiered Compilation (.NET 3.0+, on-by-default)

JIT kompiluje metodę dwa razy:
- **Tier 0** (na starcie) — szybka kompilacja, prosty kod, bez optymalizacji. Cel: jak najszybsze załadowanie.
- **Tier 1** — gdy metoda staje się "hot" (kilka wywołań), JIT re-kompiluje z pełnymi optymalizacjami (inlining, unrolling, etc.).

**Konsekwencja**: pierwsze wywołania zawsze wolniejsze. Benchmark musi zrobić warmup, inaczej mierzy tier-0.

### Dynamic PGO (.NET 8+, default-on od .NET 9)

W tier 0 JIT **instrumentuje kod** — zlicza która gałąź `if` częściej trafia, jakie typy faktycznie pojawiają się w `virtual call`. Tier 1 używa tych statystyk:
- Częsta gałąź → fast path inline'owany.
- Dominujący typ → **devirtualization + guarded inlining** (zamiast `callvirt` jest `if (obj.Type == typeof(X)) X.Method() else callvirt`).

**Impact**: 5-15% perf w realnych aplikacjach **bez zmiany kodu**. Po prostu update runtime.

### On-Stack Replacement (OSR, .NET 7+)

Problem: długie pętle uruchamiane raz w tier 0 nigdy nie zostaną "promoted" — pętla się nie kończy, więc tier-1 wersji nikt nie uruchomi. **OSR podmienia kod metody W TRAKCIE jej wykonywania** — stack frame przepisywany w locie z tier-0 do tier-1.

**Bez OSR**: `for (int i = 0; i < 10_000_000; i++)` zawsze w tier-0.
**Z OSR**: po kilku sekundach metoda jest "promoted" mid-loop.

### Escape Analysis (.NET 9+, partial)

JIT analizuje czy `new MyClass()` faktycznie "ucieka" poza metodę. Jeśli nie → **alokacja na stack zamiast heap**. Dla małych krótko-żyjących obiektów to GC-free.

Wciąż w fazie rozwoju — działa głównie dla `Span<T>`, value-typed result enumerable'ów.

### ReadyToRun (R2R)

Kompilacja AOT-podobna na poziomie assembly. Generuje natywny kod **dodatkowo** do IL, JIT może użyć od razu zamiast kompilować. BCL od .NET 6 jest R2R'd → szybszy startup.

**Wniosek**: gdy mierzysz perf .NET 8+ vs .NET 6, 10-30% różnicy bierze się **z JIT'a, nie z twojego kodu**. Update runtime przed optymalizacją.

## 48. Pułapki które trzeba znać (nie pojawiają się w blogach perf "tips")

### `Finalize` ~~Dispose~~ to GC podatek

Każdy obiekt z finalizatorem (`~MyClass()`):
- Allokuje się wolniej (rejestracja w finalization queue).
- Żyje **co najmniej 2 generacje GC** (Gen 0 → finalization queue → finalizer thread → Gen 1+).
- Blokuje cleanup do czasu wykonania finalizera.

**Pattern**: implementuj `IDisposable` zamiast finalizera. Finalizer tylko jako safety net dla unmanaged resources, z `GC.SuppressFinalize(this)` w Dispose.

### `string.Intern` ma lock contention

Globalna tablica intern'd strings używa lock'a. Pod load = bottleneck. Roslyn rezygnuje z `string.Intern` na rzecz własnego `StringTable` (mój scan #4) **właśnie z tego powodu**.

### `Dictionary<K,V>` read-only **nie jest** thread-safe

Wbrew intuicji: nawet "tylko czytasz", `Dictionary.TryGetValue` może race-condition'ić jeśli inny wątek nawet **resize'uje** structure. Pod load = corruption.

**Pattern**: użyj `FrozenDictionary` (mój scan #6), `ImmutableDictionary`, albo `ConcurrentDictionary`. Czytanie zwykłego `Dictionary` przez wiele wątków = UB.

### `ReaderWriterLock` vs `ReaderWriterLockSlim`

`ReaderWriterLock` to **stary** typ (.NET 1.1), kernel-mode locks, slow. **Nigdy nie używaj** — `ReaderWriterLockSlim` jest user-mode, ~10× szybszy. Niestety obie nazwy są na IntelliSense.

## 49. Tooling — co realnie używać

| Narzędzie | Kiedy |
|---|---|
| **BenchmarkDotNet** | Każdy micro-benchmark. Statistical analysis, warmup, GC pressure metrics. Industry standard. |
| **PerfView** | Snapshot CPU + heap + ETW dla Windows. Steep learning curve, ale niczego lepszego nie ma do GC analysis. |
| **dotnet-trace** | Cross-platform (Linux/macOS) alternative do PerfView via EventPipe. |
| **dotMemory** (JetBrains) | Heap snapshot porównanie, retention graphs. UI'd. |
| **dotTrace** (JetBrains) | Sampling i tracing profiler. Production-safe sampling. |
| **ultra** (Alexandre Mutel) | Nowy (2023+) sampling profiler dla Windows. Zero-config, ETW pod spodem. |
| **Visual Studio Profiler** | Zintegrowany. Działa, ale słabszy niż dotTrace/PerfView. |
| **ClrMD** | Library, NIE narzędzie. Pozwala napisać własny debugger / heap analyzer. |
| **Clr Heap Allocation Analyzer** | Roslyn analyzer — ostrzega o ukrytych alokacjach (`foreach` na `IEnumerable`, boxing, closures) **w trakcie pisania kodu**. |

**Rule of thumb**: zacznij od **BenchmarkDotNet** (mikro) + **dotTrace lub PerfView** (makro snapshot). Reszta to specjalizacja.

## 50. Ludzie do śledzenia

| Osoba | Czemu |
|---|---|
| **Stephen Toub** (Microsoft) | Roczne `"Performance Improvements in .NET X"` blog posty — najlepsze podsumowanie zmian runtime. Lektura obowiązkowa. |
| **Maoni Stephens** (Microsoft) | GC internals. Jeśli chcesz zrozumieć WHY twój profil pokazuje pauzy. |
| **Andrey Akinshin** | Autor BenchmarkDotNet. Książka "Pro .NET Benchmarking". |
| **Adam Sitnik** | `Span<T>`, `ArrayPool`, alokacje. |
| **Konrad Kokosa** | "Pro .NET Memory Management" (2024) — najbardziej dogłębna książka o GC. |
| **Tanner Gooding** | Hardware intrinsics, SIMD. |
| **Egor Bogatov** | JIT contributions, ostatnio dynamic PGO. |
| **Matt Warren** | Stary blog (mattwarren.org), wciąż wartościowe deep dives. |

## 51. Książki — wąska lista

- **Pro .NET Memory Management** (Kokosa, 2024) — *jedyna* aktualna książka o GC. ~700 stron, sztuka.
- **Writing High-Performance .NET Code** (Watson, 2018) — choć stara, fundamenty wciąż aktualne. Krótka, czytalna.
- **Pro .NET Benchmarking** (Akinshin, 2019) — metodologia pomiarów, nie tylko BenchmarkDotNet.
- **CLR via C#** (Richter, 2012) — fundamenty CLR. Stare, ale rozdziały o threading/memory wciąż obowiązują.

## 52. EventPipe — cross-platform ETW

Linux/macOS nie mają ETW (Windows-only). `EventPipe` to przepisanie tej koncepcji cross-platform. `dotnet-trace` używa EventPipe pod spodem.

**Konsekwencja**: `EventSource` (mój scan #29) działa wszędzie — na Windows trafia do ETW, na Linux do EventPipe → ta sama strukturalna telemetria.

## 53. ARM64 — drugi pierwszorzędny target

Microsoft mocno inwestuje w ARM64 (.NET 8+):
- ARM64 ma **weaker memory model** niż x64 → więcej memory barriers w generated code. To ma koszt.
- JIT robi **ARM64-specific peephole optimizations** (special instructions: `MADD`, `MSUB`).
- SIMD na ARM64 to `AdvSimd` (`System.Runtime.Intrinsics.Arm`), inna namespace niż x86 (`Sse2`/`Avx2`).

**W moim scanie**: widziałem `AdvSimd.IsSupported` checks w BCL collections, ale `Microsoft.VisualStudio.*` nic nie ma na ARM. VS jest x64-first.

---

## Co Adam Sitnik dodaje czego nie ma ani u mnie ani u Nikou

To trzecia oś analizy:

1. **JIT runtime mechanics** — Tiered, PGO, OSR, Escape Analysis, R2R. **Bardzo wpływowe na perf bez zmiany kodu**.
2. **Pułapki konkretnych BCL API** — `string.Intern`, finalizers, `ReaderWriterLock`, Dictionary thread-safety.
3. **Tooling ranking** — co faktycznie używać, nie tylko co istnieje.
4. **Conferences / talks** — żywa wiedza w nagraniach które nie trafiają do blogów.

---

## TL;DR część 6

| # | Koncepcja | Gdzie się dzieje |
|---|-----------|---|
| 47a | Tiered Compilation | JIT, automatic |
| 47b | Dynamic PGO | JIT, .NET 8+ default |
| 47c | OSR | JIT, mid-execution |
| 47d | Escape Analysis | JIT, .NET 9+ partial |
| 47e | ReadyToRun | Build time + JIT |
| 48a | Finalizer = GC podatek | Twój kod |
| 48b | `string.Intern` lock contention | BCL |
| 48c | `Dictionary` read-only != thread-safe | BCL |
| 48d | `ReaderWriterLock` przestarzałe | BCL |
| 49 | BenchmarkDotNet + dotTrace/PerfView | Tooling |
| 50 | Stephen Toub blog (annual) | Resource |
| 52 | EventPipe = cross-plat ETW | Runtime infra |
| 53 | ARM64 jako równorzędny target | Runtime |

---

## Końcowe podsumowanie — 3 osie analizy

Plik integruje teraz **trzy uzupełniające się perspektywy** na perf .NET:

1. **Mój scan 2064 DLL** — *jak Microsoft realnie pisze kod w produkcji* (architektury, wzorce strukturalne).
2. **Cheatsheet Nikou** — *toolbox technik* do zastosowania w swoim kodzie (Span, ArrayPool, SkipLocalsInit etc.).
3. **Awesome list Adama** — *meta-wiedza* (JIT mechanics, tooling, ludzie, książki).

Bez którejkolwiek z tych trzech nie masz pełnego obrazu:
- Sam kod → nie wiesz **co PGO zrobi z twoim kodem**.
- Sam toolbox → nie wiesz **które techniki realnie Microsoft używa** (a które tylko w blogach).
- Sama meta-wiedza → nie wiesz **jak to wygląda w 5 GB rzeczywistego IL'a**.

---

*Część 6 dopisana 2026-05-15. Łącznie **70+ pojęć** zorganizowanych w 6 części. Kompletna karta referencyjna perf .NET ufundowana na dekompilacji 2064 DLL + dwóch kuratowanych listach.*

---

# Część 7 — cross-reference ze slajdami Konrada Kokosy (Pro .NET Memory Management)

Źródło: <https://prodotnetmemory.com/slides/PerformancePatterns/> — slajdy od Konrada Kokosy (autor *Pro .NET Memory Management* 2024, ~700 stron). To **trzecia oś** — po patternach z kodu (mój scan) i toolboxie technik (cheatsheets) — **principles i memory-layout patterns** z naciskiem na CPU cache.

## 54. Frugal Object Pattern — discriminated union dla "0 / 1 / many"

To wzorzec którego **nie zauważyłem w scanie** mimo że jest pod nosem.

**Problem**: HTTP header value typowo jest jednym stringiem (`"application/json"`), czasem dwoma (`"gzip, deflate"`), prawie nigdy więcej. Naiwne `string[] values` allokuje tablicę na każdy header — niepotrzebnie.

**Rozwiązanie**: `Microsoft.Extensions.Primitives.StringValues` (ASP.NET Core):

```csharp
public readonly struct StringValues {
    private readonly object _values;       // null | string | string[]
    // null → empty
    // string → single value (no array alloc)
    // string[] → multiple values
}
```

Jedno pole, trzy stany. Dla 90% headerów (single-value) zero alokacji tablicy.

**Inne przykłady**:
- `WPF.FrugalList<T>` — to samo dla kolekcji dependency properties (typowo 1-3 elementy).
- `Optional<T>` w wielu API.

**Pattern do skopiowania**: gdy domain mówi "typowo 1, czasem kilka, rzadko dużo" — nie używaj `List<T>` jako default. Discriminated union w jednym polu.

## 55. Struct of Arrays (SOA) — Data-Oriented Design

**Klasyk game-devu, rzadki w business code.**

Zamiast:
```csharp
class Particle { float X, Y, VX, VY, R, G, B, A; }
Particle[] particles;
```

Robisz:
```csharp
float[] xs, ys, vxs, vys, rs, gs, bs, as_;
```

**Why**: pętla "update position" czyta tylko `xs, ys, vxs, vys`. W AoS (Array of Structs) każdy cache line ciągnie cały Particle (~32 B), połowa jest niepotrzebna. W SOA cache line `xs[]` to **16 wartości X** ciasno, prefetcher śledzi liniowy access.

**Konrad pokazuje 10× speedup** w benchmarku. SIMD-able dodatkowo (`Avx2` może dodać 8 floatów na raz).

**Trade-off**: łamie OOP encapsulation. Particle jako konceptualna jednostka rozsypuje się na 8 tablic. Dla **gorących pętli numerycznych** — nieoceniony. Dla CRUD biznesowego — antypattern.

**W moim scanie**: System.Numerics.Tensors używa SOA pod spodem. ML.NET też. Reszta kodu Microsoftu — nie.

## 56. Sześć "Performance Principles" Konrada — framework myślenia

Zamiast "list of tricks" — meta-framework:

| # | Zasada | Praktyczna konsekwencja |
|---|--------|-------------------------|
| **0** | Memory/CPU disrepancy | RAM jest 100× wolniejszy niż CPU. Każdy cache miss = ~100 cykli stojących. |
| **1** | Fit cache line (64B) | Hot struct ≤ 64 bajty. Inaczej fetch dwa cache line per access. |
| **2** | Fit highest cache level | Working set ≤ L1 (~32 KB) → najszybciej. ≤ L2 (~256 KB) → szybko. ≤ L3 (~8 MB) → ok. Powyżej → DRAM, dramat. |
| **3** | Design for sequential access | Linear access = prefetcher zgaduje co dalej, brak miss'ów. Random access (linked list, hash table) = miss każdy element. |
| **4** | Avoid GC overhead | Pooling, struct, stackalloc. To wszystko z części 1-6. |
| **5** | Avoid calls | Virtual call = pipeline stall (branch predict). Interface call = double indirection. Inline hot path lub `sealed`. |
| **6** | Avoid false sharing | Dwa rdzenie piszące różne pola tego samego cache line → cache coherence storm. |

**Ten framework przewija się przez wszystkie patterns Microsoftu**:
- `ObjectPool<T>` (mój #1) = #4.
- `EffectiveValueEntry` sparse struct (#36) = #1, #2 (mniej pamięci → więcej w cache).
- `SegmentedArray` (#5) = #2 (omija LOH, więcej w L3).
- `[ThreadStatic] Stack<T>` (#34) = #6 (nie ma false sharing bo thread-local).

## 57. False sharing — wzorzec o którym mało kto pamięta

Dwa rdzenie modyfikują różne pola w tym samym cache line (64 B). Każdy zapis na rdzeniu A unieważnia copy w rdzeniu B → constant cache miss. Bywa **wolniejsze niż lock** mimo "lock-free".

**Patologiczny przykład**:
```csharp
class Counter { 
    public long ThreadACount;  // offset 0
    public long ThreadBCount;  // offset 8 — TEN SAM cache line!
}
```

**Fix — padding**:
```csharp
[StructLayout(LayoutKind.Explicit, Size = 64)]
struct PaddedCounter { [FieldOffset(0)] public long Value; }
```

W BCL: `System.Threading.PaddedReference<T>`. W moim scanie widziałem padding w `ConcurrentQueueSegment.cs` (BCL) — teraz wiem dlaczego.

## 58. `Span<T>` + `async` to jest niemożliwe (i dlaczego)

`Span<T>` to **`ref struct`** — może żyć tylko na stosie. Nie można go zapakować w pole obiektu (a async methods są kompilowane do state machine **klasy** zapisującej lokalne zmienne).

**Workaround**: `Memory<T>` + `.Span` property tuż przed użyciem:

```csharp
async Task ProcessAsync(Memory<byte> data) {
    await SomethingAsync();
    var span = data.Span;   // dopiero teraz dostaję Span
    DoStuff(span);
}
```

To dlatego `PipeReader.ReadAsync()` zwraca `ValueTask<ReadResult>` z `Buffer` jako `ReadOnlySequence<byte>` (sequence of `Memory<byte>`), nie `ReadOnlySpan<byte>`.

## 59. `stackalloc` ma ukryty koszt — zeroing

`stackalloc byte[256]` **zeruje 256 bajtów** zanim ci je da. Dla małych buforów to nic, dla dużych w hot loop — mierzalne.

**Fix**: `[SkipLocalsInit]` na metodzie (mój #22) — eliminuje zeroing. **Critical**: po tym MUSISZ napisać do buforu przed odczytem (UB inaczej).

**Polyfill dla .NET < 5**: Fody plugin `[LocalsInit(false)]` — IL-rewriting wyłączający init flag w generated assembly.

## 60. Inne mniejsze obserwacje Konrada

- **`StackOverflowException` jest uncatchable** — proces umiera. `stackalloc` w pętli rekurencyjnej = ryzyko.
- **`SlabMemoryPool`** (Kestrel) — pool nie tablic, lecz **slabów** (większych bloków po 4 KB) pociętych na chunki. Mniej book-keeping niż `ArrayPool` dla małych alokacji.
- **`ArrayPool<T>.Shared` ma per-core stacks** — 17 size buckets, max 50 arrays per bucket. Trim wywołuje się sam pod GC pressure.
- **LLC (Last-Level Cache) miss rate** jako metryka benchmarku — nie tylko CPU%/alloc. PerfView i `dotnet-counters` to pokazują.

---

## TL;DR część 7 — to co Konrad dodaje

| # | Wzorzec / Koncepcja | Co dodaje |
|---|---------------------|---|
| 54 | Frugal Object (`StringValues`, `FrugalList`) | Discriminated union dla "0/1/many" — zero alokacji w common case |
| 55 | Struct of Arrays (SOA) | Cache locality + SIMD-friendly dla hot loops |
| 56 | Sześć Performance Principles | Framework myślenia: cache > GC > alloc > calls |
| 57 | False sharing + padding | Lock-free może być wolniejsze niż lock bez paddingu |
| 58 | `Span<T>` + `async` niemożliwe | `ref struct` nie wejdzie do state machine — użyj `Memory<T>` |
| 59 | `stackalloc` zeruje (koszt) | `[SkipLocalsInit]` + Fody polyfill na starsze .NET |
| 60a | `SlabMemoryPool` | Pool slabów 4 KB pociętych na chunki (Kestrel) |
| 60b | LLC miss rate jako metryka | Beyond CPU% — co pokazuje że pattern działa |

---

## Końcowe końcowe podsumowanie — 4 osie

Plik integruje teraz **cztery uzupełniające się perspektywy**:

1. **Mój scan 2064 DLL** — *konkretne architektury produkcyjne*. Co Microsoft robi (sparse storage, frame multiplexing, RPC cancel).
2. **Cheatsheet Nikou** — *toolbox technik kodowych*. Czego użyć w swoim kodzie (Span, SkipLocalsInit).
3. **Awesome list Adama** — *meta-wiedza i tooling*. Co dzieje się w runtime (PGO, OSR), czym mierzyć.
4. **Slajdy Konrada** — *zasady i memory layout*. Dlaczego warto (cache lines, false sharing, sequential access).

Każda oś bez pozostałych jest niekompletna:
- Bez **#1** → nie wiesz co Microsoft faktycznie robi (vs co pisze w blogach).
- Bez **#2** → nie masz konkretnego toolboxa.
- Bez **#3** → nie wiesz **czym mierzyć** ani **jak runtime zachowuje się pod spodem**.
- Bez **#4** → nie wiesz **dlaczego** — patterns stają się cargo-cult.

---

*Część 7 dopisana 2026-05-15. Łącznie **~75 pojęć** w 7 częściach. Plik osiągnął stan referencyjnej karty perf .NET — od JIT mechaniki przez tooling do konkretnych architektur produkcyjnych.*

---

# Część 8 — dwa wzorce z ML.NET (`dotnet/machinelearning`)

Źródło: przegląd kodu `Microsoft.ML.*` (`src/`) — biblioteka numeryczna przetwarzająca miliony wierszy/wektorów. To **inny constraint** niż VS-internal, ASP.NET czy BCL: nie IDE, nie serwer 100K req/s, lecz **high-throughput numeric pipeline**. Dwa wzorce konstrukcyjne, których nie pokrywały części 1-7.

## 61. Mutable editor nad immutable buffer — `immutable struct` + `ref struct` editor + `Commit()`

Źródło: `Microsoft.ML.DataView/VBuffer.cs`, `VBufferEditor.cs`

`ValueStringBuilder` (#26) reużywa wewnętrzny bufor, ale jest *sam w sobie* mutable `ref struct`. ML.NET idzie krok dalej: **publiczny typ jest immutable `readonly struct`**, a mutacja idzie przez **osobny `readonly ref struct` editor**, który przejmuje tablice na własność, edytuje przez `Span`, i składa z powrotem przez `Commit()`.

```csharp
// 1. Publiczny typ — readonly struct, immutable, trzyma reużywalne tablice
public readonly struct VBuffer<T> {
    private readonly T[]   _values;    // re-usable
    private readonly int[] _indices;   // null/empty => dense; inaczej sparse (parallel do _values)
    private readonly int   _count;
    public readonly int    Length;
    public ReadOnlySpan<T>   GetValues()  => _values.AsSpan(0, _count);
    public ReadOnlySpan<int> GetIndices() => IsDense ? default : _indices.AsSpan(0, _count);
}

// 2. Editor — readonly ref struct (stack-only, nie ucieknie, nie zaboksuje się)
public readonly ref struct VBufferEditor<T> {
    public readonly Span<T>   Values;
    public readonly Span<int> Indices;
    public bool CreatedNewValues { get; }   // czy growth wymusił nową alokację?
    public VBuffer<T> Commit()              => new VBuffer<T>(_logicalLength, Values.Length, _values, _indices);
    public VBuffer<T> CommitTruncated(int physicalCount); // logiczna truncacja bez realokacji
}

// 3. Wejście przez `scoped ref` — sygnał dla JIT/kompilatora że ref nie ucieka
var editor = VBufferEditor.Create(ref dst, length, keepOldOnResize: true);
for (int i = 0; i < length; i++) editor.Values[i] = ...;
dst = editor.Commit();   // dst znów immutable, tablica ta sama
```

**Kluczowe decyzje projektowe:**
- **Editor *przejmuje na własność* tablice ze starego bufora.** Dokumentacja w kodzie wprost: *„the resulting VBufferEditor is assumed to take ownership of this passed-in object... its underlying buffers are being potentially reused"*. Stary `VBuffer` po przekazaniu jest nieważny.
- **`keepOldOnResize`** — flaga steruje `Array.Resize` (zachowaj dane) vs `new T[]` (porzuć). To samo co `keepOld` w Roslynowym `EnsureSize`.
- **`CreatedNewValues`** — editor mówi callerowi czy doszło do realokacji, żeby ten przeliczył cache'owane spany. Hot path: gdy `false`, można densyfikować/przesuwać *in-place*.
- **`CommitTruncated`** — caller alokuje konserwatywnie nadmiarową pojemność (gdy końcowy rozmiar nieznany), potem obcina **logicznie** (zmienia `_count`), bez realokacji tablicy.

**Why:** rozdzielenie immutable-API od mutable-mechanizmu daje jednocześnie (a) bezpieczny, niewspółdzielący stanu typ publiczny, (b) zero-alloc reużycie w pętli po milionach elementów. `ref struct` na edytorze gwarantuje stack-only lifetime — editor nie wycieknie do pola, nie zaboksuje się, nie przeżyje scope. To uogólnienie `ValueStringBuilder` na dowolny „rebuilder" kolekcji — w tym dwutablicowy (sparse: values + indices).

**How to apply:** dla każdego typu kolekcji/wektora budowanego przyrostowo lub reużywanego w gorącej pętli — nie rób mutable typu publicznego. Zrób trójkę: `readonly struct` (immutable API) + `readonly ref struct` Editor (mutacja przez `Span`) + `Commit()`. Wejście do edytora przez `scoped ref` na starym buforze. Łączy się z [[dotnet-perf-reference]] #26 (ValueStringBuilder) i #5 (segmented arrays — jeśli bufor bywa > LOH).

## 62. `ValueGetter<T>` — delegat „push-into-caller-ref" zamiast `T Get()`

Źródło: `Microsoft.ML.DataView/IDataView.cs` (`delegate void ValueGetter<TValue>(ref TValue value);`), `IValueMapper.cs` (`delegate void ValueMapper<TSrc,TDst>(in TSrc src, ref TDst dst);`)

Klasyczny kontrakt `TValue Get()` przy strumieniu wartości tego samego kształtu **alokuje albo kopiuje na każde wywołanie** — typ referencyjny = nowy obiekt, duży struct = obronna kopia, `VBuffer` = nowa tablica. ML.NET odwraca kontrakt:

```csharp
// NIE: TValue Get();           — każde wywołanie produkuje wartość
// LECZ:
public delegate void ValueGetter<TValue>(ref TValue value);          // pisz do bufora callera
internal delegate void ValueMapper<TSrc,TDst>(in TSrc src, ref TDst dst); // transformacja: in + ref

// Użycie — caller alokuje bufor RAZ, getter reużywa go przez cały strumień:
ValueGetter<VBuffer<float>> getter = cursor.GetGetter<VBuffer<float>>(column); // pobrany raz, przed pętlą
VBuffer<float> buf = default;                  // jedna alokacja na cały przebieg
while (cursor.MoveNext())
    getter(ref buf);                           // 0 alokacji per wiersz — tablica w buf reużyta
```

**Why:** wartość nie jest *zwracana* — getter **wpisuje ją do bufora należącego do wywołującego**. Caller jest właścicielem cyklu życia i reużywa ten sam `buf` (a wewnątrz: tę samą tablicę — patrz #61) przez miliony iteracji. `in TSrc` (readonly ref) na wejściu mappera eliminuje obronną kopię dużego structu; `ref TDst` na wyjściu eliminuje alokację wyniku. Delegat pobierany jest **raz przed pętlą** — koszt rozwiązania kolumny/typu zamortyzowany.

**How to apply:** gdy masz strumień wartości o stałym kształcie (wiersze, próbki, ramki) — zamiast `T Get()` daj `void Get(ref T)`. Caller deklaruje bufor poza pętlą i przekazuje go przez `ref`. Dla transformacji: `void Map(in TSrc, ref TDst)`. Pełni rolę zero-alloc odpowiednika `IEnumerator<T>.Current` — łączy się z wzorcem kursora (`long` position + `MoveNextCore`) i z [[dotnet-perf-reference]] #61 (getter dostaje `ref VBuffer` i edytuje go przez editor).

---

## TL;DR część 8

| # | Wzorzec | Kiedy |
|---|---------|-------|
| 61 | `immutable struct` + `ref struct` editor + `Commit()` | Każdy typ kolekcji/wektora reużywany lub budowany przyrostowo w hot path |
| 62 | Delegat `void Get(ref T)` zamiast `T Get()` | Strumień wartości o stałym kształcie — caller reużywa bufor przez `ref` |

**Czego z ML.NET NIE kopiować:** `AlignedArray` z ręcznym realignmentem (od .NET 6 jest `GC.AllocateArray(pinned: true)` + `NativeMemory.AlignedAlloc`); `unsafe char*` w `StringSpanOrdinalKey` (na .NET 9+ jest `Dictionary.GetAlternateLookup<ReadOnlySpan<char>>`); własna pula `BufferPoolManager` poza zakresem LOH (`ArrayPool<T>.Shared` jest lepszy dla < 85 kB). Architektura SIMD `Microsoft.ML.CpuMath` (trójpoziomowy dispatch AVX/SSE/scalar, kaskada szerokości wektora, tablice masek wyrównania, dispatch FMA, redukcja drzewiasta) to trzeci — po BCL i Kestrelu (#25) — wzorcowy przykład organizacji „SIMD + scalar fallback", wart osobnej lektury jeśli liczysz na tablicach liczbowych.

---

*Część 8 dopisana 2026-05-16. Cross-reference z `dotnet/machinelearning`. Łącznie **~77 pojęć** w 8 częściach. ML.NET jako czwarta „szkoła" perf — high-throughput numeric pipeline: agresywne reużycie buforów (editor pattern), kontrakty push-ref (`ValueGetter`), SIMD z pełną kaskadą.*

---

# Część 9 — `Microsoft.ML.CpuMath` deep dive: anatomia SIMD + scalar fallback

Źródło: `src/Microsoft.ML.CpuMath/` (`AvxIntrinsics.cs` 71 kB, `SseIntrinsics.cs` 55 kB, `CpuMathUtils.*.cs`, `Thunk.cs`, `.csproj`). Część 8 wymieniła CpuMath jako trzeci — po BCL i Kestrelu (#25) — wzorcowy przykład organizacji SIMD. Tu pełna sekcja, bo **architektura tej biblioteki to gotowy szablon** dla każdego kodu liczącego na tablicach `float`/`int`.

## 63. Build-time fork per Target Framework — jeden publiczny typ, dwie implementacje

`Microsoft.ML.CpuMath.csproj` targetuje `netstandard2.0;net8.0` i rozdziela kod **warunkowym `<Compile Remove>`**:

```xml
<ItemGroup Condition="'$(TargetFramework)' == 'netstandard2.0'">
  <Compile Remove="CpuMathUtils.netcoreapp.cs" />
  <Compile Remove="SseIntrinsics.cs" />     <!-- managed intrinsics tylko na net8 -->
  <Compile Remove="AvxIntrinsics.cs" />
</ItemGroup>
<ItemGroup Condition="'$(TargetFramework)' == 'net8.0'">
  <Compile Remove="CpuMathUtils.netstandard.cs" />
</ItemGroup>
```

- `net8.0` → `CpuMathUtils.netcoreapp.cs` + `SseIntrinsics.cs` + `AvxIntrinsics.cs` — **managed** `System.Runtime.Intrinsics`.
- `netstandard2.0` → `CpuMathUtils.netstandard.cs` → `Thunk.cs` z `[DllImport("CpuMathNative")]` — **natywny C++ DLL** (bo `System.Runtime.Intrinsics` nie istnieje na netstandard).
- Publiczny typ to **`internal static partial class CpuMathUtils`** — ta sama nazwa, identyczna sygnatura metod (`MatrixTimesSource`, `Sum`, `DotU`...), dwie rozłączne implementacje. Konsument nie wie którą dostał.
- Pakiet NuGet wozi `build/netstandard2.0/Microsoft.ML.CpuMath.props` — MSBuild props, które na starym TFM dorzucają natywny `CpuMathNative` do output (managed-only paczka nie ma jak).

**Why:** to ten sam idiom co MessagePack `UnsafeMemory32`/`UnsafeMemory64` (#31), tylko granica przebiega po **frameworku, nie architekturze**. Stara platforma bez intrinsics nie dostaje wolnego fallbacku skalarnego — dostaje natywny C++. `partial class` pozwala trzymać wspólne stałe/helpery w trzecim pliku.

**How to apply:** gdy biblioteka ma wspierać i nowy, i stary runtime — nie `#if` wewnątrz metod (nieczytelne), lecz **osobne pliki + `<Compile Remove>` per TFM + `partial class`**. P/Invoke do natywnego DLL jako fallback tam, gdzie managed-SIMD nie ma.

## 64. Trójpoziomowy runtime dispatch — `IsSupported` WYNIESIONE poza pętlę

`CpuMathUtils.netcoreapp.cs` — fasada sprawdza zdolności CPU raz i routuje do osobnych klas:

```csharp
public static void MatrixTimesSource(bool transpose, AlignedArray matrix, ...)
{
    if (Avx.IsSupported)        AvxIntrinsics.MatMul(matrix, source, destination, ...);
    else if (Sse.IsSupported)   SseIntrinsics.MatMul(matrix, source, destination, ...);
    else { /* skalarna pętla zagnieżdżona — dot product ręcznie */ }
}
```

- Implementacje AVX i SSE żyją w **osobnych klasach** (`AvxIntrinsics`, `SseIntrinsics`), nie w jednej metodzie z `if`-ami.
- `Xxx.IsSupported` to `[Intrinsic]` rozwiązywany przez JIT do **stałej** w czasie kompilacji tieru 1 — martwa gałąź jest eliminowana, brak realnego `if` w wygenerowanym kodzie.
- Sprawdzenie jest **raz, na wejściu**, nigdy w gorącej pętli.

**Why:** branch na zdolność ISA wewnątrz pętli po milionach elementów = katastrofa predykcji. Wyniesiony na zewnątrz + intrinsic-folding JIT-a = zero kosztu.

**How to apply:** fasada per operacja → `if (Avx...) else if (Sse...) else scalar` → osobny plik/klasa per ISA. Łączy się z [[dotnet-perf-reference]] #25.

## 65. Tablice masek wyrównania — bezgałęziowa obsługa głowy i ogona

`AvxIntrinsics.cs` trzyma dwie statyczne tablice `uint[64]` = 8 masek po 8 lanów:

```csharp
public static readonly uint[] LeadingAlignmentMask = new uint[64] {
    0,0,0,0,0,0,0,0,                          // wiersz 0: 0 lanów
    0xFFFFFFFF,0,0,0,0,0,0,0,                  // wiersz 1: pierwszy lan
    0xFFFFFFFF,0xFFFFFFFF,0,0,0,0,0,0, ...     // wiersz k: pierwsze k lanów
};
// TrailingAlignmentMask: lustrzanie — wiersz k = ostatnie k lanów
```

Kernel `Sum` (i każdy inny) używa ich tak:

```csharp
int misalignment = (int)((nuint)pValues % 32);     // przesunięcie adresu w bajtach
if (misalignment != 0) {
    misalignment >>= 2;  misalignment = 8 - misalignment;   // ile floatów do granicy
    Vector256<float> mask = Avx.LoadVector256(pLeadingMask + misalignment * 8);
    result = Avx.Add(result, Avx.And(mask, Avx.LoadVector256(pValues)));  // 1 maskowany load
    pValues += misalignment;  length -= misalignment;       // teraz adres wyrównany
}
// ... główna pętla wyrównana, 8 floatów/iterację ...
if (remainder != 0) {                                       // ogon 1-7 elementów
    pValues -= (8 - remainder);                             // cofnij, by load sięgnął końca
    Vector256<float> mask = Avx.LoadVector256(pTrailingMask + remainder * 8);
    result = Avx.Add(result, Avx.And(mask, Avx.LoadVector256(pValues)));
}
```

**Why:** niewyrównana głowa (0-7 elementów do granicy 32 B) i ogon (0-7 reszty) klasycznie wymagają **pętli skalarnej z brachem na element**. Maska załatwia oba jednym wektorowym `load + AND + add` — bezgałęziowo, pełną szerokością. Tablica masek to jednorazowy koszt statyczny.

**How to apply:** dla każdej operacji redukcji/transformacji na tablicy o dowolnej długości i wyrównaniu — prekomputuj dwie tablice masek (leading/trailing), obsłuż brzegi maskowanym loadem zamiast pętli skalarnej. Edge case: gdy `misalignment & 3 != 0` (adres niewyrównany nawet do 4 B w siatce 32 B) — wyrównania nie da się osiągnąć krokiem float, idzie się czystą ścieżką unaligned.

## 66. Kaskada szerokości wektora — osobny akumulator per tier, łączone na końcu

`AvxIntrinsics.DotU` (iloczyn skalarny) — wzorzec 256→128→skalar:

```csharp
Vector256<float> result256 = Vector256<float>.Zero;
while (pSrcCurrent + 8 <= pSrcEnd) {                    // TIER 1: po 8 floatów
    result256 = MultiplyAdd(pSrcCurrent, Avx.LoadVector256(pDstCurrent), result256);
    pSrcCurrent += 8; pDstCurrent += 8;
}
result256 = VectorSum256(result256);                    // redukcja drzewiasta 8→1
Vector128<float> resultPadded = Sse.AddScalar(result256.GetLower(), GetHigh(result256));

Vector128<float> result128 = Vector128<float>.Zero;
if (pSrcCurrent + 4 <= pSrcEnd) {                       // TIER 2: jeden blok 4 floatów
    result128 = Sse.Add(result128, Sse.Multiply(Sse.LoadVector128(pSrcCurrent), Sse.LoadVector128(pDstCurrent)));
    pSrcCurrent += 4; pDstCurrent += 4;
}
result128 = SseIntrinsics.VectorSum128(result128);

while (pSrcCurrent < pSrcEnd) {                         // TIER 3: skalar, 0-3 reszty
    result128 = Sse.AddScalar(result128, Sse.MultiplyScalar(
        Sse.LoadScalarVector128(pSrcCurrent), Sse.LoadScalarVector128(pDstCurrent)));
    pSrcCurrent++; pDstCurrent++;
}
return Sse.AddScalar(result128, resultPadded).ToScalar();   // SKLEJ wyniki wszystkich tierów
```

**Why:** najszersza dostępna pętla (256 b) miele większość danych; 128 b i skalar dobierają resztę. Każdy tier ma **własny akumulator** — częściowych sum nie gubi się przy przejściu między szerokościami, sklejane są dopiero na końcu. Macierzowe kernele (`MatMul`) dokładają drugi wymiar tej techniki: 4 wiersze wyjścia liczone w 4 osobnych rejestrach naraz (ILP — instrukcje niezależne, procesor je zrównolegla).

**How to apply:** każda redukcja/mapowanie na tablicy → kaskada `Vector256` → `Vector128` → skalar, osobny akumulator per tier, sklej na końcu. Dla kerneli „dużo iloczynów skalarnych" dodatkowo rozwiń pętlę N× z N akumulatorami.

## 67. Mikro-dispatch wewnątrz kernela — FMA i gather jako inline-owane helpery

W środku kernela bywają punktowe sprawdzenia ISA, opakowane w `[MethodImpl(AggressiveInlining)]`:

```csharp
[MethodImpl(MethodImplOptions.AggressiveInlining)]
private static Vector256<float> MultiplyAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c)
    => Fma.IsSupported ? Fma.MultiplyAdd(a, b, c)            // 1 instrukcja, lepsza precyzja
                       : Avx.Add(Avx.Multiply(a, b), c);    // fallback: mul + add

[MethodImpl(MethodImplOptions.AggressiveInlining)]
private static unsafe Vector256<float> Load8(float* src, int* idx)
    => Avx2.IsSupported ? Avx2.GatherVector256(src, Avx.LoadVector256(idx), 4)  // 8 rozproszonych → 1 load
                        : Vector256.Create(src[idx[0]], src[idx[1]], ... src[idx[7]]); // ręcznie
```

**Why:** `Fma.IsSupported`/`Avx2.IsSupported` to znów intrinsic-stałe — JIT eliminuje martwą gałąź, `AggressiveInlining` (#3) wciąga helper do kernela, więc w hot loopie nie ma ani calla, ani brancha. FMA `a*b+c` to jedna instrukcja (Haswell+) o niższej latencji i bez pośredniego zaokrąglenia. `GatherVector256` ładuje 8 rozproszonych wartości (sparse!) jedną instrukcją.

**How to apply:** punktowe rozgałęzienia ISA trzymaj w maleńkich `AggressiveInlining` helperach z operatorem `?:` na `IsSupported` — nie inline'uj `if`-ów ręcznie w kernelu.

## 68. Redukcja drzewiasta w rejestrze — `HorizontalAdd`/`Shuffle`, nie pętla skalarna

Zwinięcie wektora do skalara (suma, max) przez drzewo operacji, głębokość log₂(N):

```csharp
private static Vector256<float> VectorSum256(in Vector256<float> v) {
    Vector256<float> partial = Avx.HorizontalAdd(v, v);   // sąsiednie pary zsumowane
    return Avx.HorizontalAdd(partial, partial);            // i jeszcze raz
}
private static Vector256<float> VectorMax256(in Vector256<float> v) {
    Vector256<float> x1 = Avx.Shuffle(v, v, 0xB1);         // ABCD|EFGH -> BADC|FEHG
    Vector256<float> m  = Avx.Max(v, x1);                  // max parami
    x1 = Avx.Shuffle(m, m, 0x02);
    return Avx.Max(m, x1);                                 // max w pozycji skalarnej
}
```

**Why:** 8 elementów → 1 w 2-3 instrukcjach wektorowych zamiast 8 iteracji skalarnych. To `Shuffle` + operacja binarna w drzewie — działa dla każdej operacji łącznej (suma, max, min, AND...).

**How to apply:** akumuluj w najszerszym wektorze przez całą pętlę, redukuj do skalara dopiero raz, na końcu, drzewem `Shuffle`+op — nigdy pętlą `for` po lanach.

## 69. Drobne, ale warte skopiowania (CpuMath)

- **Konwencja sufiksów w nazwach** — komentarz nagłówkowy `AvxIntrinsics.cs` definiuje: `A` = aligned+padded, `U` = unaligned+unpadded, `P` = sparse partial vector, `Tran` = transposed. Eksport ma unikalne nazwy (`DotU`, `AddScaleSU`, `MatMulTran`) bo wariantów nie da się rozróżnić po sygnaturze. **Dyscyplina nazewnicza zamiast przeciążeń** — czytając nazwę wiesz o kontrakcie wyrównania.
- **Unaligned-load API na wyrównanych danych** — w głównej pętli używają `Avx.LoadVector256` (unaligned), mimo że dane *są* wyrównane (asercja `Contracts.Assert(addr % 32 == 0)`). Powód z komentarza w kodzie: JIT składa **unaligned** load w operand pamięci instrukcji konsumującej (VEX-encoding to pozwala), `LoadAligned` zostaje osobną instrukcją. Na nowoczesnym HW aligned i unaligned load są równie szybkie gdy nie przecinają cache-line. Wniosek: trzymaj dane wyrównane (cache-line), ale wołaj **unaligned-load intrinsic** — dla foldowalności.
- **`[SuppressUnmanagedCodeSecurity]` na P/Invoke** — wszystkie `[DllImport]` w `Thunk.cs` mają ten atrybut: pomija stack-walk bezpieczeństwa przy każdym wywołaniu natywnym (istotne na .NET Framework/netstandard; na .NET Core CAS usunięto i atrybut jest w praktyce no-op). Pokrewne ideowo do `[SuppressGCTransition]` (#46a), ale to **inny** atrybut o innym mechanizmie — nie mylić.
- **`AlignedArray`** — alokuje `new float[size + cbAlign/4]` z zapasem i przesuwa dane wewnątrz (`GetBase`), by zagwarantować wyrównanie 16/32 B bez osobnej alokacji. Wzorzec historyczny — **na .NET 6+ użyj `GC.AllocateArray<T>(n, pinned: true)` + `NativeMemory.AlignedAlloc`** zamiast powielać realignment ręcznie.

---

## TL;DR część 9 — szablon biblioteki SIMD

| # | Wzorzec | Co daje |
|---|---------|---------|
| 63 | Fork per TFM: `<Compile Remove>` + `partial class` + natywny fallback | Nowy runtime = managed intrinsics, stary = P/Invoke; jedno publiczne API |
| 64 | Trójpoziomowy dispatch AVX/SSE/scalar, `IsSupported` poza pętlą | Zero brancha ISA w hot loopie (JIT foldują intrinsic-stałą) |
| 65 | Tablice masek leading/trailing | Bezgałęziowa obsługa niewyrównanej głowy i ogona |
| 66 | Kaskada 256→128→skalar, osobny akumulator per tier | Pełne wykorzystanie szerokości + poprawna reszta bez gubienia sum |
| 67 | Mikro-dispatch FMA/gather w `AggressiveInlining` helperach | 1-instrukcyjne FMA, hardware gather; bez calla i brancha w kernelu |
| 68 | Redukcja drzewiasta `HorizontalAdd`/`Shuffle` | Wektor→skalar w log₂(N) instrukcjach, nie pętlą skalarną |
| 69 | Sufiksy nazw (A/U/P/Tran), unaligned-load dla foldowalności, `AlignedArray` (przestarzałe) | Dyscyplina kontraktu wyrównania; nuanse codegenu JIT |

Pełny przepis na bibliotekę SIMD: **fasada per operacja → dispatch ISA na wejściu → osobna klasa per ISA → w kernelu kaskada szerokości + maski brzegów + redukcja drzewiasta → mikro-dispatch FMA/gather inline → natywny fallback dla TFM bez intrinsics.** Obowiązuje przestroga z #25: SIMD opłaca się przy milionach elementów/s — dla CRUD/IDE to overengineering.

---

*Część 9 dopisana 2026-05-16. Deep dive `Microsoft.ML.CpuMath`. Łącznie **~84 pojęcia** w 9 częściach. Domknięcie wątku SIMD z #25: po BCL (laboratorium intrinsics) i Kestrelu (parsing HTTP) — ML.NET CpuMath jako trzeci, najczytelniej zorganizowany wzorzec „SIMD + scalar fallback" gotowy do skopiowania jako szablon.*

---

# Część 10 — `dotnet/aspnetcore`: 26 wzorców z żywego repozytorium

Źródło: przegląd kodu repozytorium `C:\Praca\aspnetcore` (gałąź `main`, maj 2026) — pięć równoległych przebiegów: transport Kestrela, protokoły HTTP Kestrela (HTTP/1·2·3), `src/Http` + routing + `WebUtilities`, `src/Shared` (utility source-linked), oraz Components (Blazor) / SignalR / middleware. Wylistowano **wyłącznie** wzorce nieobecne w częściach 1-9. Każdy zweryfikowany w źródle. W odróżnieniu od dekompilacji z części 1-4 to **kod pierwotny z komentarzami autorów** — komentarze cytowane dosłownie są bezcennym uzasadnieniem.

## A. Nowa klasa — codegen / JIT

### 70. `[UnsafeAccessor]` — dostęp do prywatnych składowych bez refleksji

Źródło: `src/Shared/Components/ComponentsActivityLinkStore.cs`

```csharp
[UnsafeAccessor(UnsafeAccessorKind.Method, Name = "get_ActivityLinksStore")]
static extern object GetActivityLinksStore(Renderer instance);
```

**Why:** atrybut .NET 8+ — JIT wiąże `extern` metodę bezpośrednio z prywatną składową typu docelowego. Wynik: zwykłe `call`, zero `MethodInfo.Invoke`, zero stack-walku bezpieczeństwa, zero alokacji `object[]` na argumenty. W odróżnieniu od `Expression.Compile`/`DynamicMethod` ([[dotnet-perf-reference]] #39-41) nie emituje IL w runtime — jest w pełni statyczny, trim-friendly i AOT-friendly.

**How to apply:** gdy musisz wywołać prywatną metodę/pole/konstruktor cudzego typu (interop z BCL, testy, mostki między assembly) — zadeklaruj `static extern` z `[UnsafeAccessor]` zamiast cache'owanego `MethodInfo`. Sygnatura musi pasować; pierwszy parametr to instancja. Łączy się z [[dotnet-perf-reference]] #41 (codegen) jako jego statyczna, lekka alternatywa.

### 71. `SearchValues<T>` — zbiór znaków skompilowany do SIMD

Źródło: `src/Shared/ServerInfrastructure/HttpCharacters.cs`, `src/Http/Http.Abstractions/src/PathString.cs`

```csharp
private static readonly SearchValues<byte> _allowedAuthorityBytes =
    SearchValues.Create(":.-[]@0123456789ABC...xyz"u8);

public static bool ContainsInvalidAuthorityChar(ReadOnlySpan<byte> span)
    => span.IndexOfAnyExcept(_allowedAuthorityBytes) >= 0;
```

**Why:** `SearchValues<T>` (.NET 8+) analizuje zbiór raz, przy inicjalizacji statycznej, i wybiera najszybszy algorytm wyszukiwania (bitmapa / SIMD / per-znak). `IndexOfAnyExcept` daje walidację „czy wszystko w dozwolonym zbiorze" w tempie wektorowym. `PathString.ToUriComponent()` używa tego jako fast-path: jeśli `SearchValues` nie znajdzie znaku do escapowania — zwraca string bez `StringBuilder`.

**How to apply:** każda walidacja/klasyfikacja znaków (dozwolone znaki URL, separatory, whitelist) — zamień pętlę `for` po znakach lub regex na statyczny `SearchValues<char>`/`<byte>` + `IndexOfAny`/`IndexOfAnyExcept`. Najpierw fast-path „czy w ogóle trzeba coś robić", dopiero potem wolna ścieżka. Pokrewne #25 (SIMD), ale bez ręcznych intrinsics.

### 72. Source-generated `KnownHeaders` — dopasowanie nazw nagłówków przez odczyt 8 bajtów jako `ulong`

Źródło: `src/Servers/Kestrel/Core/src/Internal/Http/HttpHeaders.Generated.cs` (generowane z `KnownHeaders.cs`)

```csharp
// case-insensitive: 0xdfdfdfdf zeruje 6. bit każdego bajtu ASCII (49× w pliku)
if ((ReadUnalignedLittleEndian_ulong(ref nameStart) & 0xdfdfdfdfdfdfdfdfuL) == 0x...)
// long _bits — jeden bit na znany nagłówek:
_bits |= 0x2L;                       // set
if ((_bits & 0x2L) != 0) { ... }     // test obecności — O(1)
```

**Why:** nazwa nagłówka czytana jako liczba (`Unsafe.ReadUnaligned<ulong>`) i porównywana ze stałą — O(1) niezależnie od długości, bez alokacji stringa, bez `string.Equals`. Maska `0xdf...` daje **bezgałęziową** case-insensitivity. `long _bits` mieści ~60 znanych nagłówków: sprawdzenie „czy nagłówek już ustawiony" to jeden test bitu, jeden dostęp do cache line.

**How to apply:** parsując protokół ze skończonym słownikiem nazw — wygeneruj (source generator) kod czytający nazwę jako 2/4/8-bajtowe liczby i porównujący ze stałymi; do śledzenia obecności pól użyj bit-flag `long`/`ulong` zamiast `HashSet`. Rozszerza #4 (StringTable) o **codegen dopasowania**, a #6 (frozen) o przypadek słownika znanego w czasie kompilacji.

## B. Triki bitowe / sentinel

### 73. Fuzja bounds-check i sentinela przez rzut na `uint`

Źródło: `src/Servers/Kestrel/Core/src/Internal/Http/HttpParser.cs`

```csharp
var index = (uint)span.IndexOf(target);   // -1 (brak) → uint.MaxValue
if (index < (uint)span.Length) { ... }    // jeden branch zamiast (index >= 0 && index < len)
```

**Why:** `IndexOf` zwraca `-1` przy braku trafienia. Rzut na `uint` zamienia `-1` w `uint.MaxValue` — pojedyncze porównanie bez znaku łapie jednocześnie „nie znaleziono" i „poza zakresem". Mniej gałęzi = lepsza predykcja na hot path parsera. JIT używa tego też do eliminacji bounds-checków.

**How to apply:** gdy masz indeks, który bywa `-1`, i tak czy tak musisz sprawdzić górną granicę — rzutuj na `uint` i porównaj raz. Idiom uzupełnia #46e (fastest loops) o konkretną sztuczkę sentinelową.

### 74. Bezgałęziowe wyliczanie długości prefiksu hex (chunked encoding)

Źródło: `src/Servers/Kestrel/Core/src/Internal/Http/ChunkWriter.cs`

```csharp
total = (count > 0xffff) ? 0x10 : 0x00;  count >>= total;
shift = (count > 0x00ff) ? 0x08 : 0x00;  count >>= shift;  total |= shift;
total |= (count > 0x000f) ? 0x04 : 0x00;        // pozycja najwyższego niezerowego nibble
// ...write hex prosto do Span, hex z "0123456789abcdef"u8 (sekcja danych)
```

**Why:** wyszukiwanie binarne najwyższego niezerowego nibble — O(1), bez `div`/`mod`, bez pętli, bez alokacji. Rozmiar chunku zapisywany wprost do `Span<byte>`; tablica hex to literał `u8` mapowany do sekcji rodata (zero alokacji). `& 0x0f` przy indeksowaniu daje JIT-owi dowód na eliminację bounds-checku.

**How to apply:** konwersje o znanej z góry maksymalnej szerokości (hex, rozmiary ramek) — dekomponuj przez przesunięcia bitowe i pisz do dostarczonego bufora; tablice translacji trzymaj jako literały `u8`. Pokrewne #46d (`string.Create`), ale jeszcze niżej — wprost do `Span`.

### 75. Ustawianie continuation-bit w 7-bitowym varint przez OR zamiast brancha

Źródło: `src/Shared/Encoding/Int7BitEncodingUtils.cs`

```csharp
while (uValue > 0x7Fu) {
    target[index++] = (byte)(uValue | ~0x7Fu);   // ~0x7F = 0xFFFFFF80 — ustawia bit ciągłości
    uValue >>= 7;
}
target[index++] = (byte)uValue;
```

**Why:** klasyczny varint ustawia 8. bit warunkowo. Tu `uValue | ~0x7F` ustawia bit ciągłości **zawsze**, a rzut `(byte)` odcina nadmiar — bez gałęzi, bez mispredykcji. Operuje wprost na `Span<byte>` (zero `BinaryWriter`). Odczyt waliduje `shift == 35` (przepełnienie >32 bit) — wczesne wykrycie złych danych.

**How to apply:** kodując liczby o zmiennej długości do bufora — zamień warunkowe ustawianie flagi na bezwarunkowy OR + maskujący rzut. Pokrewne #7/#19 (parsery protokołów).

### 76. Perfect hash dla metod HTTP (generowany gperf-em)

Źródło: `src/Servers/Kestrel/Core/src/Internal/Infrastructure/HttpUtilities.cs`

```csharp
// "GET " czytane jako uint (z trailing space) — sprawdzane PIERWSZE (najczęstsze)
// dłuższe metody: do 8 bajtów jako ulong + tablica perfect-hash z GNU gperf
```

**Why:** metody HTTP to krótkie, stałe stringi ASCII — porównanie liczbowe jest O(1) i nie liczy hash stringa. Tablica perfect-hash (bezkolizyjna, minimalna) wygenerowana **offline** narzędziem gperf; lookup to 2-3 odczyty pamięci. Najczęstsza metoda (`GET`) ma osobny, najwcześniejszy fast-path.

**How to apply:** dla każdego skończonego słownika (metody, kody, schematy) — rozważ perfect hash generowany offline zamiast `Dictionary`/`switch` na stringach; uprzywilejuj statystycznie najczęstszy przypadek. Konkretyzacja idei z #6 (frozen collections).

## C. Pamięć, write-barriers, pooling

### 77. Struct-wrapper omijający kontrolę kowariancji przy zapisie do tablicy

Źródło: `src/Shared/Buffers/BufferSegmentStack.cs`, `src/Servers/Kestrel/shared/PooledStreamStack.cs`

```csharp
// komentarz w kodzie wskazuje wprost: clr!ObjIsInstanceOf
private readonly struct SegmentAsValueType {
    private readonly BufferSegment _value;
    private SegmentAsValueType(BufferSegment v) => _value = v;
    public static implicit operator SegmentAsValueType(BufferSegment s) => new(s);
    public static implicit operator BufferSegment(SegmentAsValueType s) => s._value;
}
private readonly SegmentAsValueType[] _array;   // tablica STRUKTUR, nie referencji
```

**Why:** zapis typu referencyjnego do `T[]` wymusza w CLR kontrolę covariant-array-store (`JIT_Stelem_Ref` → `ObjIsInstanceOf`) — narzut na **każdy** zapis. Opakowanie referencji w `readonly struct` sprawia, że tablica jest tablicą wartości — kontrola znika. Operatory `implicit` czynią wrapper niewidocznym dla wołających. Microsoft zmierzył to w trace'ach ETL (komentarz w `BufferSegmentStack`).

**How to apply:** tablica/pula trzymająca typy referencyjne, zapisywana w gorącej pętli — opakuj element w jednopolowy `readonly struct` z operatorami `implicit`. Eliminuje ukryty `ObjIsInstanceOf`. Skrajnie nieoczywiste; warte tylko zmierzonych hot-pathów.

### 78. `nuint` jako wartość w `ConcurrentDictionary` — eliminacja GC write barriers

Źródło: `src/Servers/Kestrel/Core/src/Internal/PinnedBlockMemoryPoolFactory.cs`

```csharp
// micro-optimization: Using nuint as the value type to avoid GC write barriers;
// could replace with ConcurrentHashSet if that becomes available
private readonly ConcurrentDictionary<PinnedBlockMemoryPool, nuint> _pools = new();
_pools.TryAdd(pool, nuint.Zero);   // wartość nieistotna — słownik użyty jako zbiór
```

**Why:** `ConcurrentDictionary` użyty jako `HashSet` (brak `ConcurrentHashSet` w BCL). Gdyby wartością był `object`, każdy wpis byłby GC-rootem wymagającym **write barrier**. `nuint` to typ wartościowy — wpis nie uczestniczy w skanowaniu GC, zapis bez bariery. Cytat z komentarza autora.

**How to apply:** potrzebujesz współbieżnego zbioru → `ConcurrentDictionary<T, nuint>` (lub inny value-type jak `byte`) zamiast `<T, object>`. Wartość ignorowana, ale znika narzut bariery zapisu. Pokrewne #4 (false sharing) — świadomość kosztów pracy GC z pamięcią.

### 79. Pulowanie przez nadpisanie `Dispose()`

Źródło: `src/Shared/CancellationTokenSourcePool.cs`

```csharp
private sealed class PooledCancellationTokenSource : CancellationTokenSource {
    protected override void Dispose(bool disposing) {
        if (disposing && !_pool.Return(this))   // próba zwrotu do puli
            base.Dispose(disposing);             // realny dispose tylko gdy pula pełna
    }
}
```

**Why:** `CancellationTokenSource` jest relatywnie drogi w alokacji. Dziedzicząc i przechwytując `Dispose`, pula staje się **całkowicie przezroczysta** dla wołającego — zwykłe `using (var cts = pool.Rent())` faktycznie oddaje obiekt do puli. Bound puli (`MaxQueueSize = 1024`) — po przekroczeniu graceful fallback do prawdziwego `Dispose`.

**How to apply:** typ, który już implementuje `IDisposable` i jest drogi — rozważ podklasę przechwytującą `Dispose` i oddającą `this` do puli. Wołający nie zmienia kodu. Uwaga: tylko gdy typ jest dziedziczalny i bezpieczny do resetu. Pokrewne #1 (object pooling) — wariant „pula niewidoczna".

### 80. Liczenie „na oko" zamiast `ConcurrentQueue.Count`

Źródło: `src/Servers/Kestrel/Transport.Sockets/src/Internal/SocketSenderPool.cs`, `src/Shared/CancellationTokenSourcePool.cs`

```csharp
// "This counting isn't accurate, but it's good enough ... to avoid using
//  _queue.Count which could be expensive"
if (_disposed || Interlocked.Increment(ref _count) > MaxQueueSize) {
    Interlocked.Decrement(ref _count);
    sender.Dispose();   // pula pełna — porzuć
    return;
}
```

**Why:** `ConcurrentQueue.Count` jest drogie (synchronizacja, przejście segmentów). Osobny licznik `int` z `Interlocked` jest tani; może chwilowo przekroczyć `MaxQueueSize` przez wyścig, ale ta drobna niedokładność jest nieporównanie tańsza niż dokładny `Count`. Cytat z komentarza.

**How to apply:** ograniczasz rozmiar współbieżnej puli/kolejki — trzymaj przybliżony licznik `Interlocked` zamiast wołać `.Count`. Zaakceptuj, że bound jest „miękki". Filozofia „good enough" — pokrewne #13 (LRU), #28 (custom Parallel).

### 81. `RemoveExpired` w jednym skanie dzięki uporządkowaniu FIFO

Źródło: `src/Servers/Kestrel/shared/PooledStreamStack.cs`

```csharp
// streamy w puli są w kolejności wygaśnięcia → pierwszy nie-wygasły = wszystkie dalsze ważne
for (var i = 0; i < size; i++)
    if (array[i].PoolExpirationTimestamp >= timestamp) return i;   // cutoff
// potem: dispose [0..cutoff), kompaktacja in-place, wyczyść ogon — zero alokacji
```

**Why:** inwariant „streamy dodawane są w kolejności rosnącego czasu wygaśnięcia" zamienia czyszczenie puli w pojedynczy skan + kompaktację tablicy w miejscu. Brak struktur pomocniczych, brak alokacji, w stanie ustalonym koszt bliski zera.

**How to apply:** pula z wygasaniem (TTL) — przechowuj elementy w kolejności wygaśnięcia, wtedy eviction to jeden skan do pierwszego żywego + kompaktacja. Pokrewne #13 (bounded LRU cache) — alternatywa bez `LinkedList`.

### 82. Dwa wskaźniki — przepisywanie spanu w miejscu

Źródło: `src/Shared/UrlDecoder/UrlDecoder.cs`, `src/Shared/PathNormalizer/PathNormalizer.cs`

```csharp
public static int DecodeInPlace(Span<byte> buffer, bool isFormEncoding) {
    var sourceIndex = 0; var destinationIndex = 0;
    // wynik dekodowania ZAWSZE ≤ wejściu → wskaźnik zapisu nie dogoni wskaźnika czytania
    while (sourceIndex < buffer.Length) { /* '+', '%XX', literal → buffer[destinationIndex++] */ }
    return destinationIndex;   // nowa długość
}
```

**Why:** URL-decode i RFC-3986 `RemoveDotSegments` mają inwariant „wynik nie jest dłuższy niż wejście". Dzięki temu można przepisać dane **w tym samym buforze**: wskaźnik czytania zawsze wyprzedza wskaźnik zapisu. Zero buforów pośrednich, zero alokacji; metoda zwraca nową długość do obcięcia.

**How to apply:** transformacja bufora, która nigdy nie wydłuża danych (dekodowanie, usuwanie znaków, kompresja whitespace) — rób ją in-place techniką dwóch wskaźników zamiast alokować bufor wyjściowy. Pokrewne #61 (editor pattern) i #46c (`CollectionsMarshal.AsSpan`).

### 83. Numer rewizji — tanie unieważnianie cache przy reuse z puli

Źródło: `src/Http/Http/src/DefaultHttpContext.cs` (`Initialize` / `Uninitialize`)

```csharp
public void Initialize(IFeatureCollection features) {
    _features.Initialize(features, revision: features.Revision);   // bump rewizji
    _request.Initialize(_features.Revision);
    // featurowe lookupy porównują zapamiętaną rewizję z bieżącą — niezgodność = przelicz
}
```

**Why:** `HttpContext` jest pulowany między żądaniami. Zamiast zerować całą strukturę i wszystkie cache feature'ów przy reuse, inkrementuje się jeden `int Revision`. Cache'owane lookupy wykrywają nieaktualność po niezgodności numeru — leniwie, bez zerowania pamięci.

**How to apply:** pulowany obiekt z cache'owanym stanem pochodnym — zamiast czyścić cache przy reuse, trzymaj licznik wersji; konsumenci cache zapamiętują wersję i przeliczają przy niezgodności. Pokrewne #1 (pooling), #34 (pula obiektów per-request).

## D. Współbieżność / async

### 84. `IOQueue` — `Thread.MemoryBarrier()` + bramka CAS na `_doingWork`

Źródło: `src/Servers/Kestrel/Transport.Sockets/src/Internal/IOQueue.cs`

```csharp
public override void Schedule(Action<object?> action, object? state) {
    _workItems.Enqueue(new Work(action, state));
    if (Interlocked.CompareExchange(ref _doingWork, 1, 0) == 0)     // tylko PIERWSZY kolejkuje
        ThreadPool.UnsafeQueueUserWorkItem(this, preferLocal: false);
}
void IThreadPoolWorkItem.Execute() {
    while (true) {
        while (_workItems.TryDequeue(out var item)) item.Callback(item.State);
        _doingWork = 0;
        Thread.MemoryBarrier();              // porządkuje nie-volatile zapis ↔ odczyt poniżej
        if (_workItems.IsEmpty) break;
        if (Interlocked.Exchange(ref _doingWork, 1) == 1) break;
    }
}
```

**Why:** wszystkie callbacki I/O łączą się w **jeden** work item ThreadPoola (mniej kontekst-switchy, lepsza lokalność). Flaga `_doingWork` przez CAS gwarantuje, że tylko pierwszy producent kolejkuje robotę. Jawny `Thread.MemoryBarrier()` ustawia porządek między zwykłym (nie-volatile) zapisem `_doingWork = 0` a późniejszym odczytem `IsEmpty` — bez kosztu volatile-read na każdej iteracji. Klasyczny sposób na wyścig „praca dodana po wyczyszczeniu flagi".

**How to apply:** producent/konsument, gdzie chcesz batchować pracę w jeden work item — flaga „pracuję" przez CAS, po opróżnieniu kolejki wyczyść flagę, `MemoryBarrier`, sprawdź ponownie pustość. Pokrewne #7 (lock-free), #21 (`Pipe`).

### 85. Koalescencja kontynuacji przez singletonowy sentinel `Action`

Źródło: `src/Servers/Kestrel/Transport.Sockets/src/Internal/SocketAwaitableEventArgs.cs`

```csharp
private static readonly Action<object?> _continuationCompleted = _ => { };
private volatile Action<object?>? _continuation;
// w OnCompleted: CAS wstawia albo realną kontynuację, albo sentinel
if (ReferenceEquals(prevContinuation, _continuationCompleted))
    ThreadPool.UnsafeQueueUserWorkItem(continuation, state, preferLocal: true);
```

**Why:** jeden statyczny, pusty `Action` jako sentinel pozwala jednym `Interlocked.CompareExchange` rozróżnić trzy stany awaitera (czeka / zakończono / zakończono-przed-rejestracją) bez osobnych pól flag i bez alokacji. Współgra z `IValueTaskSource`.

**How to apply:** implementując własny awaiter/`IValueTaskSource` — użyj statycznego sentinel-delegata jako wartownika stanu w polu kontynuacji; CAS na tym polu zastępuje oddzielny `enum`/`bool`. Konkretny trik uzupełniający #14 (`PipeAwaitable`).

### 86. Deduplikacja pracy async przez wyścig na `ConcurrentDictionary` + `TaskCompletionSource`

Źródło: `src/Middleware/OutputCaching/src/DispatcherExtensions.cs` (`WorkDispatcher`)

```csharp
while (true) {
    if (_workers.TryGetValue(key, out var task)) return await task;   // dołącz do trwającej pracy
    var tcs = new TaskCompletionSource<TValue?>(TaskCreationOptions.RunContinuationsAsynchronously);
    if (_workers.TryAdd(key, tcs.Task)) {                              // wygrałeś — jesteś producentem
        try     { tcs.TrySetResult(await valueFactory(key, state)); return await tcs.Task; }
        finally { _workers.TryRemove(key, out _); }
    }   // przegrałeś wyścig — pętla, dołącz do cudzego Tasku
}
```

**Why:** klasyczna ochrona przed **cache stampede / dogpile** — N równoległych żądań tego samego klucza wykonuje `valueFactory` tylko raz; reszta `await`-uje wspólny `Task` z `TaskCompletionSource`. Bez locków — synchronizacja wynika z atomowego `TryAdd`. `RunContinuationsAsynchronously` zapobiega odpalaniu kontynuacji na wątku producenta. Wyjątek propaguje się do wszystkich oczekujących.

**How to apply:** drogie obliczenie/pobranie cache'owane po kluczu, wołane współbieżnie — trzymaj `ConcurrentDictionary<TKey, Task<TValue>>`, producenta wyłaniaj przez `TryAdd`, usuwaj wpis w `finally`. Pokrewne #12 (`GetOrAdd`) — ale dla pracy **asynchronicznej**, gdzie `GetOrAdd` nie wystarcza.

### 87. `ConcurrentPipeWriter` — tryb passthrough, gdy nic się nie flushuje

Źródło: `src/Servers/Kestrel/Core/src/Internal/Infrastructure/PipeWriterHelpers/ConcurrentPipeWriter.cs`

```csharp
public override Memory<byte> GetMemory(int sizeHint = 0) {
    if (_currentFlushTcs == null && _head == null)        // brak flushu i brak danych
        return _innerPipeWriter.GetMemory(sizeHint);      // PASSTHROUGH — zero buforowania, zero locka
    AllocateMemoryUnsynchronized(sizeHint);
    return _tailMemory;
}
```

**Why:** dopóki nie trwa flush, writer deleguje wprost do wewnętrznego — żadnego segmentowego bufora, żadnej synchronizacji. Bufor segmentowy (lista reużywalnych segmentów) włącza się **tylko** na czas trwającego flushu, kiedy współbieżne zapisy muszą się gdzieś podziać. Jeden `TaskCompletionSource` na flush jest await-owany przez wszystkich piszących.

**How to apply:** dekorator dodający buforowanie/synchronizację „na wszelki wypadek" — wykryj wspólny przypadek, w którym dekoracja jest zbędna, i w nim deleguj bezpośrednio. Pokrewne #7 (`Allocate`/`AllocateSlow` — fast/slow split na poziomie metody).

## E. Hot-path / architektura

### 88. Per-connection cache reuse stringów żądania

Źródło: `src/Servers/Kestrel/Core/src/Internal/Http/Http1Connection.cs`

```csharp
private string? _parsedPath, _parsedQueryString, _parsedRawTarget;
// keep-alive: jeśli nowy raw target == poprzedni → reużyj referencji stringa, pomiń parsowanie
```

**Why:** połączenie keep-alive często dostaje **identyczne** żądania (polling, retry, healthcheck). Pola instancji trzymają zdekodowane stringi z poprzedniego żądania; jeśli surowy target się zgadza — reużywana jest gotowa referencja, bez ponownego dekodowania i alokacji. Per-connection (nie globalne) → zero kontencji. Wyłączalne `DisableStringReuse` dla multi-tenant.

**How to apply:** stanowy parser przetwarzający strumień podobnych wejść — cache'uj ostatni wynik w polach instancji i sprawdzaj tożsamość wejścia przed ponownym parsowaniem. Pokrewne #4 (`StringTable` dedupe) — ale lokalne, jednoelementowe, bez synchronizacji.

### 89. Nagłówek `Date` cache'owany przez heartbeat + `Volatile`

Źródło: `src/Servers/Kestrel/Core/src/Internal/Http/DateHeaderValueManager.cs`

```csharp
// timer heartbeatu raz/sek aktualizuje strukturę z gotowym byte[] (z wbudowanym "\r\nDate: ")
// wszystkie wątki: Volatile.Read tej samej instancji — zero formatowania per odpowiedź
```

**Why:** każda odpowiedź HTTP potrzebuje nagłówka `Date`, ale data zmienia się raz na sekundę. Heartbeat (jeden wątek tła) formatuje ją raz; gotowy `byte[]` zawiera już prefiks `"\r\nDate: "` i `\r\n`, więc emisja odpowiedzi to czysta kopia bajtów. Odczyt przez `Volatile.Read` — bez locka.

**How to apply:** wartość wysyłana w każdej odpowiedzi, a zmieniająca się rzadko — formatuj ją w tle na timerze, trzymaj gotowy bufor bajtów (z prefiksami), czytaj `Volatile`. Pokrewne #20 (batch + timer) i #60a — wzorzec „cache odświeżany w tle".

### 90. Pre-boxowany pusty enumerator

Źródło: `src/Http/Http/src/QueryCollection.cs`, `src/Http/Http/src/FormCollection.cs`

```csharp
private static readonly IEnumerator<KeyValuePair<string, StringValues>> EmptyIEnumerator
    = new Enumerator();   // pusty przypadek zboxowany RAZ, statycznie
// niepusta kolekcja → GetEnumerator zwraca struct Enumerator wprost (bez boxa)
// UWAGA z kodu: pole _dictionaryEnumerator NIE może być readonly — MoveNext mutuje stan
```

**Why:** `foreach` przez `IEnumerable<T>` boksuje struct-enumerator. Pusta kolekcja jest częsta (brak query stringu, brak formularza) — jej enumerator boksowany jest **raz**, do statycznego pola. Niepusta zwraca `struct` bezpośrednio. Komentarz „Do NOT make this readonly" przypomina pułapkę: `readonly` na polu mutowalnego struct-enumeratora powoduje obronne kopie i gubienie stanu.

**How to apply:** kolekcja często pusta, implementująca `IEnumerable<T>` — zboxuj pusty enumerator do `static readonly`, dla niepustej zwracaj `struct`. Nie oznaczaj `readonly` pól mutowalnych struct-enumeratorów. Pokrewne #54 (frugal object), #29.

### 91. `KeyValueAccumulator` — stopniowana ekspansja 1 → tablica → `List`

Źródło: `src/Http/WebUtilities/src/KeyValueAccumulator.cs`

```csharp
// 1 wartość: inline w głównym słowniku
// 2 wartości: StringValues z tablicą 2-elementową, wciąż w głównym słowniku
// 3+:          migracja do osobnego _expandingAccumulator (List, capacity 8),
//              w głównym słowniku marker (pusty StringValues) → "patrz drugi słownik"
```

**Why:** klucze multi-value w query/form mają zwykle 1-2 wartości; alokowanie `List` dla każdego jest marnotrawstwem. Akumulator przechodzi przez stany: pojedyncza wartość → tablica 2 → `List` w osobnym słowniku. Główny słownik nie puchnie listami; `List` powstaje dopiero przy 3. wartości, od razu z `capacity: 8`.

**How to apply:** zbierasz wartości grupowane po kluczu, gdzie „typowo 1-2, rzadko więcej" — implementuj stopniowaną ekspansję ze stanem inline; pełną kolekcję alokuj dopiero po przekroczeniu progu. Pokrewne #54 (frugal `StringValues`/`FrugalList`), #27 (`AdaptiveCapacityDictionary`).

### 92. `[MethodImpl(NoInlining)]` jawnie na zimnej ścieżce

Źródło: `src/Shared/ServerInfrastructure/BufferExtensions.cs` (`WriteNumeric` / `WriteNumericMultiWrite`)

```csharp
[MethodImpl(MethodImplOptions.AggressiveInlining)]
internal static void WriteNumeric(...) {
    if (number < 10  && ...) { ... }            // fast path: 1-3 cyfry inline
    else WriteNumericMultiWrite(ref bufferWriter, number);
}
[MethodImpl(MethodImplOptions.NoInlining)]      // ZIMNA ścieżka jawnie wyłączona z inline
private static void WriteNumericMultiWrite(...) { /* dzielenie, bufor scratch */ }
```

**Why:** to **doprecyzowanie #5/#7**. Plik mówił „hot path → osobna metoda `*Slow`, pomaga JIT inline'ować szybką ścieżkę". Tu kluczowy jest jawny `[NoInlining]` na zimnej metodzie: gwarantuje, że jej IL **nie powiększy** metody gorącej, dzięki czemu ta pozostaje pod progiem inline'owania. Samo wydzielenie metody nie wystarcza — JIT mógłby ją z powrotem wciągnąć.

**How to apply:** split fast/slow — oznacz gorącą `[AggressiveInlining]`, a zimną jawnie `[MethodImpl(MethodImplOptions.NoInlining)]`. Atrybut na zimnej części jest tak samo ważny jak na gorącej.

### 93. Anty-DRY w hot-path — świadoma duplikacja kodu (cytaty autorów)

Źródło: `src/Components/Components/src/RenderTree/RenderTreeDiffBuilder.cs`, `RenderTreeFrameArrayBuilder.cs`

```text
// RenderTreeDiffBuilder (metoda diff ~1000 linii):
//   "This is deliberately a very large method ... A naive 'extract methods'-type
//    refactoring will worsen perf by about 10%."  (koszt przekazywania parametrów)
// RenderTreeFrameArrayBuilder (check wzrostu bufora skopiowany do każdej metody Append*):
//   "intentionally inlined into each method because doing so improves intensive
//    rendering scenarios by around 1%."
```

**Why:** w skrajnie gorących ścieżkach (diff drzewa renderu) wydzielanie małych metod kosztuje — przekazywanie parametrów, mniej swobody JIT w alokacji rejestrów. Blazor mierzy: refaktor „extract method" pogarsza diff o ~10%, a skopiowany ręcznie check wzrostu bufora poprawia rendering o ~1%. Zamiast metod — `#region` jako „pre-inlined" sekcje + `ref struct` kontekst (`DiffContext`) przekazywany przez `ref` zamiast 8 parametrów.

**How to apply:** **tylko** w zmierzonych, najgorętszych ścieżkach — zaakceptuj duże metody i duplikację; udokumentuj decyzję i liczbę z benchmarku; organizuj kod `#region`-ami. Stan recursive algorytmu trzymaj w `ref struct` przekazywanym `ref` (por. #56 — przekazywanie parametrów to realny koszt; #61 — `ref struct` kontekst).

### 94. `WeakReference[]`-owy bounded LRU — gdy `ConditionalWeakTable` zawodzi

Źródło: `src/Shared/RoslynUtils/BoundedCacheWithFactory.cs`

```csharp
// tablica ~5 WeakReference<Entry>; trafienie przenoszone na koniec listy (LRU),
// przy zapełnieniu nadpisywany najstarszy / pierwszy martwy slot
```

**Why:** `ConditionalWeakTable` (#33) nie zadziała, gdy **wartość cyklicznie referuje klucz** — wtedy CWT nigdy nie zwolni wpisu. Rozwiązanie: mała tablica `WeakReference`, ręczna polityka LRU, twardy bound (~5). GC może zebrać wpisy, sloty są odzyskiwane bez przytrzymywania.

**How to apply:** cache klucz→wartość, gdzie wartość trzyma referencję do klucza — nie używaj `ConditionalWeakTable`; użyj ograniczonej tablicy `WeakReference` z LRU. **Uzupełnienie zastrzeżenia do #33**: CWT zawodzi przy cyklu wartość→klucz.

### 95. `ManualResetEventSlim(spinCount: 0)` dla długich oczekiwań

Źródło: `src/Servers/Kestrel/Core/src/Internal/Infrastructure/Heartbeat.cs`

```csharp
// "Wait time is long, so don't try to spin to exit early. Spinning would waste CPU time."
_stopEvent = new ManualResetEventSlim(false, spinCount: 0);
```

**Why:** `ManualResetEventSlim` domyślnie spinuje przed zaśnięciem — to opłaca się przy oczekiwaniach rzędu mikrosekund. Wątek-heartbeat czeka **sekundę** między tikami; spinowanie byłoby czystym marnowaniem CPU. Jawne `spinCount: 0` to wyłącza. Dedykowany wątek tła zamiast `Timer` daje przewidywalny tik dla tysięcy połączeń.

**How to apply:** używasz `ManualResetEventSlim`/`SemaphoreSlim` do oczekiwań **długich** (≫ mikrosekundy) — ustaw `spinCount: 0`. Domyślny spin jest zoptymalizowany pod krótkie czekanie. Pokrewne #48d (przestarzałe prymitywy locków) — dobór prymitywu do skali oczekiwania.

---

## TL;DR część 10

| # | Wzorzec | Kiedy |
|---|---------|-------|
| 70 | `[UnsafeAccessor]` | Wywołanie prywatnej składowej bez refleksji (.NET 8+, AOT-friendly) |
| 71 | `SearchValues<T>` + `IndexOfAnyExcept` | Walidacja/klasyfikacja znaków — SIMD, zero ręcznych intrinsics |
| 72 | Codegen nagłówków: `ulong`-read + maska `0xdf` + bit-flagi | Parsing protokołu o skończonym słowniku nazw |
| 73 | Rzut indeksu na `uint` (−1 → `uint.MaxValue`) | Fuzja sprawdzenia sentinela i górnej granicy w jeden branch |
| 74 | Bezgałęziowa długość prefiksu hex (kaskada `>>`) | Konwersja o znanej maks. szerokości, write do `Span` |
| 75 | `value \| ~0x7F` | Bezgałęziowe ustawianie continuation-bit w varint |
| 76 | Perfect hash (gperf) | Skończony słownik stringów; uprzywilejuj najczęstszy |
| 77 | `readonly struct`-wrapper na tablicę typów ref | Eliminacja covariant-array-store check (`ObjIsInstanceOf`) |
| 78 | `ConcurrentDictionary<T, nuint>` | Współbieżny zbiór bez GC write barriers |
| 79 | Pula przez `override Dispose()` | Przezroczyste pulowanie typu `IDisposable` |
| 80 | Przybliżony licznik `Interlocked` | Miękki bound puli zamiast drogiego `.Count` |
| 81 | Eviction jednym skanem (porządek FIFO wygaśnięcia) | Pula/cache z TTL |
| 82 | Dwa wskaźniki, przepisanie spanu in-place | Transformacja, która nigdy nie wydłuża danych |
| 83 | Numer rewizji | Tanie unieważnianie cache pulowanego obiektu (bez zerowania) |
| 84 | `MemoryBarrier` + bramka CAS `_doingWork` | Batchowanie callbacków w jeden work item |
| 85 | Singletonowy sentinel `Action` | Stan awaitera w jednym polu + CAS |
| 86 | Wyścig `ConcurrentDictionary` + `TaskCompletionSource` | Deduplikacja drogiej pracy **async** (anti-stampede) |
| 87 | Tryb passthrough w dekoratorze | Pomiń buforowanie/lock, gdy zbędne we wspólnym przypadku |
| 88 | Per-connection reuse stringów | Stanowy parser strumienia podobnych wejść |
| 89 | Heartbeat-cache + `Volatile` | Wartość w każdej odpowiedzi, zmienna rzadko |
| 90 | Pre-boxowany pusty enumerator | Kolekcja często pusta implementująca `IEnumerable<T>` |
| 91 | Stopniowana ekspansja 1→tablica→`List` | Wartości grupowane po kluczu, typowo 1-2 |
| 92 | `[NoInlining]` na zimnej ścieżce | Doprecyzowanie #5 — chroni rozmiar metody gorącej |
| 93 | Anty-DRY w hot-path (zmierzony) | Najgorętsze ścieżki: duża metoda + duplikacja zamiast extract |
| 94 | `WeakReference[]` bounded LRU | Cache, gdzie wartość referuje klucz (CWT zawiedzie — por. #33) |
| 95 | `ManualResetEventSlim(spinCount: 0)` | Oczekiwania długie (≫ µs) — wyłącz domyślny spin |

---

*Część 10 dopisana 2026-05-16. Przegląd żywego repozytorium `dotnet/aspnetcore` (gałąź `main`). Łącznie **~110 pojęć** w 10 częściach. W odróżnieniu od części 1-9 (dekompilacja + listy kuratowane) — to kod pierwotny z komentarzami autorów; cytaty z komentarzy (`SocketSenderPool` „good enough", `RenderTreeDiffBuilder` „−10%", `PinnedBlockMemoryPoolFactory` „avoid GC write barriers") dają uzasadnienie niewidoczne w zdekompilowanym IL. Cztery najbardziej nieoczywiste: #77 (struct-wrapper vs covariant-store), #78 (`nuint` vs write-barrier), #70 (`[UnsafeAccessor]`), #86 (anti-stampede).*
