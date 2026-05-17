# Microsoft Performance Best Practices — from VS 2022 Decompilation

Source: decompilation of `.NET` DLLs from `C:\Program Files\Microsoft Visual Studio\2022\Community\` using `ilspycmd` (ICSharpCode.Decompiler 10.0.1). Results synthesised from 10 key assemblies:

- `Microsoft.CodeAnalysis.dll` / `Microsoft.CodeAnalysis.CSharp.dll` (Roslyn)
- `Microsoft.VisualStudio.Threading.dll` / `Microsoft.VisualStudio.Validation.dll`
- `Microsoft.VisualStudio.Text.Data.dll` / `Microsoft.VisualStudio.Text.Logic.dll`
- `System.Collections.Immutable.dll` / `System.Memory.dll`
- `Newtonsoft.Json.dll` / `StreamJsonRpc.dll`

~3690 `.cs` files decompiled in total to `C:\Temp\vsdecomp\`.

---

## 1. Object pooling — the foundation (740+ hits)

### Canonical `ObjectPool<T>` (Roslyn)

Source: `Microsoft.CodeAnalysis.PooledObjects.ObjectPool.cs`

- **"First item" fast path** — separate `_firstItem` field + fallback array `_items[ProcessorCount * 2 - 1]`.
- **Lock-free** via `Interlocked.CompareExchange` — never `lock`.
- **Hot/Slow split** — `Allocate` (inline-able) calls `AllocateSlow` on a miss. JIT can inline only the hot path.
- `[Conditional("DEBUG")]` on validation → zero IL in Release.

```csharp
internal T Allocate() {
    T val = _firstItem;
    if (val == null || val != Interlocked.CompareExchange(ref _firstItem, null, val))
        val = AllocateSlow();
    return val;
}
```

### Pooled wrappers — a pattern to copy

Source: `PooledStringBuilder.cs`, `PooledHashSet.cs`, `PooledDictionary.cs`, `ArrayBuilder.cs`

- `GetInstance()` / `ToStringAndFree()` — idiom that enforces returning to the pool.
- **Discard-when-too-big** — `if (Builder.Capacity <= 1024) { Clear(); Free(); }` so large instances don't pin memory in the pool.
- `[Obsolete("Consider calling ToStringAndFree instead.")]` on `ToString()` so developers don't forget to return it.

---

## 2. `readonly struct` as the default (217 hits / 147 files)

Virtually every value type in Roslyn/VS is a `readonly struct`. This eliminates defensive copies made by the compiler. Examples: `SyntaxToken`, `SyntaxTrivia`, `TextSpan`, `LinePosition`, `TypeWithAnnotations`.

**Rule of thumb**: when writing a new value type, start with `readonly struct` and change it only if the build fails.

---

## 3. `[MethodImpl(AggressiveInlining)]` — deliberate, not everywhere (250 hits / 40 files)

Concentrated in:
- `System.Memory` (Span/ReadOnlySpan accessors, MemoryMarshal, BinaryPrimitives).
- `System.Collections.Frozen` (table lookups).
- `SegmentedArray` indexers.

**Lesson**: apply to getters/indexers/math operators — NOT to multi-line methods (JIT will consider them anyway).

---

## 4. Two-tier hash table (`StringTable.cs` in Roslyn)

String deduplication with two tiers:

| Tier | Size | Scope | Synchronisation |
|------|------|-------|-----------------|
| `_localTable` | 2048 | per-instance (pooled) | none (thread-local) |
| `s_sharedTable` | 65536 (bucket=16) | static | lock-free CAS |

- All sizes are powers of 2 (`& mask` instead of `%`).
- `Environment.TickCount` as initial random for bucket selection — avoids pathological chains.

---

## 5. Segmented arrays — avoiding the LOH

Source: `Microsoft.CodeAnalysis.Collections.SegmentedArray<T>`

`T[][]` (jagged) instead of `T[]`. Each segment < 85 KB → bypasses the Large Object Heap. Used by `SegmentedDictionary`, `SegmentedHashSet` for large collections in Roslyn.

**Why:** allocations > 85 KB go to the LOH, which: (a) is not compacted by default, (b) causes fragmentation, (c) has long lifetimes.

---

## 6. Frozen collections — code-generated per key shape

`System.Collections.Frozen` contains dozens of specialised subclasses:

- `OrdinalStringFrozenDictionary_LeftJustifiedSingleChar`
- `OrdinalStringFrozenDictionary_LeftJustifiedSubstring`
- `OrdinalStringFrozenDictionary_RightJustifiedCaseInsensitiveAsciiSubstring`
- `Int32FrozenDictionary`, `SmallValueTypeDefaultComparerFrozenDictionary`...

On `ToFrozenDictionary()` the data is analysed and the fastest variant is chosen — build cost is one-time, reads are O(1) with minimal instructions.

---

## 7. Lock-free synchronisation (738 hits / 57 files)

- **`Interlocked.CompareExchange`** dominates on the hot path.
- **`Volatile.Read/Write`** instead of `lock` for single-reader scenarios.
- **`ImmutableInterlocked`** — atomic operations on immutable structures.
- VS Threading has **its own**: `AsyncReaderWriterLock`, `AsyncSemaphore`, `ReentrantSemaphore`, `JoinableTaskFactory` (vs BCL — because BCL cannot handle UI thread coordination).
- `ConfigureAwait(false)` used 75 times in `Microsoft.VisualStudio.Threading` alone.

---

## 8. Span/stackalloc only at specific points (10 hits)

Despite the hype — **Span is used sparingly in higher-level layers**. It appears in:
- JSON-RPC serialisation (`TraceParent`, `HeaderDelimitedMessageHandler`).
- Parsers (`BoundStackAllocArrayCreation`, `SyntaxFacts`).

**Conclusion**: `Span<T>` makes sense when it genuinely eliminates an allocation in a hot loop — it is not a magic replacement for `string`/`array` everywhere.

---

## 9. Small but classic

- **`[Conditional("DEBUG")]`** on validators (Validate, AssertInvariants) — 0 IL in Release.
- **Boxing avoidance** — `Microsoft.CodeAnalysis.Boxes` with `static readonly object BoxedTrue/BoxedFalse`.
- **`SharedStopwatch`** — one static `Stopwatch`, no allocation per measurement.
- **`ReaderWriterLockSlimExtensions`** — RAII disposable helper, eliminates `try/finally` around locks.
- **`EmptyStruct`** — placeholder for generics with zero footprint.
- **`WeakKeyDictionary`** — a cache that does not keep objects alive.

---

## TL;DR — adoption

1. **Pool everything you allocate in a loop** — StringBuilder, List, Dictionary. The `PooledStringBuilder` + `ObjectPool` pattern is a ready-made template to copy.
2. **`readonly struct` as the default** for new value types.
3. **`Interlocked.CompareExchange` on "first item"** instead of a lock in a pool/cache.
4. **Discard a pooled object when it exceeds the size threshold** — otherwise it leaks.
5. **Hot path → separate `*Slow` method** — helps JIT inline the fast path.
6. **Arrays > LOH? → segmented (jagged)**.
7. **`[Conditional("DEBUG")]` on assertions** instead of `if (DEBUG)`.
8. **Frozen collections** for read-only lookup tables loaded once at startup.

---

---

# Part 2 — second batch of 14 DLLs

Additional assemblies: `Microsoft.VisualStudio.Composition`, `Microsoft.VisualStudio.Telemetry` (821 files), `Microsoft.VisualStudio.RpcContracts`, `Microsoft.VisualStudio.Utilities` (452 files), `Microsoft.VisualStudio.Shell.Framework` (373 files), `Microsoft.VisualStudio.Text.UI` (447 files), `Microsoft.VisualStudio.LanguageServer.Protocol`, `Microsoft.VisualStudio.Imaging`, `Microsoft.ServiceHub.Client`, `Microsoft.Bcl.AsyncInterfaces`, `Nerdbank.Streams`, `MessagePack`, `System.IO.Pipelines`, `System.Text.Json`.

## 10. `ArrayPool<byte>.Shared` as the standard for I/O buffers

Source: `System.Text.Json.PooledByteBufferWriter.cs`

An ideal pattern to copy:

```csharp
public PooledByteBufferWriter(int initialCapacity) {
    _rentedBuffer = ArrayPool<byte>.Shared.Rent(initialCapacity);
    _index = 0;
}

public void Dispose() {
    if (_rentedBuffer != null) {
        _rentedBuffer.AsSpan(0, _index).Clear();   // clear before returning (sensitive data + GC roots)
        ArrayPool<byte>.Shared.Return(_rentedBuffer);
        _rentedBuffer = null;
    }
}

private void CheckAndResizeBuffer(int sizeHint) {
    // growth strategy: doubling, but with a 2 GB guard
    int num4 = num + Math.Max(sizeHint, num);
    ...
    _rentedBuffer = ArrayPool<byte>.Shared.Rent(num4);   // rent from pool again
    // copy old → return old to pool → continue on new
}
```

**Key**: Clear() before Return() — otherwise the GC may keep alive objects referenced by the buffer.

## 11. `Lazy<T>` as the foundation of MEF/Composition (43 hits)

`Microsoft.VisualStudio.Composition.LazyServices` + `Roslyn.Utilities.RoslynLazyInitializer` use `Lazy<T>` with `LazyThreadSafetyMode.PublicationOnly` extensively — MEF components are **lazily initialised**.

**Lesson**: VS starts fast because practically nothing is loaded at startup. Every service is a lambda in `Lazy<>` or `ExportFactory<>`. First use of a feature = first initialisation of its dependencies. Apply this to every "heavy" object in your system.

## 12. `ConcurrentDictionary.GetOrAdd` as the default cache (700+ hits)

The most frequently seen caching pattern:

```csharp
private readonly ConcurrentDictionary<Type, IFormatter> _cache = new();

public IFormatter GetFormatter(Type t) =>
    _cache.GetOrAdd(t, static type => CreateFormatter(type));
```

Specific locations:
- `MessagePack.Resolvers.CachingFormatterResolver` — formatter cache per type.
- `MessagePack.Internal.ThreadsafeTypeKeyHashTable` — custom, beats ConcurrentDictionary when the key is `Type` (custom hash).
- `System.Text.Json.ReflectionEmitCachingMemberAccessor` — IL-emit once, cache the delegate forever.
- `Roslyn.Utilities.ConcurrentDictionaryExtensions` — custom GetOrAdd extensions.

**Tip from MessagePack**: when the key is `Type`, a custom hashtable with `RuntimeHelpers.GetHashCode` beats ConcurrentDictionary.

## 13. `ConcurrentLruCache<K,V>` — when GetOrAdd is not enough

Source: `Microsoft.CodeAnalysis.InternalUtilities.ConcurrentLruCache.cs`

When you need a **bounded** cache (LRU eviction), Roslyn uses **Dictionary + LinkedList + lock** instead of Concurrent. Why not `ConcurrentDictionary`?
- LRU requires an atomic operation on TWO structures (map + list).
- `lock` with short critical sections is faster here than double-CAS.

```csharp
private readonly Dictionary<K, CacheValue> _cache;
private readonly LinkedList<K> _nodeList;
private readonly object _lockObject = new object();
```

## 14. `PipeAwaitable` — custom struct-based awaiter (zero allocation)

Source: `System.IO.Pipelines.PipeAwaitable.cs`

One of the nicest low-level patterns:

- **`internal struct PipeAwaitable`** — NOT a class, NOT a Task. The awaiter itself is a value type.
- `[Flags] AwaitableState { None, Completed=1, Running=2, Canceled=4, UseSynchronizationContext=8 }` — 4 states in one int, bit operations.
- `[MethodImpl(AggressiveInlining)]` on **every** state-changing method.
- `ValueTaskSourceOnCompletedFlags` instead of a full Task — `ValueTask` does not allocate.
- `cancellationToken.UnsafeRegister(...)` (instead of `Register`) — skips `ExecutionContext` capture (10x faster).

**This shows**: when you genuinely need zero-allocation async (millions of ops/sec), you implement your own `IValueTaskSource` on a struct.

## 15. `ValueTask` in hot paths (85 hits / 48 files)

`ValueTask` instead of `Task` when:
- The operation frequently completes synchronously (data already in the buffer).
- Hot path with many calls per second.

Classic example: `PipeReader.ReadAsync` returns `ValueTask<ReadResult>` — when data is already in the buffer there is no `Task` allocation.

**Roslyn has `ValueTaskFactory`** — prebuilt `ValueTask` for common cases (TrueResult, FalseResult, EmptyResult).

## 16. `IAsyncDisposable` / `DisposeAsync` (152 hits / 55 files)

Everything that allocates async resources (streams, pipes, connections, locks) implements `IAsyncDisposable`. The old `IDisposable.Dispose()` would be blocking — `DisposeAsync` allows clean cleanup without blocking a thread.

VS Threading introduced **its own** `Microsoft.VisualStudio.Threading.IAsyncDisposable` (before the standard) — showing how important this pattern was.

## 17. `CancellationToken` propagation over IPC

Source: `StreamJsonRpc.StandardCancellationStrategy`, `Microsoft.VisualStudio.Threading.CancellableJoinComputation`

Cancellation in RPC has no native support in JSON-RPC. Microsoft implements it as:
- Each call gets a correlation ID.
- Client cancels → sends `$/cancelRequest` with that ID.
- Server translates it into a local `CancellationToken`.

**Key insight**: do not ignore `CancellationToken` at a process boundary — propagate it.

## 18. `cancellationToken.UnsafeRegister` instead of `Register` (167 hits on CancellationTokenSource)

`UnsafeRegister` skips `ExecutionContext` capture in `CallbackNode`. Use it when the callback does not rely on `AsyncLocal`/`CallContext` — which is **almost always** the case in infrastructure code. Roslyn, Pipelines, StreamJsonRpc, Nerdbank — all use it.

## 19. `MultiplexingStream` — multiple logical streams over one physical connection

Source: `Nerdbank.Streams.MultiplexingStream.cs` (41 PipeReader hits)

How VS multiplexes multiple RPC channels over one named pipe / socket:
- Each logical channel has an 8-byte header (ID + flags + length).
- Internally `PipeReader`/`PipeWriter` per channel + frame demuxer.
- Backpressure operates per-channel — a slow channel does not block others.

**This is how VS runs dozens of out-of-proc services (Roslyn server, LSP servers) over 1-2 physical connections**.

## 20. Telemetry batching — `TaskTimer` + `PersistenceTransmitter`

Source: `Microsoft.VisualStudio.Telemetry`

Instead of sending every event immediately:
- `TelemetryCollector` aggregates into a buffer.
- `TaskTimer` (custom timer with `CancellationTokenSource.CancelAfter`) — flush every N seconds.
- `PersistenceTransmitter` — if the network is unavailable, write to disk and retry later.

**Pattern**: never block the hot path with I/O. Batch + delay + persist + retry.

## 21. `Channel<T>` — absent; Microsoft favours `Pipe`/`PipeReader`

Interestingly: **0 hits for `Channel<T>`** across 24 DLLs. Microsoft Visual Studio uses **`System.IO.Pipelines.Pipe`** (binary streaming) instead of `System.Threading.Channels.Channel<T>` (typed messaging).

Reason: for binary IPC communication `Pipe` is lower-level and more controllable (buffer size, backpressure, zero-copy).

`Channel<T>` makes sense for in-process producer/consumer with .NET types. Pipe makes sense for bytes-over-wire.

---

## TL;DR Part 2 — additional patterns

| # | Pattern | When |
|---|---------|------|
| 10 | `ArrayPool<byte>.Shared.Rent` + Clear-on-return | Any buffer > 1 KB in a hot path |
| 11 | `Lazy<T>` with `PublicationOnly` | Any "heavy" singleton/service |
| 12 | `ConcurrentDictionary.GetOrAdd` static lambda | Default cache (Type → Formatter etc.) |
| 13 | Custom LRU with `lock` | Bounded cache with eviction |
| 14 | `IValueTaskSource` on a struct | Hot async path without Task allocation |
| 15 | `ValueTask` instead of `Task` | Sync-completing async methods |
| 16 | `IAsyncDisposable` | Resources requiring async cleanup |
| 17 | RPC correlation ID + reverse-cancel | Cross-process cancellation |
| 18 | `UnsafeRegister` on CT | When you don't need ExecutionContext |
| 19 | Frame-multiplexing | Multiple logical channels over 1 socket |
| 20 | Batch + timer + persist | I/O telemetry, logging, analytics |
| 21 | `Pipe` > `Channel<T>` | Binary IPC; `Channel<T>` for in-proc typed |

---

*Part 2 added 2026-05-15. 24 DLLs in total, ~7300 `.cs` files in `C:\Temp\vsdecomp\`. `Roslyn.Utilities.dll` does not exist as a separate file — its contents live inside `Microsoft.CodeAnalysis.dll`.*

---

# Part 3 — mass scan of 1000 DLLs (252,039 `.cs` files)

After scanning the 1000 largest managed Microsoft/System DLLs from VS 2022 (BCL, ASP.NET Core, Kestrel, Roslyn full stack, ML.NET, Build, MessagePack, Cosmos, etc.) **new** classes of patterns emerge that were invisible in the first batch of 24 assemblies.

## 22. `[SkipLocalsInit]` — skipping zero-initialisation of locals (1068 hits / 134 files)

The most common "hidden" perf trick in new Microsoft code. CIL has an `init` flag that forces all local variables to be zeroed. `[SkipLocalsInit]` disables it → JIT generates fewer instructions.

**Critical**: do not use if you read locals before writing (UB). In practice safe for:
- Methods that make heavy use of `stackalloc` (without this, stackalloc zeroes itself for free).
- Hot paths with simple value-type locals that are written before being read.

**Microsoft practice**: applied at **assembly level** in `AssemblyInfo.cs`. Each of VS Copilot, CoreUtility, Extensibility, LanguageServer, SolutionPersistence, ProjectSystem, Completions has its own polyfill `SkipLocalsInitAttribute.cs` (because the attribute requires .NET 5+, so down-level projects declare it internally — this is sufficient, the C# compiler recognises it by name).

## 23. `[InlineArray(N)]` (.NET 8) — fixed-size buffer in a struct

47 hits across 46 files. Files `__InlineArray2.cs`, `__InlineArray8.cs`, `__InlineArray38.cs` — these types are **emitted by the C# compiler** when it sees `Span<T> stackalloc[N]` patterns or `[InlineArray]` used manually.

**Concrete use**: `Microsoft.AspNetCore.Razor.Utilities.Checksum` stores a 32-byte hash as `InlineArray<32, byte>` in a struct — fixed-size content addressing without a heap allocation. Roslyn LSP Protocol, Razor, Workspaces, Features, Copilot — all use it.

**Lesson**: for fixed-size data (hash, UUID, fixed packets) `[InlineArray]` replaces `unsafe fixed byte[N]` with full type-system support.

## 24. `[ModuleInitializer]` — run-once code on assembly load (32 hits)

Replaces a static class constructor when you want code that runs **once, when the assembly loads**, regardless of which class is first accessed.

```csharp
internal static class StartupInitializer {
    [ModuleInitializer]
    public static void Init() {
        // e.g. DI registration, codec registration, SQLite native dll preload
    }
}
```

Concrete users: Copilot (Roslyn, CodeMappers, Common, Service, Vsix, UI.Core), Extensibility Framework, SolutionPersistence, ProjectSystem, **Microsoft.VisualStudio.Cache.SqliteInitializer** (preload native sqlite.dll), `Microsoft.Windows.SDK.NET.ProjectionInitializer` (WinRT projections).

## 25. SIMD intrinsics — where they actually live (391 hits / 32 files)

SIMD is used **very selectively**. Practically only in:

| File | What it does via SIMD |
|------|-----------------------|
| `System.Private.CoreLib\System.Text\Ascii.cs` (38) | ASCII byte validation/conversion |
| `System.Private.CoreLib\System.Text.Unicode\Utf8Utility.cs` (23) | UTF-8 validation/transcoding |
| `System.Private.CoreLib\System.Buffers\IndexOfAnyAsciiSearcher.cs` (6) | multi-byte IndexOfAny |
| `System.Private.CoreLib\System.Buffers\ProbabilisticMap.cs` (10) | string search |
| `System.Private.CoreLib\System.Buffers.Text\Base64.cs` (14) | Base64 encode/decode |
| `System.Private.CoreLib\System\HexConverter.cs` (9) | hex encoding |
| **`Kestrel.StringUtilities.cs` (25)** | hex-encoding HTTP connection IDs via `Ssse3.Shuffle` |
| `System.Text.Encodings.Web\OptimizedInboxTextEncoder.cs` (46) | HTML/JS escaping |
| `System.Numerics.Tensors\TensorPrimitives.cs` (57) | ML/AI vector math |

Pattern from Kestrel (`StringUtilities.cs:61-68`):
```csharp
if (Ssse3.IsSupported) {
    Vector128<byte> vector = Ssse3.Shuffle(
        Vector128.CreateScalarUnsafe(item3).AsByte(),
        Vector128.Create(15,15,3,15,...).AsByte()); // nibble permutation
    // ...mask, shuffle table "0123456789ABCDEF"... 
    Unsafe.WriteUnaligned(ref ..., left);  // 16 bytes in one instruction
}
else { /* scalar fallback */ }
```

**Pattern**: SIMD path + scalar fallback, gated by `Ssse3.IsSupported`/`Avx2.IsSupported`/`AdvSimd.IsSupported`. **Microsoft does not use SIMD in VS code** — only in BCL and Kestrel (request parsing).

## 26. `ValueStringBuilder` — `ref struct` instead of `StringBuilder`

Stack-allocated string builder with an `ArrayPool` fallback on growth:

```csharp
internal ref struct ValueStringBuilder {
    private char[] _arrayToReturnToPool;
    private Span<char> _chars;
    private int _pos;

    public ValueStringBuilder(Span<char> initialBuffer) {  // typically: stackalloc char[256]
        _arrayToReturnToPool = null;
        _chars = initialBuffer;
        _pos = 0;
    }
    // Grow() → ArrayPool.Rent → CopyTo → Return old
    public override string ToString() {
        string result = _chars.Slice(0, _pos).ToString();
        Dispose();  // returns rented buffer
        return result;
    }
}
```

**Microsoft polyfills this type across many projects**: ASP.NET HttpLogging, Http.Extensions, Razor, Roslyn (as `PooledStringBuilder` or `StringBuilderPool`). It is the **standard replacement** for `new StringBuilder()` in hot paths.

**Usage idiom**:
```csharp
Span<char> initialBuffer = stackalloc char[256];
var sb = new ValueStringBuilder(initialBuffer);
sb.Append(...);
return sb.ToString();  // alloc only when <= 256 is not enough
```

## 27. `AdaptiveCapacityDictionary` — array for small N, Dictionary after promotion

Source: `Microsoft.AspNetCore.Internal.AdaptiveCapacityDictionary`

```csharp
private const int DefaultArrayThreshold = 10;
internal KeyValuePair<TKey, TValue>[]? _arrayStorage;
private int _count;
// when _count > 10 → promote to Dictionary<TKey, TValue>
```

Linear scan over `_arrayStorage` beats Dictionary for a small number of keys (hashCode overhead + indirection). Only above 10 elements does it switch to a hash.

**Mainly used in**: HTTP routing (typically 3-5 route values), HTTP headers (typically 10-15), MVC binders. Anywhere `N` is usually small but can occasionally be large.

## 28. `RoslynParallel` — custom wrapper for `Parallel.ForEach`

273 `Parallel.*` hits / 122 files, but VS-specific code uses its **own** `RoslynParallel` (4-6 hits per copy in `Microsoft.CodeAnalysis`, `Workspaces`, `MSBuild.BuildHost`).

What custom wrappers add over stock `Parallel.ForEach`:
- Cancellation propagation with `OperationCanceledException` → `TaskCanceledException`.
- Aggregate exception flattening.
- `JoinableTaskFactory` integration (does not block the UI thread).
- `IAsyncEnumerable` support (stock `Parallel` got `ForEachAsync` only in .NET 6).

**ML.NET** also has its own `ParallelUtilities`, `KISSParallel`, `ParallelAsync` (Azure Cosmos), `EnumerableParallelizationExtensions`.

## 29. `EventSource` (ETW) — one per component

`HostingEventSource`, `KestrelEventSource`, `MessagePackEventSource`, `HttpConnectionsEventSource` — every large Microsoft component has its own EventSource. Low cost (when there is no listener, `WriteEvent` is a no-op after the IsEnabled check), structured events via ETW/EventPipe.

```csharp
[EventSource(Name = "Microsoft-AspNetCore-Hosting")]
internal sealed class HostingEventSource : EventSource {
    public static readonly HostingEventSource Log = new HostingEventSource();
    [Event(1, Level = EventLevel.Informational)]
    public void HostStart() { if (IsEnabled()) WriteEvent(1); }
}
```

**Lesson**: for libraries `EventSource` > `ILogger.LogTrace` — it is zero-cost when nobody is listening, whereas `ILogger` always allocates an argv array.

## 30. `UnmanagedBufferAllocator` + `MemoryPoolBlock` (Kestrel)

Kestrel goes *below* the GC — it uses **unmanaged native memory** for HTTP buffers:
- `UnmanagedBufferAllocator` — allocates via `NativeMemory.Alloc`.
- `MemoryPoolBlock` — reusable blocks of native memory.

**Why**: GC does not scan it, does not compact it, no pinning needed. For a server handling millions of requests this is the difference between sporadic pauses and flat latency.

**For 99% of applications**: `ArrayPool<byte>.Shared` is sufficient. This technique is for high-throughput servers.

## 31. `MessagePack.UnsafeMemory32` / `UnsafeMemory64` — per-architecture selection

Two copies of the same code — one uses 32-bit pointer arithmetic, the other 64-bit:
```csharp
if (IntPtr.Size == 8) UnsafeMemory64.WriteRaw1(...); 
else UnsafeMemory32.WriteRaw1(...);
```

Each uses `Unsafe.WriteUnaligned` + `Unsafe.As<byte, X>`. That is 31 hits in a single file. **Brutal** unsafe code for maximum serialisation speed.

## 32. `[FieldOffset]` / `LayoutKind.Explicit` — union view

Heavily used for C-style unions:
- `MessagePack.GuidBits` — Guid as 16 bytes for quick encoding.
- `Microsoft.Azure.Cosmos.HybridRow.Float128`, `UnixDateTime`, `HybridRowHeader`, `MongoDbObjectId` — packed binary formats.
- `Microsoft.AspNetCore.HttpSys.Internal.HttpApiTypes` — interop with native http.sys.
- `VSPerfReader._LARGE_INTEGER`, `_EVENTBLOB`, `_SAMPLEEVENTBLOB` — VS profiler binary format.

```csharp
[StructLayout(LayoutKind.Explicit)]
internal struct GuidBits {
    [FieldOffset(0)] public Guid Value;
    [FieldOffset(0)] public ulong Low;
    [FieldOffset(8)] public ulong High;
}
```

Zero copies, atomic read of two ulongs instead of 16 bytes.

## 33. `ConditionalWeakTable<TKey, TValue>` — attaching data without retention

Lets you attach "extra" data to an object as if it were a field, but **does not keep the object alive**. Microsoft uses it for:
- `Microsoft.Cci.TrivialHashtableUsingWeakReferences`, `WeakValuesEnumerator` — FxCop analyzer state.
- `System.Management.Automation.WeakReferenceDictionary` — PowerShell engine.
- VS Debugger `WeakEventDelegate` — unsubscribable event handlers.

**Use case**: cache or state attached to a user-supplied object whose lifetime you do not control.

## 34. `StackObjectPool` (ASP.NET Components) — thread-local pool

Stack-based LIFO pool instead of a bag/queue. Idea: the most recently returned object is probably "warm" in the CPU cache.

```csharp
[ThreadStatic] private static Stack<T>? t_pool;
public T Rent() => (t_pool?.Count > 0) ? t_pool.Pop() : new T();
public void Return(T item) { (t_pool ??= new Stack<T>()).Push(item); }
```

Used in `RenderTreeBuilder` for elements touched many times per render. No contention (thread-local), cache-friendly.

## 35. `PerformanceSensitiveAttribute` — analyzer hint

Roslyn has its own attribute `[PerformanceSensitive(...)]` to mark methods where allocation would be harmful. An analyzer warns when this contract is violated.

```csharp
[PerformanceSensitive("https://...", AllowCaptures = false)]
private void HotPath() { ... }
```

**Lesson**: consider a custom analyzer for hot-path methods in your codebase — Roslyn shows this scales well.

---

## TL;DR Part 3 — what is new

| # | Pattern | Adoption |
|---|---------|----------|
| 22 | `[SkipLocalsInit]` assembly-wide | Every new Microsoft project |
| 23 | `[InlineArray(N)]` (.NET 8) | Fixed-size hashes, packets, UUID-like |
| 24 | `[ModuleInitializer]` | DI registration, native preload on assembly load |
| 25 | SIMD `Vector128`/`Avx2`/`Ssse3` | BCL + Kestrel string ops + ML tensors only |
| 26 | `ValueStringBuilder` (`ref struct`) | Instead of `new StringBuilder()` in hot paths |
| 27 | `AdaptiveCapacityDictionary` | Routing, headers — typically small N |
| 28 | Custom Parallel wrappers | Cancellation + JoinableTask integration |
| 29 | Per-component `EventSource` | Instead of `ILogger.LogTrace` in libraries |
| 30 | `UnmanagedBufferAllocator` | High-throughput servers (Kestrel) |
| 31 | 32-bit / 64-bit code fork | Brutal speed for unsafe serialisation |
| 32 | `[FieldOffset]` union | Binary protocols, interop, packed data |
| 33 | `ConditionalWeakTable` | Attach state to user objects |
| 34 | `[ThreadStatic] Stack<T>` pool | Cache-warm, no contention |
| 35 | `[PerformanceSensitive]` analyzer | Enforcing the no-alloc contract |

---

## Meta-observations from 1000 DLLs

1. **VS-specific code (`Microsoft.VisualStudio.*`) relies mainly on high-level patterns** — pooling, `JoinableTaskFactory`, `Lazy<T>`, immutable. Little SIMD, little unsafe.

2. **ASP.NET Core / Kestrel operates at a completely different level of aggression** — SIMD, unmanaged memory, custom PipelineSchedulers. Different constraints (server at 100K req/s vs an IDE).

3. **System.Private.CoreLib (BCL)** is the laboratory for SIMD and `[Intrinsic]`. You can see there how `IndexOf` achieves 1 cycle per 16 bytes.

4. **Roslyn is the richest source of general patterns** (`ObjectPool`, `ArrayBuilder`, `SegmentedArray`, `StringTable`, `ConcurrentLruCache`). You can copy them almost verbatim as templates.

5. **MessagePack and StreamJsonRpc** are the reference for **IPC perf patterns** — `IValueTaskSource`, `Pipe`, multiplexing.

6. **Every serious component has its own `EventSource`** — ETW is cheaper than logging in libraries.

7. **`SkipLocalsInit` is a hidden trick** — 1068 hits, nobody talks loudly about it, but Microsoft uses it everywhere.

8. **No `Channel<T>` even across 1000 DLLs** in VS-specific streaming code. This is truly a pipeline-first stack.

---

*Part 3 added 2026-05-15. 1024 DLLs in total decompiled to `C:\Temp\vsdecomp\` (~3.5 GB, 252,039 `.cs` files).*

---

# Part 4 — rounding out to 2064 DLLs (100% managed)

An additional 1040 DLLs: full WPF (`PresentationCore`, `PresentationFramework`, `WindowsBase`), `Microsoft.Extensions.DependencyInjection`, EntityFramework, F# Compiler Service, ML.NET Data, Xamarin, Azure SDK, BuildXL. They contribute a **new** class of patterns from domains other than VS-internal/Roslyn/ASP.NET.

## 36. WPF DependencyProperty — sparse storage struct (`EffectiveValueEntry`)

Source: `WindowsBase\System.Windows\EffectiveValueEntry.cs`, `DependencyObject.cs:69` hits

WPF must handle thousands of properties on UI elements, but a typical element sets only ~5-10. Instead of an `object[N]` array for all possible properties — a **sparse array of 12-byte structs**:

```csharp
internal struct EffectiveValueEntry {
    private object _value;
    private short _propertyIndex;        // 2 B — index, not a full reference to DependencyProperty
    private FullValueSource _source;     // 4 B bitfield (Local|Style|Inherited|Animated...)
}
```

- `short` (2 bytes) instead of a full `DependencyProperty` reference (8 B) — assumes < 32k DPs per type.
- `FullValueSource` bitfield holds 8 priorities (Local, Style, Trigger, Inherited...) in one int.
- The array is sorted by `_propertyIndex` → binary search.
- `DTypeMap` (PresentationCore) — fast type→handler dispatch.

**Pattern to copy**: when you have "an object with potentially hundreds of attributes but typically a handful" — a sparse `Entry[]` sorted by int index beats `Dictionary<Key, Value>` for small N.

## 37. WPF Freezable — opt-in immutability

Brushes, Geometries, and Animations in WPF inherit from `Freezable`. By default **mutable** (a developer can change a brush's colour), but after calling `.Freeze()`:
- It becomes **thread-safe** (requires no synchronisation).
- All change-notification subscriptions are discarded.
- It can be shared between threads / windows.

```csharp
var brush = new SolidColorBrush(Colors.Red);
brush.Freeze();  // now immutable, shareable, no notifications
```

**Perf benefit**: a single `Brushes.Red` (frozen) reused thousands of times. Without Freezable every UI element would need its own copy or would have to subscribe to change events.

`AbstractFreezable` is also visible in `ICSharpCode.Decompiler` (build → freeze → use forever).

## 38. `RBTree<T>` + `LiveShapingList` (WPF Data) — incremental sort

`MS.Internal.Data.RBTree` — Red-Black tree for view collections. `LiveShapingList` / `LiveShapingTree` updates filter/sort/group **incrementally** when the source changes, instead of from scratch.

**Pattern**: for an "always-up-to-date sorted view of a changing collection", a tree with O(log n) insert beats O(n log n) resort.

## 39. `Expression.Compile` — runtime IL emit cache

Dozens of libraries (CsvHelper, EntityFramework, MessagePack, F#, MVC ModelBinder, MAPI parsers, Azure SDK) **compile expressions once** and cache the delegate forever:

```csharp
private static readonly ConcurrentDictionary<Type, Func<object>> s_factories = new();
Func<object> factory = s_factories.GetOrAdd(t, type => {
    var ctor = type.GetConstructor(Type.EmptyTypes);
    return Expression.Lambda<Func<object>>(Expression.New(ctor)).Compile();
});
```

**Reflection called once per type, then a JIT'd delegate.** 10-100x faster than `Activator.CreateInstance(t)` on a hot path.

## 40. DI `ActivatorUtilities.CreateFactory` — pre-compile constructor

`Microsoft.Extensions.DependencyInjection.ActivatorUtilities`:
- `CreateFactory(Type, Type[])` — emits IL with the chosen constructor and `IServiceProvider` lookups.
- Returns a cached `ObjectFactory` delegate.

Instead of reflection per request: one compilation per type for the entire application lifetime.

`Microsoft.Extensions.DependencyInjection.ServiceLookup` (CallSite engine) builds the dependency graph as expressions, compiles them to delegates, and reuses them.

## 41. Runtime IL emit via `DynamicMethod` / `ILGenerator`

Where `Expression.Compile` is not sufficient (structs, ref params), Microsoft drops down to raw IL:

| Consumer | What it emits |
|----------|---------------|
| `MessagePack.DynamicObjectTypeBuilder` (96 hits) | Per-type serializer/deserializer (full IL) |
| `MessagePack.ILGeneratorExtensions` (81 hits) | DSL on ILGenerator for common operations |
| `EntityFramework.EntityProxyFactory` (9 hits) | Proxy classes for lazy-loading entities |
| `EntityFramework.IPocoImplementor` (113 hits) | Property getter/setter override |
| `F# Compiler.ILDynamicAssemblyWriter` (**262 hits**) | F# compiles entire assemblies in-memory |
| `dotnet-svcutil.CodeGenerator` (72 hits) | WCF service contract emission |
| `SignalR.TypedClientBuilder` (23 hits) | Strongly-typed hub proxies |
| `Roslyn.MessagePack` (in workspaces) | Code analysis serialization |

**Conclusion**: Runtime IL emit is not a niche — the largest .NET libraries use it. Pattern: **codegen at first-use, cache result, run forever**.

## 42. Segmented sort — `SegmentedArraySortHelper`

Since Roslyn has `SegmentedArray<T>` (jagged → bypasses LOH), `Array.Sort` won't work. Microsoft wrote its own `SegmentedGenericArraySortHelper` (a port of the classic introsort/heapsort from BCL `ArraySortHelper`) that operates on `T[][]`.

This file is seen in 7 different DLLs (`Microsoft.CodeAnalysis.Workspaces`, `MSBuild.BuildHost`, `Microsoft.VisualStudio.CoreUtility`, `Microsoft.Build.Framework`, `InteractiveHost`) — the code copied identically. **This is how Microsoft shares code**: copy-paste between assemblies instead of creating a dependency.

## 43. F# uses **persistent collections** at the source level

`FSharp.Core` uses its own immutable maps/sets implemented as persistent AVL trees (in `FSharp.Collections`). Every "modification" returns a new structure sharing >90% of memory with the old one.

This is a **different model** from System.Collections.Immutable (which uses a B-tree for `ImmutableSortedDictionary` but AVL for `ImmutableSortedSet`). Microsoft maintains BOTH libraries because F# has distinct constraints (functional purity, structural equality).

## 44. WPF `InsertionSortMap` — sorting small sets

`WindowsBase\MS.Utility\InsertionSortMap.cs` — insertion sort as a dedicated helper for maps/lists with N < ~16. WPF knows its domain: most collections are small → insertion sort is O(n²) but with a **small constant**, without quicksort's overhead.

**Pattern**: when you know N is typically <16, insertion sort beats the quicksort/Array.Sort overhead.

## 45. `[CallerArgumentExpression]` polyfill + assert helpers

Visible in all newer Microsoft projects. `ArgumentNullException.ThrowIfNull(x, nameof(x))` uses `[CallerArgumentExpression]` so the compiler supplies the argument text automatically.

`Microsoft.VisualStudio.Validation.Requires` + `Verify` have used this pattern for a long time — now BCL has adopted it.

---

## TL;DR Part 4

| # | Pattern | Origin |
|---|---------|--------|
| 36 | Sparse `Entry[]` struct for objects with hundreds of optional fields | WPF `DependencyObject` |
| 37 | Mutable → `.Freeze()` → immutable shareable | WPF Brushes/Animations |
| 38 | RB-tree for "live sorted view" of a changing collection | WPF `LiveShapingList` |
| 39 | `Expression.Compile` + cache per type | DI, EF, MessagePack, ASP.NET MVC |
| 40 | `ActivatorUtilities.CreateFactory` | Microsoft.Extensions.DI |
| 41 | `DynamicMethod` + `ILGenerator` with cache | MessagePack, EF proxy, F# compiler |
| 42 | Custom sort on segmented arrays | Roslyn (copied 7×) |
| 43 | Persistent AVL trees instead of immutable B-tree | F#.Core |
| 44 | Insertion sort for N<16 | WPF |
| 45 | `[CallerArgumentExpression]` in guards | Everywhere in new code |

---

## Final meta-observations (after 2064 DLLs)

1. **Three "schools" of perf at Microsoft**:
   - **VS-internal** (Roslyn, VS Threading) — pooling, immutable, async-first, no SIMD/unsafe
   - **High-throughput server** (Kestrel, ASP.NET Core) — SIMD, unmanaged memory, custom awaiters
   - **BCL** (`System.Private.CoreLib`) — the `[Intrinsic]`, AVX/SSE, RyuJIT-aware code laboratory

2. **WPF is a world of its own** — it was developed before `Span<T>`/`ArrayPool` existed. Its patterns (sparse storage, freezable, RBTree) are still excellent models for **other** problems beyond UI.

3. **Codegen is the standard** — `Expression.Compile`/`DynamicMethod` everywhere reflection speed matters. Roslyn analyzers actively enforce this technique (`[PerformanceSensitive]`).

4. **Microsoft prefers copy-paste over shared dependency** for low-level utilities (`SegmentedArraySortHelper`, `SkipLocalsInitAttribute`, `ValueStringBuilder`). Larger binary, but no version conflicts.

5. **F# and C# stacks are separate** — F# does not use Roslyn or BCL collections; it has a parallel compiler implementation + persistent collections.

6. **Pooling > SIMD in terms of ROI** for IDE/business applications. SIMD pays off when you are parsing millions of bytes/s.

---

*Part 4 added 2026-05-15. **2064 managed DLLs decompiled** in total in `C:\Temp\vsdecomp\` (~5 GB, 370,860 `.cs` files). 100% coverage of managed Microsoft+System+3rd-party assemblies from the VS 2022 install (>=32 KB, not filtered as native runtime stubs).*

---

# Part 5 — cross-reference with dotnet-optimization-cheatsheet (Nikou Usalp)

Source: <https://github.com/nikouu/dotnet-optimization-cheatsheet> — 40 perf techniques collected from blogs (Stephen Toub, NDepend, Adam Sitnik et al.).

## What the cheatsheet confirms from my scan (cross-validation)

Practically 1:1 with my findings — `ArrayPool`, `ObjectPool`, `ValueTask`, `Span<T>`, `stackalloc`, `[MethodImpl(AggressiveInlining)]`, `[SkipLocalsInit]`, `[InlineArray]`, `MemoryMarshal`, `Unsafe.*`, custom struct awaiters, SIMD. What Microsoft writes about in blogs, **it actually does in production code**. Very few companies work this way.

## 46. What the cheatsheet adds (new vs my scan)

### A. Techniques missing from my analysis

**`[SuppressGCTransition]` on P/Invoke** (#36 cheatsheet) — attribute on `[DllImport]` that skips switching the thread to "preemptive GC mode" for the duration of a native call. Saves ~10 ns per call. Only for **very fast** native calls (such as `GetTickCount`) — for long blocking calls this will crash (GC stuck).

**`GC.TryStartNoGCRegion(size)` + `GC.EndNoGCRegion()`** (#37) — disables GC in a critical section. Used in real-time / low-latency scenarios (HFT, audio processing). I did not look for this in my scan — I checked now and it does appear in a few components.

**`CollectionsMarshal.AsSpan(List<T>)`** (#15) — returns a `Span<T>` over the list's internal array **without copying**. Index-based iteration over the span beats `foreach` over `List<T>`. Be careful: the list must not be modified during iteration.

**`string.Create<TState>(int length, TState state, SpanAction<char, TState>)`** (#5) — builds a string of known length with a single `Span`-write, without `StringBuilder` or intermediate buffers. Ideal for formatting (UUID, hex, timestamps).

**`MemoryMarshal.GetArrayDataReference()` + `Unsafe.IsAddressLessThan()`** (#31) — "fastest loops" — iteration via pointer arithmetic without bounds checks. Used in BCL (`Ascii.cs`, `Utf8Utility.cs` — my scan saw these files but I did not name the technique).

**`[UnscopedRef]`** (#28) — allows a ref-field in a struct to "escape" beyond a method's scope when the compiler would normally block this. Required for some advanced ref-struct patterns (C# 11+).

**Default `GetHashCode`/`Equals` for `struct` uses reflection** (#29) — one of the least-known gotchas. If you use a struct as a `Dictionary` key, you **must** override `Equals`/`GetHashCode` manually or you will get a reflection-based slow path. For `record struct` the compiler generates them automatically.

**Static lambdas** (#32) — `Dict.GetOrAdd(key, static k => Create(k))` instead of `Dict.GetOrAdd(key, k => Create(k))`. The `static` keyword forces no closure capture → zero delegate allocation. Used implicitly in my scan but worth naming explicitly.

### B. Techniques outside the scan (config / build-level, not visible in IL)

- **Server GC** — `<ServerGarbageCollection>true</ServerGarbageCollection>` in csproj. Multi-threaded mark/sweep, better for servers (worse for desktop). VS runs in Workstation GC mode.
- **Native AOT** — `<PublishAot>true</PublishAot>`. Native compilation, no JIT, no reflection runtime. Very fast startup, small binary. Trade-off: dynamic code (Expression.Compile, EF) is not available.

### C. Domain-specific tricks from the cheatsheet

- **`HttpClient.GetStreamAsync` / `GetFromJsonAsync<T>`** instead of `GetAsync().ReadAsStringAsync()` — skips an intermediate buffer.
- **`EF.CompileQuery(...)`** — pre-compiles a LINQ query to a delegate. Microsoft Extensions DI uses the same pattern (Part 4 #40).
- **`IsSuccessStatusCode` instead of try/catch on HttpResponse** — throwing an exception costs 10-100 μs.
- **Do not use `int?`/`bool?` in a hot path** — unboxing + null check per access.
- **`String.Compare(a, b, StringComparison.OrdinalIgnoreCase)`** instead of `a.ToLower() == b.ToLower()` — no allocations.
- **`RecyclableMemoryStream`** (`Microsoft.IO.RecyclableMemoryStream` NuGet) — pooled `MemoryStream` with chunked resizing. A replacement for `new MemoryStream()` in serialisation hot paths.

### D. `ReadOnlySequence<T>.Slice` is slow (#40)

A cheatsheet surprise: `ReadOnlySequence<T>.Slice` has overhead. It is better to take `.First.Span` if the first segment is sufficient. This affects all code that parses Pipelines.

## What MY scan adds that is not in the cheatsheet

This matters — the cheatsheet collects "general" techniques but **does not show architectures**. My scan revealed **structural patterns**:

- **`SegmentedArray<T>` / `SegmentedDictionary` / `SegmentedHashSet`** (Roslyn, copied 7×) — avoiding the LOH via jagged arrays.
- **Two-tier hash table** (Roslyn `StringTable` — local + shared).
- **Sparse struct storage** (WPF `EffectiveValueEntry`).
- **Frame-multiplexing** (Nerdbank `MultiplexingStream`).
- **RPC correlation-ID cancel propagation** (StreamJsonRpc).
- **Per-shape frozen dictionaries** (BCL `OrdinalStringFrozenDictionary_LeftJustifiedSingleChar` etc.).
- **`Freezable` opt-in immutability** (WPF).
- **Custom LRU with `lock`** (not `ConcurrentDictionary`, because eviction requires an atomic op on 2 structures).
- **`ValueStringBuilder` polyfill in hundreds of projects**.
- **`UnsafeMemory32` / `UnsafeMemory64` dual code path** (MessagePack).
- **`AdaptiveCapacityDictionary`** (array for N≤10, then promotion).
- **Hot/Slow method split** (`Allocate` / `AllocateSlow`).
- **`[ThreadStatic]` `Stack<T>` pool** (ASP.NET RenderTree).
- **`PerformanceSensitive` analyzer attribute** in Roslyn.
- **Per-component `EventSource`** instead of `ILogger`.
- **F# `ILDynamicAssemblyWriter`** (262 emit calls) — a parallel compiler stack alongside Roslyn.

The cheatsheet is a **list of tools**. My scan is a **list of architectures**. Together they form the complete picture.

---

## TL;DR Part 5 — what to add to the toolbox

| # | Technique | What it adds |
|---|-----------|-------------|
| 46a | `[SuppressGCTransition]` on P/Invoke | ~10 ns/call for quick native calls |
| 46b | `GC.TryStartNoGCRegion` | No-GC critical sections (real-time) |
| 46c | `CollectionsMarshal.AsSpan(list)` | Span view without copying List<T> |
| 46d | `string.Create<TState>` | Known-length string without intermediate buffer |
| 46e | `MemoryMarshal.GetArrayDataReference` + `Unsafe.IsAddressLessThan` | Fastest loops (pointer arith, no bounds check) |
| 46f | `[UnscopedRef]` | Ref fields in a struct can "escape" |
| 46g | Override `GetHashCode`/`Equals` for struct keys | Because default = reflection |
| 46h | `static` lambdas | Zero closure allocation |
| 46i | Server GC config | Multi-threaded apps, not desktop |
| 46j | Native AOT | Startup + binary size (at the cost of reflection) |
| 46k | `HttpClient.GetStreamAsync` / `GetFromJsonAsync<T>` | No intermediate string |
| 46l | `EF.CompileQuery` | Pre-compiled LINQ |
| 46m | `IsSuccessStatusCode` vs throw | Exception = 10-100 μs |
| 46n | Avoid `int?`/`bool?` in hot path | Boxing + null check overhead |
| 46o | `String.Compare(..., StringComparison.X)` | No ToLower() allocation |
| 46p | `RecyclableMemoryStream` | Pooled MemoryStream |
| 46q | `ReadOnlySequence<T>.Slice` is slow | Use `.First.Span` |

---

*Part 5 added 2026-05-15. **63 patterns** in total (45 from my scan + 17 new from the cheatsheet + 1 cross-reference). The file is becoming a reference card for .NET perf.*

---

# Part 6 — cross-reference with awesome-dot-net-performance (Adam Sitnik)

Source: <https://github.com/adamsitnik/awesome-dot-net-performance> — a curated list of tools, talks, and books from Adam Sitnik (.NET runtime team, author of `Span<T>`/`ArrayPool` perf improvements, BenchmarkDotNet contributor).

This is a **different axis** from my analysis and Nikou's cheatsheet: not code patterns, but **meta-knowledge** — tools, people, JIT concepts that you **cannot see** in decompiled IL.

## 47. JIT-level mechanisms (invisible in IL, critical for perf)

These are techniques that happen **at execution time**, not in the code itself:

### Tiered Compilation (.NET 3.0+, on by default)

JIT compiles a method twice:
- **Tier 0** (at startup) — fast compilation, simple code, no optimisations. Goal: load as quickly as possible.
- **Tier 1** — when a method becomes "hot" (after several calls), JIT recompiles it with full optimisations (inlining, unrolling, etc.).

**Consequence**: first calls are always slower. A benchmark must warm up, otherwise it measures tier-0.

### Dynamic PGO (.NET 8+, default-on since .NET 9)

In tier 0 the JIT **instruments the code** — counting which `if` branch is hit more often, which types actually appear in `virtual call`s. Tier 1 uses those statistics:
- Frequent branch → fast path inlined.
- Dominant type → **devirtualization + guarded inlining** (instead of `callvirt` you get `if (obj.Type == typeof(X)) X.Method() else callvirt`).

**Impact**: 5-15% perf improvement in real applications **without changing code**. Simply update the runtime.

### On-Stack Replacement (OSR, .NET 7+)

Problem: long loops started once in tier 0 will never be "promoted" — the loop does not end, so nobody ever calls the tier-1 version. **OSR replaces the method code WHILE IT IS EXECUTING** — the stack frame is rewritten on the fly from tier-0 to tier-1.

**Without OSR**: `for (int i = 0; i < 10_000_000; i++)` always runs in tier-0.
**With OSR**: after a few seconds the method is "promoted" mid-loop.

### Escape Analysis (.NET 9+, partial)

JIT analyses whether `new MyClass()` actually "escapes" the method. If not → **allocation on the stack instead of the heap**. For small, short-lived objects this is GC-free.

Still in development — currently works mainly for `Span<T>` and value-typed result enumerables.

### ReadyToRun (R2R)

AOT-like compilation at the assembly level. Generates native code **in addition** to IL; JIT can use it immediately instead of compiling. BCL has been R2R'd since .NET 6 → faster startup.

**Conclusion**: when measuring perf on .NET 8+ vs .NET 6, 10-30% of the difference comes **from the JIT, not your code**. Update the runtime before optimising.

## 48. Pitfalls you must know (not covered in perf "tips" blogs)

### `Finalize` ~~Dispose~~ is a GC tax

Every object with a finalizer (`~MyClass()`):
- Allocates more slowly (registration in the finalisation queue).
- Lives for **at least 2 GC generations** (Gen 0 → finalisation queue → finalizer thread → Gen 1+).
- Blocks cleanup until the finalizer runs.

**Pattern**: implement `IDisposable` instead of a finalizer. Use a finalizer only as a safety net for unmanaged resources, with `GC.SuppressFinalize(this)` in Dispose.

### `string.Intern` has lock contention

The global interned-strings table uses a lock. Under load = bottleneck. Roslyn abandons `string.Intern` in favour of its own `StringTable` (my scan #4) **for exactly this reason**.

### `Dictionary<K,V>` read-only **is not** thread-safe

Counter-intuitively: even when you are "only reading", `Dictionary.TryGetValue` can race-condition if another thread even **resizes** the structure. Under load = corruption.

**Pattern**: use `FrozenDictionary` (my scan #6), `ImmutableDictionary`, or `ConcurrentDictionary`. Reading a plain `Dictionary` from multiple threads = UB.

### `ReaderWriterLock` vs `ReaderWriterLockSlim`

`ReaderWriterLock` is an **old** type (.NET 1.1), kernel-mode locks, slow. **Never use it** — `ReaderWriterLockSlim` is user-mode, ~10× faster. Unfortunately both names appear in IntelliSense.

## 49. Tooling — what to actually use

| Tool | When |
|------|------|
| **BenchmarkDotNet** | Every micro-benchmark. Statistical analysis, warmup, GC pressure metrics. Industry standard. |
| **PerfView** | CPU + heap + ETW snapshot for Windows. Steep learning curve, but nothing better exists for GC analysis. |
| **dotnet-trace** | Cross-platform (Linux/macOS) alternative to PerfView via EventPipe. |
| **dotMemory** (JetBrains) | Heap snapshot comparison, retention graphs. UI-based. |
| **dotTrace** (JetBrains) | Sampling and tracing profiler. Production-safe sampling. |
| **ultra** (Alexandre Mutel) | New (2023+) sampling profiler for Windows. Zero-config, ETW under the hood. |
| **Visual Studio Profiler** | Integrated. Works, but weaker than dotTrace/PerfView. |
| **ClrMD** | A library, NOT a tool. Lets you write your own debugger / heap analyser. |
| **Clr Heap Allocation Analyzer** | Roslyn analyser — warns about hidden allocations (`foreach` over `IEnumerable`, boxing, closures) **while writing code**. |

**Rule of thumb**: start with **BenchmarkDotNet** (micro) + **dotTrace or PerfView** (macro snapshot). The rest is specialisation.

## 50. People to follow

| Person | Why |
|--------|-----|
| **Stephen Toub** (Microsoft) | Annual `"Performance Improvements in .NET X"` blog posts — the best summary of runtime changes. Essential reading. |
| **Maoni Stephens** (Microsoft) | GC internals. If you want to understand WHY your profile shows pauses. |
| **Andrey Akinshin** | Author of BenchmarkDotNet. Book: "Pro .NET Benchmarking". |
| **Adam Sitnik** | `Span<T>`, `ArrayPool`, allocations. |
| **Konrad Kokosa** | "Pro .NET Memory Management" (2024) — the most in-depth book on GC. |
| **Tanner Gooding** | Hardware intrinsics, SIMD. |
| **Egor Bogatov** | JIT contributions, recently dynamic PGO. |
| **Matt Warren** | Old blog (mattwarren.org), still valuable deep dives. |

## 51. Books — a short list

- **Pro .NET Memory Management** (Kokosa, 2024) — the *only* current book on GC. ~700 pages, masterclass.
- **Writing High-Performance .NET Code** (Watson, 2018) — old but fundamentals still apply. Short and readable.
- **Pro .NET Benchmarking** (Akinshin, 2019) — measurement methodology, not just BenchmarkDotNet.
- **CLR via C#** (Richter, 2012) — CLR fundamentals. Old, but the threading/memory chapters still hold.

## 52. EventPipe — cross-platform ETW

Linux/macOS do not have ETW (Windows-only). `EventPipe` is a cross-platform rewrite of that concept. `dotnet-trace` uses EventPipe under the hood.

**Consequence**: `EventSource` (my scan #29) works everywhere — on Windows it goes to ETW, on Linux to EventPipe → the same structured telemetry.

## 53. ARM64 — the second first-class target

Microsoft is investing heavily in ARM64 (.NET 8+):
- ARM64 has a **weaker memory model** than x64 → more memory barriers in generated code. This has a cost.
- JIT does **ARM64-specific peephole optimisations** (special instructions: `MADD`, `MSUB`).
- SIMD on ARM64 is `AdvSimd` (`System.Runtime.Intrinsics.Arm`), a different namespace from x86 (`Sse2`/`Avx2`).

**In my scan**: I saw `AdvSimd.IsSupported` checks in BCL collections, but `Microsoft.VisualStudio.*` has nothing for ARM. VS is x64-first.

---

## What Adam Sitnik adds that is absent from both my scan and Nikou's cheatsheet

This is the third axis of analysis:

1. **JIT runtime mechanics** — Tiered, PGO, OSR, Escape Analysis, R2R. **Highly impactful on perf without changing code**.
2. **Pitfalls of specific BCL APIs** — `string.Intern`, finalizers, `ReaderWriterLock`, Dictionary thread-safety.
3. **Tooling ranking** — what to actually use, not just what exists.
4. **Conferences / talks** — living knowledge in recordings that never make it to blogs.

---

## TL;DR Part 6

| # | Concept | Where it happens |
|---|---------|-----------------|
| 47a | Tiered Compilation | JIT, automatic |
| 47b | Dynamic PGO | JIT, .NET 8+ default |
| 47c | OSR | JIT, mid-execution |
| 47d | Escape Analysis | JIT, .NET 9+ partial |
| 47e | ReadyToRun | Build time + JIT |
| 48a | Finalizer = GC tax | Your code |
| 48b | `string.Intern` lock contention | BCL |
| 48c | `Dictionary` read-only != thread-safe | BCL |
| 48d | `ReaderWriterLock` is obsolete | BCL |
| 49 | BenchmarkDotNet + dotTrace/PerfView | Tooling |
| 50 | Stephen Toub blog (annual) | Resource |
| 52 | EventPipe = cross-platform ETW | Runtime infra |
| 53 | ARM64 as an equal-first target | Runtime |

---

## Final summary — 3 axes of analysis

The file now integrates **three complementary perspectives** on .NET perf:

1. **My scan of 2064 DLLs** — *how Microsoft actually writes production code* (architectures, structural patterns).
2. **Nikou's cheatsheet** — *toolbox of techniques* to apply in your own code (Span, ArrayPool, SkipLocalsInit etc.).
3. **Adam's awesome list** — *meta-knowledge* (JIT mechanics, tooling, people, books).

Without any of these three you don't have the full picture:
- Code alone → you don't know **what PGO will do to your code**.
- Toolbox alone → you don't know **which techniques Microsoft actually uses** (vs only writes about in blogs).
- Meta-knowledge alone → you don't know **what it looks like in 5 GB of real IL**.

---

*Part 6 added 2026-05-15. **70+ concepts** organised in 6 parts in total. A complete .NET perf reference card grounded in decompilation of 2064 DLLs + two curated lists.*

---

# Part 7 — cross-reference with Konrad Kokosa's slides (Pro .NET Memory Management)

Source: <https://prodotnetmemory.com/slides/PerformancePatterns/> — slides from Konrad Kokosa (author of *Pro .NET Memory Management* 2024, ~700 pages). This is the **third axis** — after code patterns (my scan) and the technique toolbox (cheatsheets) — **principles and memory-layout patterns** with a focus on CPU cache.

## 54. Frugal Object Pattern — discriminated union for "0 / 1 / many"

This is a pattern I **missed in the scan** even though it was right in front of me.

**Problem**: an HTTP header value is typically one string (`"application/json"`), sometimes two (`"gzip, deflate"`), almost never more. A naive `string[] values` allocates an array for every header — unnecessarily.

**Solution**: `Microsoft.Extensions.Primitives.StringValues` (ASP.NET Core):

```csharp
public readonly struct StringValues {
    private readonly object _values;       // null | string | string[]
    // null → empty
    // string → single value (no array alloc)
    // string[] → multiple values
}
```

One field, three states. For 90% of headers (single-value) zero array allocation.

**Other examples**:
- `WPF.FrugalList<T>` — the same for collections of dependency properties (typically 1-3 elements).
- `Optional<T>` in many APIs.

**Pattern to copy**: when the domain says "typically 1, sometimes a few, rarely many" — don't use `List<T>` as the default. A discriminated union in a single field.

## 55. Struct of Arrays (SOA) — Data-Oriented Design

**A classic from game dev, rare in business code.**

Instead of:
```csharp
class Particle { float X, Y, VX, VY, R, G, B, A; }
Particle[] particles;
```

You do:
```csharp
float[] xs, ys, vxs, vys, rs, gs, bs, as_;
```

**Why**: the "update position" loop reads only `xs, ys, vxs, vys`. With AoS (Array of Structs) each cache line pulls the entire Particle (~32 B), half of which is unused. With SOA a cache line of `xs[]` holds **16 X values** tightly packed; the prefetcher tracks linear access.

**Konrad shows a 10× speedup** in a benchmark. Also SIMD-able (`Avx2` can add 8 floats at a time).

**Trade-off**: breaks OOP encapsulation. A Particle as a conceptual unit is scattered across 8 arrays. For **hot numerical loops** — invaluable. For business CRUD — an anti-pattern.

**In my scan**: System.Numerics.Tensors uses SOA internally. ML.NET as well. The rest of Microsoft's code — no.

## 56. Konrad's six "Performance Principles" — a thinking framework

Instead of a "list of tricks" — a meta-framework:

| # | Principle | Practical consequence |
|---|-----------|----------------------|
| **0** | Memory/CPU discrepancy | RAM is 100x slower than the CPU. Every cache miss = ~100 stalled cycles. |
| **1** | Fit cache line (64 B) | Hot struct <= 64 bytes. Otherwise two cache-line fetches per access. |
| **2** | Fit highest cache level | Working set <= L1 (~32 KB) -> fastest. <= L2 (~256 KB) -> fast. <= L3 (~8 MB) -> ok. Above -> DRAM, painful. |
| **3** | Design for sequential access | Linear access = prefetcher guesses ahead, no misses. Random access (linked list, hash table) = miss on every element. |
| **4** | Avoid GC overhead | Pooling, struct, stackalloc. Everything from parts 1-6. |
| **5** | Avoid calls | Virtual call = pipeline stall (branch prediction). Interface call = double indirection. Inline hot path or use `sealed`. |
| **6** | Avoid false sharing | Two cores writing different fields in the same cache line -> cache coherence storm. |

**This framework runs through all of Microsoft's patterns**:
- `ObjectPool<T>` (my #1) = #4.
- `EffectiveValueEntry` sparse struct (#36) = #1, #2 (less memory -> more fits in cache).
- `SegmentedArray` (#5) = #2 (avoids LOH, more fits in L3).
- `[ThreadStatic] Stack<T>` (#34) = #6 (no false sharing because thread-local).

## 57. False sharing — the pattern few people remember

Two cores modify different fields in the same cache line (64 B). Every write on core A invalidates the copy on core B -> constant cache misses. Can be **slower than a lock** despite being "lock-free".

**Pathological example**:
```csharp
class Counter { 
    public long ThreadACount;  // offset 0
    public long ThreadBCount;  // offset 8 — SAME cache line!
}
```

**Fix — padding**:
```csharp
[StructLayout(LayoutKind.Explicit, Size = 64)]
struct PaddedCounter { [FieldOffset(0)] public long Value; }
```

In BCL: `System.Threading.PaddedReference<T>`. In my scan I saw padding in `ConcurrentQueueSegment.cs` (BCL) — now I know why.

## 58. `Span<T>` + `async` is impossible (and why)

`Span<T>` is a **`ref struct`** — it can only live on the stack. It cannot be packed into a field of an object (and async methods are compiled to a state machine **class** that stores local variables).

**Workaround**: `Memory<T>` + `.Span` property right before use:

```csharp
async Task ProcessAsync(Memory<byte> data) {
    await SomethingAsync();
    var span = data.Span;   // only now do I get the Span
    DoStuff(span);
}
```

This is why `PipeReader.ReadAsync()` returns `ValueTask<ReadResult>` with `Buffer` as `ReadOnlySequence<byte>` (a sequence of `Memory<byte>`), not `ReadOnlySpan<byte>`.

## 59. `stackalloc` has a hidden cost — zeroing

`stackalloc byte[256]` **zeroes 256 bytes** before giving them to you. For small buffers this is nothing; for large ones in a hot loop — measurable.

**Fix**: `[SkipLocalsInit]` on the method (my #22) — eliminates zeroing. **Critical**: after this you MUST write to the buffer before reading (UB otherwise).

**Polyfill for .NET < 5**: Fody plugin `[LocalsInit(false)]` — IL-rewriting that disables the init flag in the generated assembly.

## 60. Other smaller observations from Konrad

- **`StackOverflowException` is uncatchable** — the process dies. `stackalloc` in a recursive loop = risk.
- **`SlabMemoryPool`** (Kestrel) — pools not arrays but **slabs** (larger 4 KB blocks) sliced into chunks. Less book-keeping than `ArrayPool` for small allocations.
- **`ArrayPool<T>.Shared` has per-core stacks** — 17 size buckets, max 50 arrays per bucket. Trimming happens automatically under GC pressure.
- **LLC (Last-Level Cache) miss rate** as a benchmark metric — not just CPU%/allocs. PerfView and `dotnet-counters` show this.

---

## TL;DR Part 7 — what Konrad adds

| # | Pattern / Concept | What it adds |
|---|-------------------|-------------|
| 54 | Frugal Object (`StringValues`, `FrugalList`) | Discriminated union for "0/1/many" — zero allocation in the common case |
| 55 | Struct of Arrays (SOA) | Cache locality + SIMD-friendly for hot loops |
| 56 | Six Performance Principles | Thinking framework: cache > GC > alloc > calls |
| 57 | False sharing + padding | Lock-free can be slower than a lock without padding |
| 58 | `Span<T>` + `async` impossible | `ref struct` cannot enter a state machine — use `Memory<T>` |
| 59 | `stackalloc` zeroes (cost) | `[SkipLocalsInit]` + Fody polyfill for older .NET |
| 60a | `SlabMemoryPool` | Pool of 4 KB slabs sliced into chunks (Kestrel) |
| 60b | LLC miss rate as a metric | Beyond CPU% — shows whether the pattern is working |

---

## Final final summary — 4 axes

The file now integrates **four complementary perspectives**:

1. **My scan of 2064 DLLs** — *concrete production architectures*. What Microsoft does (sparse storage, frame multiplexing, RPC cancel).
2. **Nikou's cheatsheet** — *toolbox of coding techniques*. What to use in your own code (Span, SkipLocalsInit).
3. **Adam's awesome list** — *meta-knowledge and tooling*. What happens in the runtime (PGO, OSR), what to measure with.
4. **Konrad's slides** — *principles and memory layout*. Why it matters (cache lines, false sharing, sequential access).

Each axis without the others is incomplete:
- Without **#1** -> you don't know what Microsoft actually does (vs what it writes about in blogs).
- Without **#2** -> you don't have a concrete toolbox.
- Without **#3** -> you don't know **what to measure with** or **how the runtime behaves under the hood**.
- Without **#4** -> you don't know **why** — patterns become cargo-cult.

---

*Part 7 added 2026-05-15. **~75 concepts** in 7 parts in total. The file has reached the state of a .NET perf reference card — from JIT mechanics through tooling to concrete production architectures.*

---

# Part 8 — two patterns from ML.NET (`dotnet/machinelearning`)

Source: code review of `Microsoft.ML.*` (`src/`) — a numeric library that processes millions of rows/vectors. This is a **different constraint** from VS-internal, ASP.NET, or BCL: not an IDE, not a server at 100K req/s, but a **high-throughput numeric pipeline**. Two structural patterns not covered in parts 1-7.

## 61. Mutable editor over an immutable buffer — `immutable struct` + `ref struct` editor + `Commit()`

Source: `Microsoft.ML.DataView/VBuffer.cs`, `VBufferEditor.cs`

`ValueStringBuilder` (#26) reuses its internal buffer, but is *itself* a mutable `ref struct`. ML.NET goes one step further: **the public type is an immutable `readonly struct`**, and mutation goes through a **separate `readonly ref struct` editor** that takes ownership of the arrays, edits them through `Span`, and reassembles the result via `Commit()`.

```csharp
// 1. Public type — readonly struct, immutable, holds reusable arrays
public readonly struct VBuffer<T> {
    private readonly T[]   _values;    // re-usable
    private readonly int[] _indices;   // null/empty => dense; otherwise sparse (parallel to _values)
    private readonly int   _count;
    public readonly int    Length;
    public ReadOnlySpan<T>   GetValues()  => _values.AsSpan(0, _count);
    public ReadOnlySpan<int> GetIndices() => IsDense ? default : _indices.AsSpan(0, _count);
}

// 2. Editor — readonly ref struct (stack-only, won't escape, won't box)
public readonly ref struct VBufferEditor<T> {
    public readonly Span<T>   Values;
    public readonly Span<int> Indices;
    public bool CreatedNewValues { get; }   // did growth force a new allocation?
    public VBuffer<T> Commit()              => new VBuffer<T>(_logicalLength, Values.Length, _values, _indices);
    public VBuffer<T> CommitTruncated(int physicalCount); // logical truncation without reallocation
}

// 3. Entry via `scoped ref` — signals to JIT/compiler that the ref does not escape
var editor = VBufferEditor.Create(ref dst, length, keepOldOnResize: true);
for (int i = 0; i < length; i++) editor.Values[i] = ...;
dst = editor.Commit();   // dst is immutable again, same array
```

**Key design decisions:**
- **The editor *takes ownership* of the arrays from the old buffer.** The code documentation states explicitly: *"the resulting VBufferEditor is assumed to take ownership of this passed-in object... its underlying buffers are being potentially reused"*. The old `VBuffer` is invalid after being passed.
- **`keepOldOnResize`** — flag controls `Array.Resize` (keep data) vs `new T[]` (discard). Same as `keepOld` in Roslyn's `EnsureSize`.
- **`CreatedNewValues`** — the editor tells the caller whether a reallocation occurred, so the caller can recompute cached spans. Hot path: when `false`, you can densify/shift *in-place*.
- **`CommitTruncated`** — the caller allocates conservatively with excess capacity (when the final size is unknown), then truncates **logically** (changes `_count`), without reallocating the array.

**Why:** separating the immutable API from the mutable mechanism gives simultaneously (a) a safe, non-state-sharing public type, (b) zero-alloc reuse in a loop over millions of elements. The `ref struct` on the editor guarantees a stack-only lifetime — the editor will not escape to a field, will not box, will not outlive its scope. This generalises `ValueStringBuilder` to any "rebuilder" of a collection — including a two-array one (sparse: values + indices).

**How to apply:** for any collection/vector type built incrementally or reused in a hot loop — do not make the public type mutable. Create a triple: `readonly struct` (immutable API) + `readonly ref struct` Editor (mutation via `Span`) + `Commit()`. Enter the editor via `scoped ref` on the old buffer. Connects with [[dotnet-perf-reference]] #26 (ValueStringBuilder) and #5 (segmented arrays — if the buffer can exceed the LOH).

## 62. `ValueGetter<T>` — "push-into-caller-ref" delegate instead of `T Get()`

Source: `Microsoft.ML.DataView/IDataView.cs` (`delegate void ValueGetter<TValue>(ref TValue value);`), `IValueMapper.cs` (`delegate void ValueMapper<TSrc,TDst>(in TSrc src, ref TDst dst);`)

The classic `TValue Get()` contract on a stream of values of the same shape **allocates or copies on every call** — reference type = new object, large struct = defensive copy, `VBuffer` = new array. ML.NET inverts the contract:

```csharp
// NOT: TValue Get();           — every call produces a value
// BUT:
public delegate void ValueGetter<TValue>(ref TValue value);          // write into caller's buffer
internal delegate void ValueMapper<TSrc,TDst>(in TSrc src, ref TDst dst); // transform: in + ref

// Usage — caller allocates the buffer ONCE; getter reuses it through the entire stream:
ValueGetter<VBuffer<float>> getter = cursor.GetGetter<VBuffer<float>>(column); // fetched once, before the loop
VBuffer<float> buf = default;                  // one allocation for the entire run
while (cursor.MoveNext())
    getter(ref buf);                           // 0 allocations per row — array inside buf reused
```

**Why:** the value is not *returned* — the getter **writes it into the caller's buffer**. The caller owns the lifetime and reuses the same `buf` (and internally: the same array — see #61) for millions of iterations. `in TSrc` (readonly ref) on the mapper input eliminates the defensive copy of a large struct; `ref TDst` on the output eliminates result allocation. The delegate is fetched **once before the loop** — the cost of resolving column/type is amortised.

**How to apply:** when you have a stream of values with a fixed shape (rows, samples, frames) — instead of `T Get()`, provide `void Get(ref T)`. The caller declares the buffer outside the loop and passes it via `ref`. For transformations: `void Map(in TSrc, ref TDst)`. Acts as a zero-alloc equivalent of `IEnumerator<T>.Current` — connects with the cursor pattern (`long` position + `MoveNextCore`) and with [[dotnet-perf-reference]] #61 (the getter receives a `ref VBuffer` and edits it through the editor).

---

## TL;DR Part 8

| # | Pattern | When |
|---|---------|------|
| 61 | `immutable struct` + `ref struct` editor + `Commit()` | Any collection/vector type reused or built incrementally in a hot path |
| 62 | `void Get(ref T)` delegate instead of `T Get()` | Stream of values with a fixed shape — caller reuses the buffer via `ref` |

**What NOT to copy from ML.NET:** `AlignedArray` with manual realignment (since .NET 6 there is `GC.AllocateArray(pinned: true)` + `NativeMemory.AlignedAlloc`); `unsafe char*` in `StringSpanOrdinalKey` (on .NET 9+ there is `Dictionary.GetAlternateLookup<ReadOnlySpan<char>>`); the custom `BufferPoolManager` pool outside the LOH range (`ArrayPool<T>.Shared` is better for < 85 KB). The SIMD architecture of `Microsoft.ML.CpuMath` (three-tier AVX/SSE/scalar dispatch, vector-width cascade, alignment mask tables, FMA dispatch, tree-based reduction) is the third — after BCL and Kestrel (#25) — exemplary illustration of "SIMD + scalar fallback" organisation, worth a separate read if you do arithmetic on arrays.

---

*Part 8 added 2026-05-16. Cross-reference with `dotnet/machinelearning`. **~77 concepts** in 8 parts in total. ML.NET as the fourth "school" of perf — high-throughput numeric pipeline: aggressive buffer reuse (editor pattern), push-ref contracts (`ValueGetter`), SIMD with a full cascade.*

---

# Part 9 — `Microsoft.ML.CpuMath` deep dive: anatomy of SIMD + scalar fallback

Source: `src/Microsoft.ML.CpuMath/` (`AvxIntrinsics.cs` 71 KB, `SseIntrinsics.cs` 55 KB, `CpuMathUtils.*.cs`, `Thunk.cs`, `.csproj`). Part 8 listed CpuMath as the third — after BCL and Kestrel (#25) — exemplary illustration of SIMD organisation. A full section here because **this library's architecture is a ready-made template** for any code computing over `float`/`int` arrays.

## 63. Build-time fork per Target Framework — one public type, two implementations

`Microsoft.ML.CpuMath.csproj` targets `netstandard2.0;net8.0` and separates code with **conditional `<Compile Remove>`**:

```xml
<ItemGroup Condition="'$(TargetFramework)' == 'netstandard2.0'">
  <Compile Remove="CpuMathUtils.netcoreapp.cs" />
  <Compile Remove="SseIntrinsics.cs" />     <!-- managed intrinsics only on net8 -->
  <Compile Remove="AvxIntrinsics.cs" />
</ItemGroup>
<ItemGroup Condition="'$(TargetFramework)' == 'net8.0'">
  <Compile Remove="CpuMathUtils.netstandard.cs" />
</ItemGroup>
```

- `net8.0` -> `CpuMathUtils.netcoreapp.cs` + `SseIntrinsics.cs` + `AvxIntrinsics.cs` — **managed** `System.Runtime.Intrinsics`.
- `netstandard2.0` -> `CpuMathUtils.netstandard.cs` -> `Thunk.cs` with `[DllImport("CpuMathNative")]` — **native C++ DLL** (because `System.Runtime.Intrinsics` does not exist on netstandard).
- The public type is **`internal static partial class CpuMathUtils`** — the same name, identical method signatures (`MatrixTimesSource`, `Sum`, `DotU`...), two disjoint implementations. The consumer does not know which one it got.
- The NuGet package ships `build/netstandard2.0/Microsoft.ML.CpuMath.props` — MSBuild props that on the old TFM add the native `CpuMathNative` to the output (a managed-only package has no other way).

**Why:** this is the same idiom as MessagePack `UnsafeMemory32`/`UnsafeMemory64` (#31), but the boundary runs along the **framework, not the architecture**. An old platform without intrinsics does not get a slow scalar fallback — it gets native C++. `partial class` allows shared constants/helpers to live in a third file.

**How to apply:** when a library must support both new and old runtimes — not `#if` inside methods (unreadable), but **separate files + `<Compile Remove>` per TFM + `partial class`**. P/Invoke to a native DLL as fallback where managed SIMD is unavailable.

## 64. Three-tier runtime dispatch — `IsSupported` HOISTED outside the loop

`CpuMathUtils.netcoreapp.cs` — the facade checks CPU capabilities once and routes to separate classes:

```csharp
public static void MatrixTimesSource(bool transpose, AlignedArray matrix, ...)
{
    if (Avx.IsSupported)        AvxIntrinsics.MatMul(matrix, source, destination, ...);
    else if (Sse.IsSupported)   SseIntrinsics.MatMul(matrix, source, destination, ...);
    else { /* nested scalar loop — dot product by hand */ }
}
```

- AVX and SSE implementations live in **separate classes** (`AvxIntrinsics`, `SseIntrinsics`), not in one method with `if` branches.
- `Xxx.IsSupported` is an `[Intrinsic]` that JIT resolves to a **constant** during tier-1 compilation — the dead branch is eliminated, no real `if` in the generated code.
- The check is done **once, at the entry point**, never in the hot loop.

**Why:** a branch on ISA capability inside a loop over millions of elements = prediction catastrophe. Hoisted outside + JIT intrinsic-folding = zero cost.

**How to apply:** one facade per operation -> `if (Avx...) else if (Sse...) else scalar` -> separate file/class per ISA. Connects with [[dotnet-perf-reference]] #25.

## 65. Alignment mask tables — branchless handling of the head and tail

`AvxIntrinsics.cs` holds two static arrays `uint[64]` = 8 masks of 8 lanes each:

```csharp
public static readonly uint[] LeadingAlignmentMask = new uint[64] {
    0,0,0,0,0,0,0,0,                          // row 0: 0 lanes
    0xFFFFFFFF,0,0,0,0,0,0,0,                  // row 1: first lane
    0xFFFFFFFF,0xFFFFFFFF,0,0,0,0,0,0, ...     // row k: first k lanes
};
// TrailingAlignmentMask: mirror — row k = last k lanes
```

The `Sum` kernel (and every other) uses them like this:

```csharp
int misalignment = (int)((nuint)pValues % 32);     // address offset in bytes
if (misalignment != 0) {
    misalignment >>= 2;  misalignment = 8 - misalignment;   // how many floats to the boundary
    Vector256<float> mask = Avx.LoadVector256(pLeadingMask + misalignment * 8);
    result = Avx.Add(result, Avx.And(mask, Avx.LoadVector256(pValues)));  // 1 masked load
    pValues += misalignment;  length -= misalignment;       // address is now aligned
}
// ... main loop aligned, 8 floats/iteration ...
if (remainder != 0) {                                       // tail: 1-7 elements
    pValues -= (8 - remainder);                             // step back so load reaches the end
    Vector256<float> mask = Avx.LoadVector256(pTrailingMask + remainder * 8);
    result = Avx.Add(result, Avx.And(mask, Avx.LoadVector256(pValues)));
}
```

**Why:** an unaligned head (0-7 elements to the 32 B boundary) and tail (0-7 remainder) classically require **a scalar loop with a branch per element**. The mask handles both with a single vector `load + AND + add` — branchlessly, at full width. The mask table is a one-time static cost.

**How to apply:** for any reduction/transform operation on an array of arbitrary length and alignment — precompute two mask tables (leading/trailing), handle the edges with a masked load instead of a scalar loop. Edge case: when `misalignment & 3 != 0` (address not even aligned to 4 B on the 32 B grid) — alignment cannot be achieved in float-size steps, so use the purely unaligned path.

## 66. Vector-width cascade — separate accumulator per tier, merged at the end

`AvxIntrinsics.DotU` (dot product) — pattern 256->128->scalar:

```csharp
Vector256<float> result256 = Vector256<float>.Zero;
while (pSrcCurrent + 8 <= pSrcEnd) {                    // TIER 1: 8 floats at a time
    result256 = MultiplyAdd(pSrcCurrent, Avx.LoadVector256(pDstCurrent), result256);
    pSrcCurrent += 8; pDstCurrent += 8;
}
result256 = VectorSum256(result256);                    // tree reduction 8->1
Vector128<float> resultPadded = Sse.AddScalar(result256.GetLower(), GetHigh(result256));

Vector128<float> result128 = Vector128<float>.Zero;
if (pSrcCurrent + 4 <= pSrcEnd) {                       // TIER 2: one block of 4 floats
    result128 = Sse.Add(result128, Sse.Multiply(Sse.LoadVector128(pSrcCurrent), Sse.LoadVector128(pDstCurrent)));
    pSrcCurrent += 4; pDstCurrent += 4;
}
result128 = SseIntrinsics.VectorSum128(result128);

while (pSrcCurrent < pSrcEnd) {                         // TIER 3: scalar, 0-3 remainder
    result128 = Sse.AddScalar(result128, Sse.MultiplyScalar(
        Sse.LoadScalarVector128(pSrcCurrent), Sse.LoadScalarVector128(pDstCurrent)));
    pSrcCurrent++; pDstCurrent++;
}
return Sse.AddScalar(result128, resultPadded).ToScalar();   // MERGE results from all tiers
```

**Why:** the widest available loop (256 b) processes most of the data; 128 b and scalar handle the remainder. Each tier has **its own accumulator** — partial sums are not lost when switching between widths; they are merged only at the end. Matrix kernels (`MatMul`) add a second dimension to this technique: 4 output rows computed in 4 separate registers simultaneously (ILP — independent instructions that the CPU parallelises).

**How to apply:** every reduction/map on an array -> cascade `Vector256` -> `Vector128` -> scalar, separate accumulator per tier, merge at the end. For "many dot products" kernels, additionally unroll the loop N times with N accumulators.

## 67. Micro-dispatch inside the kernel — FMA and gather as inlined helpers

Inside a kernel there are occasional ISA checks, wrapped in `[MethodImpl(AggressiveInlining)]`:

```csharp
[MethodImpl(MethodImplOptions.AggressiveInlining)]
private static Vector256<float> MultiplyAdd(Vector256<float> a, Vector256<float> b, Vector256<float> c)
    => Fma.IsSupported ? Fma.MultiplyAdd(a, b, c)            // 1 instruction, better precision
                       : Avx.Add(Avx.Multiply(a, b), c);    // fallback: mul + add

[MethodImpl(MethodImplOptions.AggressiveInlining)]
private static unsafe Vector256<float> Load8(float* src, int* idx)
    => Avx2.IsSupported ? Avx2.GatherVector256(src, Avx.LoadVector256(idx), 4)  // 8 scattered -> 1 load
                        : Vector256.Create(src[idx[0]], src[idx[1]], ... src[idx[7]]); // manually
```

**Why:** `Fma.IsSupported`/`Avx2.IsSupported` are again intrinsic constants — JIT eliminates the dead branch, `AggressiveInlining` (#3) pulls the helper into the kernel, so in the hot loop there is neither a call nor a branch. FMA `a*b+c` is one instruction (Haswell+) with lower latency and no intermediate rounding. `GatherVector256` loads 8 scattered values (sparse!) in one instruction.

**How to apply:** keep punctual ISA branches in tiny `AggressiveInlining` helpers with a `?:` operator on `IsSupported` — do not inline `if` statements manually inside the kernel.

## 68. Tree-based reduction in registers — `HorizontalAdd`/`Shuffle`, not a scalar loop

Folding a vector to a scalar (sum, max) through a tree of operations, depth log₂(N):

```csharp
private static Vector256<float> VectorSum256(in Vector256<float> v) {
    Vector256<float> partial = Avx.HorizontalAdd(v, v);   // adjacent pairs summed
    return Avx.HorizontalAdd(partial, partial);            // and once more
}
private static Vector256<float> VectorMax256(in Vector256<float> v) {
    Vector256<float> x1 = Avx.Shuffle(v, v, 0xB1);         // ABCD|EFGH -> BADC|FEHG
    Vector256<float> m  = Avx.Max(v, x1);                  // pairwise max
    x1 = Avx.Shuffle(m, m, 0x02);
    return Avx.Max(m, x1);                                 // max in scalar position
}
```

**Why:** 8 elements -> 1 in 2-3 vector instructions instead of 8 scalar iterations. This is `Shuffle` + binary operation in a tree — works for any associative operation (sum, max, min, AND...).

**How to apply:** accumulate in the widest vector throughout the loop, reduce to scalar only once at the end using a `Shuffle`+op tree — never with a `for` loop over lanes.

## 69. Small but worth copying (CpuMath)

- **Name suffix convention** — the header comment of `AvxIntrinsics.cs` defines: `A` = aligned+padded, `U` = unaligned+unpadded, `P` = sparse partial vector, `Tran` = transposed. Exports have unique names (`DotU`, `AddScaleSU`, `MatMulTran`) because variants cannot be distinguished by signature alone. **Naming discipline instead of overloads** — reading the name tells you the alignment contract.
- **Unaligned-load API on aligned data** — in the main loop they use `Avx.LoadVector256` (unaligned), even though the data *is* aligned (assertion `Contracts.Assert(addr % 32 == 0)`). Reason from the code comment: JIT folds an **unaligned** load into the memory operand of the consuming instruction (VEX-encoding allows this); `LoadAligned` becomes a separate instruction. On modern hardware aligned and unaligned loads are equally fast when they don't cross a cache line. Conclusion: keep data aligned (cache-line), but call the **unaligned-load intrinsic** — for foldability.
- **`[SuppressUnmanagedCodeSecurity]` on P/Invoke** — all `[DllImport]` entries in `Thunk.cs` carry this attribute: it skips the security stack-walk on every native call (relevant on .NET Framework/netstandard; on .NET Core CAS was removed and the attribute is practically a no-op). Conceptually related to `[SuppressGCTransition]` (#46a), but a **different** attribute with a different mechanism — do not confuse them.
- **`AlignedArray`** — allocates `new float[size + cbAlign/4]` with extra capacity and shifts the data internally (`GetBase`) to guarantee 16/32 B alignment without a separate allocation. A historical pattern — **on .NET 6+ use `GC.AllocateArray<T>(n, pinned: true)` + `NativeMemory.AlignedAlloc`** instead of replicating the realignment by hand.

---

## TL;DR Part 9 — SIMD library template

| # | Pattern | What it gives |
|---|---------|---------------|
| 63 | Fork per TFM: `<Compile Remove>` + `partial class` + native fallback | New runtime = managed intrinsics, old = P/Invoke; one public API |
| 64 | Three-tier AVX/SSE/scalar dispatch, `IsSupported` outside the loop | Zero ISA branch in the hot loop (JIT folds the intrinsic constant) |
| 65 | Leading/trailing mask tables | Branchless handling of the unaligned head and tail |
| 66 | Cascade 256->128->scalar, separate accumulator per tier | Full width utilisation + correct remainder without losing partial sums |
| 67 | Micro-dispatch FMA/gather in `AggressiveInlining` helpers | Single-instruction FMA, hardware gather; no call and no branch in the kernel |
| 68 | Tree-based reduction `HorizontalAdd`/`Shuffle` | Vector->scalar in log2(N) instructions, not a scalar loop |
| 69 | Name suffixes (A/U/P/Tran), unaligned-load for foldability, `AlignedArray` (obsolete) | Alignment contract discipline; JIT codegen nuances |

Full recipe for a SIMD library: **one facade per operation -> ISA dispatch at entry -> separate class per ISA -> inside the kernel: width cascade + edge masks + tree reduction -> FMA/gather micro-dispatch inline -> native fallback for TFMs without intrinsics.** The warning from #25 applies: SIMD pays off at millions of elements/s — for CRUD/IDE it is overengineering.

---

*Part 9 added 2026-05-16. Deep dive into `Microsoft.ML.CpuMath`. **~84 concepts** in 9 parts in total. Closing the SIMD thread from #25: after BCL (intrinsics laboratory) and Kestrel (HTTP parsing) — ML.NET CpuMath as the third, most clearly organised "SIMD + scalar fallback" pattern ready to copy as a template.*

---

# Part 10 — `dotnet/aspnetcore`: 26 patterns from a live repository

Source: code review of the `C:\Praca\aspnetcore` repository (branch `main`, May 2026) — five parallel passes: Kestrel transport, Kestrel HTTP protocols (HTTP/1·2·3), `src/Http` + routing + `WebUtilities`, `src/Shared` (utility source-linked), and Components (Blazor) / SignalR / middleware. Listed are **only** patterns not present in parts 1-9. Each one verified in the source. Unlike the decompilation in parts 1-4, this is **original code with author comments** — literally-quoted comments are invaluable as justification.

## A. New class — codegen / JIT

### 70. `[UnsafeAccessor]` — access to private members without reflection

Source: `src/Shared/Components/ComponentsActivityLinkStore.cs`

```csharp
[UnsafeAccessor(UnsafeAccessorKind.Method, Name = "get_ActivityLinksStore")]
static extern object GetActivityLinksStore(Renderer instance);
```

**Why:** a .NET 8+ attribute — JIT binds the `extern` method directly to a private member of the target type. Result: a plain `call`, zero `MethodInfo.Invoke`, zero security stack-walk, zero `object[]` allocation for arguments. Unlike `Expression.Compile`/`DynamicMethod` ([[dotnet-perf-reference]] #39-41) it does not emit IL at runtime — it is fully static, trim-friendly, and AOT-friendly.

**How to apply:** when you need to call a private method/field/constructor of someone else's type (BCL interop, tests, bridges between assemblies) — declare `static extern` with `[UnsafeAccessor]` instead of a cached `MethodInfo`. The signature must match; the first parameter is the instance. Connects with [[dotnet-perf-reference]] #41 (codegen) as its static, lightweight alternative.

### 71. `SearchValues<T>` — character set compiled to SIMD

Source: `src/Shared/ServerInfrastructure/HttpCharacters.cs`, `src/Http/Http.Abstractions/src/PathString.cs`

```csharp
private static readonly SearchValues<byte> _allowedAuthorityBytes =
    SearchValues.Create(":.-[]@0123456789ABC...xyz"u8);

public static bool ContainsInvalidAuthorityChar(ReadOnlySpan<byte> span)
    => span.IndexOfAnyExcept(_allowedAuthorityBytes) >= 0;
```

**Why:** `SearchValues<T>` (.NET 8+) analyses the set once at static initialisation and selects the fastest search algorithm (bitmap / SIMD / per-character). `IndexOfAnyExcept` gives "is everything in the allowed set" validation at vector speed. `PathString.ToUriComponent()` uses this as a fast-path: if `SearchValues` finds no character to escape — return the string without `StringBuilder`.

**How to apply:** any character validation/classification (allowed URL characters, separators, whitelist) — replace a `for` loop over characters or a regex with a static `SearchValues<char>`/`<byte>` + `IndexOfAny`/`IndexOfAnyExcept`. Fast-path "is there anything to do at all" first, then the slow path. Related to #25 (SIMD), but without manual intrinsics.

### 72. Source-generated `KnownHeaders` — matching header names by reading 8 bytes as a `ulong`

Source: `src/Servers/Kestrel/Core/src/Internal/Http/HttpHeaders.Generated.cs` (generated from `KnownHeaders.cs`)

```csharp
// case-insensitive: 0xdfdfdfdf clears bit 6 of every ASCII byte (49x in the file)
if ((ReadUnalignedLittleEndian_ulong(ref nameStart) & 0xdfdfdfdfdfdfdfdfuL) == 0x...)
// long _bits — one bit per known header:
_bits |= 0x2L;                       // set
if ((_bits & 0x2L) != 0) { ... }     // presence test — O(1)
```

**Why:** the header name is read as a number (`Unsafe.ReadUnaligned<ulong>`) and compared with a constant — O(1) regardless of length, no string allocation, no `string.Equals`. The `0xdf...` mask gives **branchless** case-insensitivity. `long _bits` holds ~60 known headers: checking "is the header already set" is one bit test, one cache-line access.

**How to apply:** when parsing a protocol with a finite vocabulary of names — generate (via source generator) code that reads names as 2/4/8-byte integers and compares them with constants; use bit-flag `long`/`ulong` instead of `HashSet` to track field presence. Extends #4 (StringTable) with **codegen matching**, and #6 (frozen) with the case of a dictionary known at compile time.

## B. Bit tricks / sentinel

### 73. Fusing bounds-check and sentinel via cast to `uint`

Source: `src/Servers/Kestrel/Core/src/Internal/Http/HttpParser.cs`

```csharp
var index = (uint)span.IndexOf(target);   // -1 (not found) -> uint.MaxValue
if (index < (uint)span.Length) { ... }    // one branch instead of (index >= 0 && index < len)
```

**Why:** `IndexOf` returns `-1` when nothing is found. Casting to `uint` turns `-1` into `uint.MaxValue` — a single unsigned comparison simultaneously handles "not found" and "out of range". Fewer branches = better prediction on the parser hot path. JIT also uses this to eliminate bounds checks.

**How to apply:** when you have an index that can be `-1` and you need to check the upper bound anyway — cast to `uint` and compare once. This idiom complements #46e (fastest loops) with a concrete sentinel trick.

### 74. Branchless hex prefix length calculation (chunked encoding)

Source: `src/Servers/Kestrel/Core/src/Internal/Http/ChunkWriter.cs`

```csharp
total = (count > 0xffff) ? 0x10 : 0x00;  count >>= total;
shift = (count > 0x00ff) ? 0x08 : 0x00;  count >>= shift;  total |= shift;
total |= (count > 0x000f) ? 0x04 : 0x00;        // position of the highest non-zero nibble
// ...write hex directly to Span, hex from "0123456789abcdef"u8 (data section)
```

**Why:** binary search for the highest non-zero nibble — O(1), no `div`/`mod`, no loop, no allocation. Chunk size written directly to `Span<byte>`; the hex table is a `u8` literal mapped to the rodata section (zero allocations). `& 0x0f` when indexing gives JIT proof to eliminate the bounds check.

**How to apply:** conversions with a known maximum width (hex, frame sizes) — decompose using bit shifts and write to the supplied buffer; keep translation tables as `u8` literals. Related to #46d (`string.Create`), but even lower level — directly to `Span`.

### 75. Setting the continuation bit in a 7-bit varint via OR instead of a branch

Source: `src/Shared/Encoding/Int7BitEncodingUtils.cs`

```csharp
while (uValue > 0x7Fu) {
    target[index++] = (byte)(uValue | ~0x7Fu);   // ~0x7F = 0xFFFFFF80 — sets the continuation bit
    uValue >>= 7;
}
target[index++] = (byte)uValue;
```

**Why:** classic varint sets bit 8 conditionally. Here `uValue | ~0x7F` sets the continuation bit **unconditionally**, and the `(byte)` cast discards the overflow — no branch, no misprediction. Operates directly on `Span<byte>` (zero `BinaryWriter`). Reading validates `shift == 35` (overflow > 32 bits) — early detection of bad data.

**How to apply:** when encoding variable-length numbers to a buffer — replace the conditional flag-setting with an unconditional OR + masking cast. Related to #7/#19 (protocol parsers).

### 76. Perfect hash for HTTP methods (generated by gperf)

Source: `src/Servers/Kestrel/Core/src/Internal/Infrastructure/HttpUtilities.cs`

```csharp
// "GET " read as uint (with trailing space) — checked FIRST (most common)
// longer methods: up to 8 bytes as ulong + perfect-hash table from GNU gperf
```

**Why:** HTTP methods are short, fixed ASCII strings — numeric comparison is O(1) and does not hash the string. A perfect-hash table (collision-free, minimal) generated **offline** by gperf; lookup is 2-3 memory reads. The most common method (`GET`) has a separate, earliest fast-path.

**How to apply:** for any finite vocabulary (methods, codes, schemes) — consider an offline-generated perfect hash instead of `Dictionary`/`switch` on strings; privilege the statistically most common case. A concrete realisation of the idea from #6 (frozen collections).

## C. Memory, write-barriers, pooling

### 77. Struct-wrapper bypassing covariance check on array write

Source: `src/Shared/Buffers/BufferSegmentStack.cs`, `src/Servers/Kestrel/shared/PooledStreamStack.cs`

```csharp
// code comment points directly to: clr!ObjIsInstanceOf
private readonly struct SegmentAsValueType {
    private readonly BufferSegment _value;
    private SegmentAsValueType(BufferSegment v) => _value = v;
    public static implicit operator SegmentAsValueType(BufferSegment s) => new(s);
    public static implicit operator BufferSegment(SegmentAsValueType s) => s._value;
}
private readonly SegmentAsValueType[] _array;   // array of STRUCTS, not references
```

**Why:** writing a reference type to `T[]` forces the CLR to perform a covariant-array-store check (`JIT_Stelem_Ref` -> `ObjIsInstanceOf`) — overhead on **every** write. Wrapping a reference in a `readonly struct` makes the array an array of values — the check disappears. The `implicit` operators make the wrapper invisible to callers. Microsoft measured this in ETL traces (comment in `BufferSegmentStack`).

**How to apply:** an array/pool holding reference types written in a hot loop — wrap the element in a single-field `readonly struct` with `implicit` operators. Eliminates the hidden `ObjIsInstanceOf`. Extremely non-obvious; only worth doing for measured hot paths.

### 78. `nuint` as value in `ConcurrentDictionary` — eliminating GC write barriers

Source: `src/Servers/Kestrel/Core/src/Internal/PinnedBlockMemoryPoolFactory.cs`

```csharp
// micro-optimization: Using nuint as the value type to avoid GC write barriers;
// could replace with ConcurrentHashSet if that becomes available
private readonly ConcurrentDictionary<PinnedBlockMemoryPool, nuint> _pools = new();
_pools.TryAdd(pool, nuint.Zero);   // value is irrelevant — dictionary used as a set
```

**Why:** `ConcurrentDictionary` used as a `HashSet` (no `ConcurrentHashSet` in BCL). If the value were `object`, every entry would be a GC root requiring a **write barrier**. `nuint` is a value type — the entry does not participate in GC scanning, write without a barrier. Quoted from the author's comment.

**How to apply:** you need a concurrent set -> `ConcurrentDictionary<T, nuint>` (or another value type like `byte`) instead of `<T, object>`. The value is ignored, but the write-barrier overhead disappears. Related to #4 (false sharing) — awareness of GC memory costs.

### 79. Pooling by overriding `Dispose()`

Source: `src/Shared/CancellationTokenSourcePool.cs`

```csharp
private sealed class PooledCancellationTokenSource : CancellationTokenSource {
    protected override void Dispose(bool disposing) {
        if (disposing && !_pool.Return(this))   // attempt to return to the pool
            base.Dispose(disposing);             // real dispose only when the pool is full
    }
}
```

**Why:** `CancellationTokenSource` is relatively expensive to allocate. By inheriting and intercepting `Dispose`, the pool becomes **completely transparent** to the caller — a plain `using (var cts = pool.Rent())` actually returns the object to the pool. Pool bound (`MaxQueueSize = 1024`) — when exceeded, graceful fallback to real `Dispose`.

**How to apply:** a type that already implements `IDisposable` and is expensive — consider a subclass that intercepts `Dispose` and returns `this` to the pool. The caller changes nothing. Note: only when the type is inheritable and safe to reset. Related to #1 (object pooling) — the "invisible pool" variant.

### 80. Approximate counting instead of `ConcurrentQueue.Count`

Source: `src/Servers/Kestrel/Transport.Sockets/src/Internal/SocketSenderPool.cs`, `src/Shared/CancellationTokenSourcePool.cs`

```csharp
// "This counting isn't accurate, but it's good enough ... to avoid using
//  _queue.Count which could be expensive"
if (_disposed || Interlocked.Increment(ref _count) > MaxQueueSize) {
    Interlocked.Decrement(ref _count);
    sender.Dispose();   // pool full — discard
    return;
}
```

**Why:** `ConcurrentQueue.Count` is expensive (synchronisation, traversal of segments). A separate `int` counter with `Interlocked` is cheap; it can momentarily exceed `MaxQueueSize` due to a race, but that minor inaccuracy is incomparably cheaper than an accurate `Count`. Quoted from the comment.

**How to apply:** bounding the size of a concurrent pool/queue — keep an approximate `Interlocked` counter instead of calling `.Count`. Accept that the bound is "soft". The "good enough" philosophy — related to #13 (LRU), #28 (custom Parallel).

### 81. `RemoveExpired` in a single scan thanks to FIFO ordering

Source: `src/Servers/Kestrel/shared/PooledStreamStack.cs`

```csharp
// streams in the pool are in expiry order -> first non-expired = all subsequent are valid
for (var i = 0; i < size; i++)
    if (array[i].PoolExpirationTimestamp >= timestamp) return i;   // cutoff
// then: dispose [0..cutoff), in-place compaction, clear tail — zero allocations
```

**Why:** the invariant "streams are added in ascending expiry order" turns pool cleanup into a single scan + in-place array compaction. No auxiliary structures, no allocations; in steady state the cost is near zero.

**How to apply:** a pool with TTL — store elements in expiry order; eviction becomes a single scan to the first live element + compaction. Related to #13 (bounded LRU cache) — an alternative without `LinkedList`.

### 82. Two pointers — rewriting a span in-place

Source: `src/Shared/UrlDecoder/UrlDecoder.cs`, `src/Shared/PathNormalizer/PathNormalizer.cs`

```csharp
public static int DecodeInPlace(Span<byte> buffer, bool isFormEncoding) {
    var sourceIndex = 0; var destinationIndex = 0;
    // decoded result is ALWAYS <= input -> write pointer never catches read pointer
    while (sourceIndex < buffer.Length) { /* '+', '%XX', literal -> buffer[destinationIndex++] */ }
    return destinationIndex;   // new length
}
```

**Why:** URL-decode and RFC-3986 `RemoveDotSegments` have the invariant "result is no longer than input". This allows rewriting the data **in the same buffer**: the read pointer always stays ahead of the write pointer. Zero intermediate buffers, zero allocations; the method returns the new length for truncation.

**How to apply:** a buffer transformation that never lengthens data (decoding, removing characters, whitespace compression) — do it in-place with the two-pointer technique instead of allocating an output buffer. Related to #61 (editor pattern) and #46c (`CollectionsMarshal.AsSpan`).

### 83. Revision number — cheap cache invalidation on pool reuse

Source: `src/Http/Http/src/DefaultHttpContext.cs` (`Initialize` / `Uninitialize`)

```csharp
public void Initialize(IFeatureCollection features) {
    _features.Initialize(features, revision: features.Revision);   // bump revision
    _request.Initialize(_features.Revision);
    // feature lookups compare the saved revision with the current one — mismatch = recompute
}
```

**Why:** `HttpContext` is pooled between requests. Instead of zeroing the entire structure and all feature caches on reuse, a single `int Revision` is incremented. Cached lookups detect staleness by a revision mismatch — lazily, without zeroing memory.

**How to apply:** a pooled object with cached derived state — instead of clearing the cache on reuse, keep a version counter; cache consumers remember the version and recompute on mismatch. Related to #1 (pooling), #34 (per-request object pool).

## D. Concurrency / async

### 84. `IOQueue` — `Thread.MemoryBarrier()` + CAS gate on `_doingWork`

Source: `src/Servers/Kestrel/Transport.Sockets/src/Internal/IOQueue.cs`

```csharp
public override void Schedule(Action<object?> action, object? state) {
    _workItems.Enqueue(new Work(action, state));
    if (Interlocked.CompareExchange(ref _doingWork, 1, 0) == 0)     // only the FIRST queues work
        ThreadPool.UnsafeQueueUserWorkItem(this, preferLocal: false);
}
void IThreadPoolWorkItem.Execute() {
    while (true) {
        while (_workItems.TryDequeue(out var item)) item.Callback(item.State);
        _doingWork = 0;
        Thread.MemoryBarrier();              // orders the non-volatile write <-> read below
        if (_workItems.IsEmpty) break;
        if (Interlocked.Exchange(ref _doingWork, 1) == 1) break;
    }
}
```

**Why:** all I/O callbacks merge into **one** ThreadPool work item (fewer context switches, better locality). The `_doingWork` flag via CAS guarantees that only the first producer queues the work. The explicit `Thread.MemoryBarrier()` orders the non-volatile write `_doingWork = 0` with the subsequent `IsEmpty` read — without the cost of a volatile-read on every iteration. The classic fix for the "work added after clearing the flag" race.

**How to apply:** producer/consumer where you want to batch work into one work item — "am working" flag via CAS, after draining the queue clear the flag, `MemoryBarrier`, check emptiness again. Related to #7 (lock-free), #21 (`Pipe`).

### 85. Continuation coalescing via a singleton sentinel `Action`

Source: `src/Servers/Kestrel/Transport.Sockets/src/Internal/SocketAwaitableEventArgs.cs`

```csharp
private static readonly Action<object?> _continuationCompleted = _ => { };
private volatile Action<object?>? _continuation;
// in OnCompleted: CAS inserts either the real continuation or the sentinel
if (ReferenceEquals(prevContinuation, _continuationCompleted))
    ThreadPool.UnsafeQueueUserWorkItem(continuation, state, preferLocal: true);
```

**Why:** a single static empty `Action` as a sentinel allows one `Interlocked.CompareExchange` to distinguish three awaiter states (waiting / completed / completed-before-registration) without separate flag fields and without allocation. Works together with `IValueTaskSource`.

**How to apply:** when implementing a custom awaiter/`IValueTaskSource` — use a static sentinel delegate as a state guardian in the continuation field; a CAS on that field replaces a separate `enum`/`bool`. A concrete trick complementing #14 (`PipeAwaitable`).

### 86. Async work deduplication via race on `ConcurrentDictionary` + `TaskCompletionSource`

Source: `src/Middleware/OutputCaching/src/DispatcherExtensions.cs` (`WorkDispatcher`)

```csharp
while (true) {
    if (_workers.TryGetValue(key, out var task)) return await task;   // join in-progress work
    var tcs = new TaskCompletionSource<TValue?>(TaskCreationOptions.RunContinuationsAsynchronously);
    if (_workers.TryAdd(key, tcs.Task)) {                              // you won — you are the producer
        try     { tcs.TrySetResult(await valueFactory(key, state)); return await tcs.Task; }
        finally { _workers.TryRemove(key, out _); }
    }   // you lost the race — loop and join someone else's Task
}
```

**Why:** classic protection against **cache stampede / dogpile** — N parallel requests for the same key execute `valueFactory` only once; the rest `await` the shared `Task` from `TaskCompletionSource`. No locks — synchronisation follows from the atomic `TryAdd`. `RunContinuationsAsynchronously` prevents continuations from running on the producer's thread. Exceptions propagate to all awaiters.

**How to apply:** an expensive computation/fetch cached by key, called concurrently — hold a `ConcurrentDictionary<TKey, Task<TValue>>`, elect the producer via `TryAdd`, remove the entry in `finally`. Related to #12 (`GetOrAdd`) — but for **asynchronous** work, where `GetOrAdd` is insufficient.

### 87. `ConcurrentPipeWriter` — passthrough mode when nothing is flushing

Source: `src/Servers/Kestrel/Core/src/Internal/Infrastructure/PipeWriterHelpers/ConcurrentPipeWriter.cs`

```csharp
public override Memory<byte> GetMemory(int sizeHint = 0) {
    if (_currentFlushTcs == null && _head == null)        // no flush in progress and no data buffered
        return _innerPipeWriter.GetMemory(sizeHint);      // PASSTHROUGH — zero buffering, zero lock
    AllocateMemoryUnsynchronized(sizeHint);
    return _tailMemory;
}
```

**Why:** as long as no flush is in progress, the writer delegates directly to the inner one — no segmented buffer, no synchronisation. The segmented buffer (list of reusable segments) activates **only** during an in-progress flush, when concurrent writes need somewhere to go. One `TaskCompletionSource` per flush is awaited by all writers.

**How to apply:** a decorator adding "just in case" buffering/synchronisation — detect the common case where the decoration is unnecessary and delegate directly in that case. Related to #7 (`Allocate`/`AllocateSlow` — fast/slow split at the method level).

## E. Hot-path / architecture

### 88. Per-connection request string reuse cache

Source: `src/Servers/Kestrel/Core/src/Internal/Http/Http1Connection.cs`

```csharp
private string? _parsedPath, _parsedQueryString, _parsedRawTarget;
// keep-alive: if new raw target == previous -> reuse string reference, skip parsing
```

**Why:** a keep-alive connection often receives **identical** requests (polling, retry, healthcheck). Instance fields hold the decoded strings from the previous request; if the raw target matches — the ready reference is reused, without re-decoding and re-allocating. Per-connection (not global) -> zero contention. `DisableStringReuse` can be turned off for multi-tenant scenarios.

**How to apply:** a stateful parser processing a stream of similar inputs — cache the last result in instance fields and check input identity before re-parsing. Related to #4 (`StringTable` dedupe) — but local, single-element, without synchronisation.

### 89. `Date` header cached by heartbeat + `Volatile`

Source: `src/Servers/Kestrel/Core/src/Internal/Http/DateHeaderValueManager.cs`

```csharp
// heartbeat timer once/sec updates the struct with a ready byte[] (including "\r\nDate: " prefix)
// all threads: Volatile.Read of the same instance — zero formatting per response
```

**Why:** every HTTP response needs a `Date` header, but the date changes only once per second. The heartbeat (one background thread) formats it once; the ready `byte[]` already includes the `"\r\nDate: "` prefix and `\r\n`, so emitting the response is a pure byte copy. Read via `Volatile.Read` — no lock.

**How to apply:** a value sent in every response but changing rarely — format it in the background on a timer, keep a ready byte buffer (with prefixes), read `Volatile`. Related to #20 (batch + timer) and #60a — the "background-refreshed cache" pattern.

### 90. Pre-boxed empty enumerator

Source: `src/Http/Http/src/QueryCollection.cs`, `src/Http/Http/src/FormCollection.cs`

```csharp
private static readonly IEnumerator<KeyValuePair<string, StringValues>> EmptyIEnumerator
    = new Enumerator();   // empty case boxed ONCE, statically
// non-empty collection -> GetEnumerator returns struct Enumerator directly (no box)
// NOTE from code: field _dictionaryEnumerator must NOT be readonly — MoveNext mutates state
```

**Why:** `foreach` over `IEnumerable<T>` boxes a struct-enumerator. The empty collection is common (no query string, no form) — its enumerator is boxed **once** into a static field. Non-empty returns the `struct` directly. The comment "Do NOT make this readonly" highlights the trap: `readonly` on a field holding a mutable struct-enumerator causes defensive copies and lost state.

**How to apply:** a collection that is often empty and implements `IEnumerable<T>` — box the empty enumerator into a `static readonly`; for non-empty, return the `struct`. Do not mark fields holding mutable struct-enumerators as `readonly`. Related to #54 (frugal object), #29.

### 91. `KeyValueAccumulator` — staged expansion 1 -> array -> `List`

Source: `src/Http/WebUtilities/src/KeyValueAccumulator.cs`

```csharp
// 1 value: inline in the main dictionary
// 2 values: StringValues with a 2-element array, still in the main dictionary
// 3+:       migration to a separate _expandingAccumulator (List, capacity 8),
//           marker in main dictionary (empty StringValues) -> "see second dictionary"
```

**Why:** multi-value keys in query/form typically have 1-2 values; allocating a `List` for each is wasteful. The accumulator transitions through states: single value -> 2-element array -> `List` in a separate dictionary. The main dictionary does not grow with lists; the `List` is created only on the 3rd value, immediately with `capacity: 8`.

**How to apply:** collecting values grouped by key where "typically 1-2, rarely more" — implement staged expansion with inline state; allocate the full collection only after exceeding the threshold. Related to #54 (frugal `StringValues`/`FrugalList`), #27 (`AdaptiveCapacityDictionary`).

### 92. `[MethodImpl(NoInlining)]` explicitly on the cold path

Source: `src/Shared/ServerInfrastructure/BufferExtensions.cs` (`WriteNumeric` / `WriteNumericMultiWrite`)

```csharp
[MethodImpl(MethodImplOptions.AggressiveInlining)]
internal static void WriteNumeric(...) {
    if (number < 10  && ...) { ... }            // fast path: 1-3 digits inline
    else WriteNumericMultiWrite(ref bufferWriter, number);
}
[MethodImpl(MethodImplOptions.NoInlining)]      // COLD path explicitly excluded from inlining
private static void WriteNumericMultiWrite(...) { /* division, scratch buffer */ }
```

**Why:** this **refines #5/#7**. Earlier sections said "hot path -> separate `*Slow` method, helps JIT inline the fast path". The key here is the explicit `[NoInlining]` on the cold method: it guarantees that its IL **will not enlarge** the hot method, keeping it below the inlining threshold. Extracting the method alone is not enough — JIT might pull it back in.

**How to apply:** fast/slow split — mark the hot method `[AggressiveInlining]` and the cold one explicitly `[MethodImpl(MethodImplOptions.NoInlining)]`. The attribute on the cold part is just as important as on the hot part.

### 93. Anti-DRY in hot-path — deliberate code duplication (author quotes)

Source: `src/Components/Components/src/RenderTree/RenderTreeDiffBuilder.cs`, `RenderTreeFrameArrayBuilder.cs`

```text
// RenderTreeDiffBuilder (diff method ~1000 lines):
//   "This is deliberately a very large method ... A naive 'extract methods'-type
//    refactoring will worsen perf by about 10%."  (parameter-passing cost)
// RenderTreeFrameArrayBuilder (buffer growth check copied to every Append* method):
//   "intentionally inlined into each method because doing so improves intensive
//    rendering scenarios by around 1%."
```

**Why:** in extremely hot paths (render tree diff) extracting small methods costs — parameter passing, less freedom for JIT in register allocation. Blazor measures: an "extract method" refactor worsens diff by ~10%, and a manually copied buffer growth check improves rendering by ~1%. Instead of methods — `#region` as "pre-inlined" sections + `ref struct` context (`DiffContext`) passed by `ref` instead of 8 parameters.

**How to apply:** **only** in measured, hottest paths — accept large methods and duplication; document the decision and the benchmark number; organise code with `#region`. Hold the state of a recursive algorithm in a `ref struct` passed by `ref` (cf. #56 — parameter passing is a real cost; #61 — `ref struct` context).

### 94. `WeakReference[]`-based bounded LRU — when `ConditionalWeakTable` fails

Source: `src/Shared/RoslynUtils/BoundedCacheWithFactory.cs`

```csharp
// array of ~5 WeakReference<Entry>; hit moved to end of list (LRU),
// on overflow: overwrite oldest / first dead slot
```

**Why:** `ConditionalWeakTable` (#33) does not work when the **value cyclically references the key** — in that case CWT will never release the entry. Solution: a small array of `WeakReference`, manual LRU policy, hard bound (~5). GC can collect entries; slots are reclaimed without retention.

**How to apply:** a key->value cache where the value holds a reference to the key — do not use `ConditionalWeakTable`; use a bounded `WeakReference` array with LRU. **Addendum to the caveat in #33**: CWT fails on a value->key cycle.

### 95. `ManualResetEventSlim(spinCount: 0)` for long waits

Source: `src/Servers/Kestrel/Core/src/Internal/Infrastructure/Heartbeat.cs`

```csharp
// "Wait time is long, so don't try to spin to exit early. Spinning would waste CPU time."
_stopEvent = new ManualResetEventSlim(false, spinCount: 0);
```

**Why:** `ManualResetEventSlim` spins by default before sleeping — this pays off for waits in the microsecond range. The heartbeat thread waits **one second** between ticks; spinning would be pure CPU waste. Explicit `spinCount: 0` disables it. A dedicated background thread instead of a `Timer` gives a predictable tick for thousands of connections.

**How to apply:** using `ManualResetEventSlim`/`SemaphoreSlim` for **long** waits (much longer than microseconds) — set `spinCount: 0`. The default spin is optimised for short waits. Related to #48d (obsolete lock primitives) — choosing the primitive to match the wait duration.

---

## TL;DR Part 10

| # | Pattern | When |
|---|---------|------|
| 70 | `[UnsafeAccessor]` | Calling a private member without reflection (.NET 8+, AOT-friendly) |
| 71 | `SearchValues<T>` + `IndexOfAnyExcept` | Character validation/classification — SIMD, zero manual intrinsics |
| 72 | Header codegen: `ulong`-read + `0xdf` mask + bit-flags | Protocol parsing with a finite vocabulary of names |
| 73 | Index cast to `uint` (-1 -> `uint.MaxValue`) | Fusing sentinel check and upper bound into one branch |
| 74 | Branchless hex prefix length (cascade `>>`) | Conversion with known max width, write to `Span` |
| 75 | `value \| ~0x7F` | Branchless continuation-bit setting in varint |
| 76 | Perfect hash (gperf) | Finite string vocabulary; privilege the most common |
| 77 | `readonly struct` wrapper on array of ref types | Eliminate covariant-array-store check (`ObjIsInstanceOf`) |
| 78 | `ConcurrentDictionary<T, nuint>` | Concurrent set without GC write barriers |
| 79 | Pool via `override Dispose()` | Transparent pooling of an `IDisposable` type |
| 80 | Approximate `Interlocked` counter | Soft pool bound instead of expensive `.Count` |
| 81 | Eviction in a single scan (FIFO expiry order) | Pool/cache with TTL |
| 82 | Two pointers, in-place span rewrite | Transformation that never lengthens data |
| 83 | Revision number | Cheap cache invalidation on pooled object reuse (no zeroing) |
| 84 | `MemoryBarrier` + CAS gate `_doingWork` | Batching callbacks into one work item |
| 85 | Singleton sentinel `Action` | Awaiter state in one field + CAS |
| 86 | Race on `ConcurrentDictionary` + `TaskCompletionSource` | Deduplication of expensive **async** work (anti-stampede) |
| 87 | Passthrough mode in decorator | Skip buffering/lock when unnecessary in the common case |
| 88 | Per-connection string reuse | Stateful parser of a stream of similar inputs |
| 89 | Heartbeat-cache + `Volatile` | Value in every response, changing rarely |
| 90 | Pre-boxed empty enumerator | Collection often empty and implementing `IEnumerable<T>` |
| 91 | Staged expansion 1->array->`List` | Values grouped by key, typically 1-2 |
| 92 | `[NoInlining]` on cold path | Refinement of #5 — protects the hot method's size |
| 93 | Anti-DRY in hot-path (measured) | Hottest paths: large method + duplication instead of extract |
| 94 | `WeakReference[]` bounded LRU | Cache where value references key (CWT will fail — cf. #33) |
| 95 | `ManualResetEventSlim(spinCount: 0)` | Long waits (much longer than us) — disable default spin |

---

*Part 10 added 2026-05-16. Review of the live `dotnet/aspnetcore` repository (branch `main`). **~110 concepts** in 10 parts in total. Unlike parts 1-9 (decompilation + curated lists) — this is original code with author comments; quotes from the comments (`SocketSenderPool` "good enough", `RenderTreeDiffBuilder` "-10%", `PinnedBlockMemoryPoolFactory` "avoid GC write barriers") provide justification not visible in decompiled IL. The four most non-obvious: #77 (struct-wrapper vs covariant-store), #78 (`nuint` vs write-barrier), #70 (`[UnsafeAccessor]`), #86 (anti-stampede).*
