; Unshipped analyzer release
; https://github.com/dotnet/roslyn-analyzers/blob/main/src/Microsoft.CodeAnalysis.Analyzers/ReleaseTrackingAnalyzers.Help.md

### New Rules

Rule ID | Category | Severity | Notes
--------|----------|----------|-------
OVERFIT001 | Performance | Warning | Heap array allocation in per-call code — use PooledBuffer/PooledArray/TensorStorage/stackalloc
OVERFIT002 | Performance | Warning | Jagged array allocation in per-call code — use a flat array Span-sliced per row
OVERFIT003 | Performance | Warning | Boxing conversion in per-call code
OVERFIT004 | Performance | Warning | Closure/delegate allocation in per-call code (capturing lambda or instance method group)
OVERFIT005 | Performance | Warning | foreach over an interface-typed collection in per-call code
OVERFIT006 | Performance | Warning | String interpolation/concatenation in per-call code (exception path exempt)
OVERFIT007 | Performance | Warning | params call materialises a hidden array in per-call code
OVERFIT008 | Performance | Warning | Raw Parallel.For/ForEach/Invoke — use suppress-aware OverfitParallel
OVERFIT009 | Performance | Warning | .ToArray() in per-call code — slice or use pooled buffers
OVERFIT010 | Performance | Warning | Growable collection (List/Dictionary/HashSet/Queue/Stack/StringBuilder/MemoryStream) allocated in per-call code — pool it
OVERFIT011 | Performance | Warning | Struct dictionary/set key without IEquatable<T> — reflection-based default equality
OVERFIT012 | Performance | Warning | Finalizer declared — slower allocation, extra GC generation; use IDisposable + GC.SuppressFinalize
OVERFIT013 | Performance | Warning | .Count on ConcurrentQueue/ConcurrentBag — synchronized segment walk; use IsEmpty or an Interlocked counter
OVERFIT014 | Performance | Warning | Case-folding string equality (a.ToLower() == b.ToLower()) — use string.Equals with StringComparison.OrdinalIgnoreCase
OVERFIT015 | Performance | Warning | Direct intrinsics IsSupported — gate ISA paths through CpuFeatures
OVERFIT016 | Performance | Warning | Large struct (est. > 64 B) passed by value — pass as `in`
OVERFIT017 | Performance | Warning | Struct with only readonly fields is not declared `readonly struct` — defensive copies
OVERFIT018 | Performance | Warning | readonly field of a mutable struct — defensive copy per access, mutation silently lost
OVERFIT019 | Performance | Warning | Non-capturing lambda without `static` — guard against future accidental captures
OVERFIT900 | Performance | Error | A per-call OVERFIT rule fired inside an [OverfitHotPath] member/type — escalated to a build error
