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
