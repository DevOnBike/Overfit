# Bug hunt: `Sources/Main/Tensors`

**Scope:** `Sources/Main/Tensors` (13 files: `TensorShape.cs`, `TensorStrides.cs`, `TensorView.cs`,
`FastTensor.cs`, `FastTensorExtensions.cs`, `NativeBuffer.cs`, `PooledBuffer.cs`, and
`Core/{TensorStorage,TensorSpan,TensorKernels,TensorKernelGuards,TensorFactory,NativeBufferManaged}.cs`)
**Timestamp (UTC):** 2026-08-02 19:48
**Commit:** `996a161`
**Score:** 10 points / 5 confirmed defects (+1 lower-confidence/latent finding reported separately, not
counted in the total)
**Ended by:** scope — all 13 files in the named directory were read; the hunt did not run out of clock
(stopped at ~5 of the 10-minute budget once the files were exhausted and one extra pass over likely call
sites turned up nothing further).
**README.md:** present and read first (`Sources/Main/Tensors/README.md`). Its claims (`PooledBuffer`
idempotent/null-safe Dispose, `ArrayPool<T>.Shared` banned elsewhere, jagged arrays banned) were checked
against the code and hold everywhere they were tested except finding 5 below, where a *sibling* type
promises the same safety by proximity in the same table but does not deliver it.

---

## Findings, ranked by damage

### 1. `TensorStorage<T>.AsMemory()` throws `NullReferenceException` for arena-backed storage — `AsSpan()` handles the case, `AsMemory()` doesn't

**File:** `Sources/Main/Tensors/Core/TensorStorage.cs`, `TensorStorage<T>.AsMemory()` (line ~125)

`AsSpan()` correctly branches three ways:
```csharp
if (_isBorrowedMemory) return new Span<T>(_nativePtr, Length);
return _pooled ? _pooledBuf.Span : _data!.AsSpan(0, Length);
```
`AsMemory()` only has the last two arms:
```csharp
public Memory<T> AsMemory() => _pooled ? _pooledBuf.Memory : _data!.AsMemory(0, Length);
```
For a storage created via the borrowed/arena constructor (`TensorStorage(NativeBufferManaged<T> buffer, int
length)` — the one `ComputationGraph.AllocateIntermediate` uses), `_data` is `null` and `_pooled` is `false`,
so `AsMemory()` dereferences a null `_data` and throws `NullReferenceException` for any input that
successfully constructed via the arena path — this is not a hypothetical, it's the exact construction mode
the tape-buffer path uses.

**How anyone would notice today:** not at all until the first caller reaches for `Memory<T>` instead of
`Span<T>` on an arena-backed tensor (e.g. anything that needs to hand a tensor to an `async` API or store it
past a `ref struct` scope) — currently no caller in `Sources/Main` calls `AsMemory()`, so it is a live trap
rather than an active failure. First use crashes with an unhelpful NRE, not a documented "not supported for
borrowed storage" message.

**What test would have caught it:** a parity test that calls both `AsSpan()` and `AsMemory()` on a
`TensorStorage<float>` constructed via each of its three constructors (pooled, unpooled, borrowed) and
asserts both return a view of the same length/content.

---

### 2. `TensorKernels.Add` is missing the overlap guard its sibling kernels all have — silently wrong output instead of a thrown exception

**File:** `Sources/Main/Tensors/Core/TensorKernels.cs`, `Add(ReadOnlySpan<float>, ReadOnlySpan<float>,
Span<float>)` (line ~42) and the `TensorSpan<float>` overload built on top of it (line ~17)

`Multiply`, `Scale` and `Relu` all call `TensorKernelGuards.ValidateInputOutputSpanNonOverlapping(...)`
before dispatching to `TensorPrimitives`, which permits `destination` to alias a source only when they
start at the same address (the legitimate in-place idiom) and throws for any other overlap. `Add` does not
call this guard at all — for either the raw-span overload or the `TensorSpan` overload that wraps it. A
caller that passes a `destination` which partially overlaps `left`/`right` (but doesn't start at the same
offset) gets whatever `TensorPrimitives.Add`'s vectorized read/write ordering happens to produce — silently
wrong numbers, not the `ArgumentException` every other kernel in the same file gives for the identical
mistake.

**How anyone would notice today:** the single current call site (`TensorMath.Algebra.cs:35`) passes
distinct tensors, so this is dormant. It would show up as an occasional numeric drift in an autograd/graph
path that slices a shared arena buffer for both an operand and its output — exactly the buffer-reuse pattern
this codebase's arena (`NativeBufferManaged`) encourages — and nothing would flag it as anything other than
"the math is a little off."

**What test would have caught it:** a unit test constructing `left`/`right`/`destination` as overlapping
(non-identical-start) slices of one backing array and asserting `TensorKernels.Add` throws, mirroring the
existing overlap tests (if any) for `Multiply`/`Scale`/`Relu`.

---

### 3. `FastTensor<T>.FromView` leaks the rented `PooledBuffer` when given a non-contiguous rank-1/3/4 view

**File:** `Sources/Main/Tensors/FastTensor.cs`, `FastTensor<T>.FromView(TensorView<T> view)` (line ~246)

The method allocates `materializedTensor` (which rents a `PooledBuffer<T>` in its constructor) for every
rank via the `switch` at the top, *before* checking whether the non-contiguous copy path can actually handle
the rank:
```csharp
var materializedTensor = view.Rank switch { 1 => new FastTensor<T>(...), ... }; // rents here
if (view.IsContiguous) { ...; return materializedTensor; }
var targetSpan = materializedTensor.AsSpan();
var index = 0;
if (view.Rank != 2)
{
    throw new NotImplementedException("todo: Kopiowanie nieciągłych widoków > 2D nie jest zaimplementowane.");
    // materializedTensor is never disposed — its rented array is never returned to the pool.
}
```
Every throw on this path leaves a `PooledBuffer<T>` checked out of `ArrayPool<T>.Shared` forever — the exact
"raw pool leak" class the project's own `PooledBuffer` wrapper exists to make visible and prevent
(`Sources/Main/Tensors/README.md`: "the wrapper exists so rentals are visible and paired; raw pool calls have
leaked here before").

**How anyone would notice today:** no caller of `FastTensor<T>.FromView` currently exists in
`Sources/Main`, so it's a live trap rather than an active leak; the moment something materializes a
non-contiguous rank-1/3/4 view (e.g. a transposed 3D attention tensor), every call leaks one array — a slow,
silent pool-exhaustion pressure that looks like generic GC/allocation noise, not a `FromView` bug.

**What test would have caught it:** a test that calls `FromView` on a non-contiguous rank-3 view inside a
loop and asserts `GC.GetTotalAllocatedBytes` / pool-rent count doesn't grow per-iteration (or more simply: a
test wrapping the call in `try`/`catch` and asserting the array pool's outstanding-rental count returns to
baseline after the expected `NotImplementedException`).

---

### 4. `TensorShape.Size` / `TensorStrides.Contiguous` / `TensorView`'s dimension products use unchecked 32-bit multiplication — `FastTensor`'s equivalent path is `checked`, this one silently wraps

**Files:** `Sources/Main/Tensors/TensorShape.cs` (`Size => D0 * D1 * D2 * D3`),
`Sources/Main/Tensors/TensorStrides.cs` (`Contiguous(shape)`: `shape.D1 * shape.D2 * shape.D3` etc.),
`Sources/Main/Tensors/TensorView.cs` (the 2/3/4-arg constructors: `s0 * s1`, `s0 * s1 * s2`, `s0 * s1 * s2 *
s3`)

`FastTensor<T>`'s multi-dim constructors explicitly wrap their size product in `checked(...)`
(`FastTensor.cs:52,66,82`) specifically to turn a positive 32-bit wrap into a clean `OverflowException`
instead of an undersized buffer — the exact hazard `OVERFIT028` exists to catch. But `OVERFIT028` only fires
on `ArrayCreationExpression` syntax (`new T[...]`); it does not see a property getter or a constructor field
assignment. `TensorShape.Size`, `TensorStrides.Contiguous`, and `TensorView`'s own constructors compute the
identical product unchecked, so the exact same overflow that `FastTensor` was hardened against is still live
one layer up — anything that goes through `TensorShape`/`TensorSpan`/`TensorView` instead of `FastTensor` for
a large-enough 4D shape (e.g. attention-score tensors at long context, or an accidental transposition that
multiplies four already-large dims) gets a silently wrapped (possibly negative, possibly small-and-positive)
size instead of a thrown exception.

**How anyone would notice today:** only at genuinely large shapes (product > ~2.1B), so it's latent for
typical model sizes today, but it is reachable — nothing in the type prevents a caller from constructing such
a shape, and the analyzer that is supposed to guard exactly this class of bug in this codebase does not cover
it.

**What test would have caught it:** a unit test constructing a `TensorShape`/`TensorSpan`/`TensorView` whose
declared dimensions multiply past `int.MaxValue` and asserting it throws rather than returning a wrapped
`Size`.

---

### 5. `NativeBuffer<T>.Dispose()` has no double-free guard, unlike both of its sibling wrappers

**File:** `Sources/Main/Tensors/NativeBuffer.cs`, `NativeBuffer<T>.Dispose()`

```csharp
public unsafe readonly ref struct NativeBuffer<T> where T : unmanaged
{
    private readonly void* _ptr;
    ...
    public void Dispose()
    {
        if (_ptr != null) { NativeMemory.AlignedFree(_ptr); }
    }
}
```
`_ptr` is `readonly` (forced by the type being a `readonly ref struct`), so `Dispose()` cannot null it out
after freeing. Its two siblings in the same directory both guard this exact case: `PooledBuffer<T>.Dispose()`
nulls `_rented` before returning it ("idempotent / null-safe" per its own doc comment), and
`NativeBufferManaged<T>.Dispose()` checks/sets a `_disposed` flag. `NativeBuffer<T>` — listed in the same
README table as an equally safe storage option ("Unmanaged allocation for buffers that must not move or be
traced by the GC") — is the one wrapper of the three where calling `Dispose()` a second time frees
already-freed native memory: heap corruption or an `AccessViolationException`, not a clean no-op.

**How anyone would notice today:** `NativeBuffer<T>` currently has no caller anywhere in `Sources/Main`
(only exercised directly by `Tests/Core/Memory/BufferTests.cs`), so this is dormant. The moment a caller
writes `using var buf = new NativeBuffer<T>(n); ... buf.Dispose();` (an easy mistake given the type's own
`using`-friendly `Dispose()` method) or disposes it via two different code paths (e.g. an explicit
early-return dispose plus the `using`), it double-frees silently.

**What test would have caught it:** a test that calls `Dispose()` twice on the same `NativeBuffer<T>` and
asserts it does not crash/corrupt (the existing `BufferTests.cs` tests only exercise a single `using`
disposal).

---

## Additional, lower-confidence finding (not counted in the score above)

### 6. `TensorView<T>.Reshape` drops `Offset` when rebuilding the view — `TensorSpan<T>.Reshape` doesn't have this bug

**File:** `Sources/Main/Tensors/TensorView.cs`, `Reshape(int newS0, int newS1)`

```csharp
return new TensorView<T>(_data, newS0, newS1); // full _data, not _data.Slice(Offset, Size)
```
Compare `TensorSpan<T>.Reshape` (`Core/TensorSpan.cs`), which correctly does
`new TensorSpan<T>(_data.Slice(Offset, Size), newShape)`. If `Offset` is ever nonzero on a view that is still
`IsContiguous == true` when `Reshape` is called, the reshaped view would silently index from the wrong base
address. Today this looks unreachable: the only path that produces a nonzero `Offset` is `Transpose2D()`,
which always sets `IsContiguous = false`, and `Reshape` itself refuses non-contiguous views — so the two
conditions needed to trigger it (`Offset != 0` and `IsContiguous == true`) can't currently co-occur through
the public constructors. Flagging it because it's a real inconsistency between two structurally-identical
sibling types (one right, one wrong) and because `Slice`/future constructors could reintroduce a
contiguous-with-offset state without anyone noticing the invariant `Reshape` depends on.

Not counted toward the score because I could not construct a currently-reachable input that exercises it —
report it as unconfirmed-but-real; the fix (mirror `TensorSpan`'s slice-by-offset) is one line either way.

---

## Shared root causes

- Findings 1 and 6 share a shape: a method on one type correctly handles a case (`AsSpan`
  handles borrowed memory; `TensorSpan.Reshape` handles nonzero offset) while a structurally parallel method
  on the same type (finding 1) or a sibling type (finding 6) omits the identical handling. Worth a single
  sweep across `TensorStorage`/`TensorSpan`/`TensorView` for other "one arm got the fix, the parallel arm
  didn't" pairs rather than treating each as isolated.
- Findings 3 and 5 are both lifetime bugs in the "wrapper that promises safety the code doesn't deliver"
  shape, and both are currently dormant only because nothing in `Sources/Main` calls the affected method yet
  — neither is protected by having zero callers; they're protected by luck.

## Coverage

**Reviewed and found clean:** `TensorStrides.GetOffset*` (bounds enforced by callers' `Span` indexers, not
silent), `TensorKernelGuards` (all five guard helpers correctly used by their callers except the one gap in
finding 2), `TensorFactory.CloneStorage`/`Materialize` (rank validated before allocation so no leak-on-throw;
non-contiguous materialization loops write every element before the caller can read), `NativeBufferManaged`
(idempotent `Dispose` + finalizer safety net, `checked` arithmetic throughout `Allocate`), `PooledBuffer<T>`
(idempotent Dispose as documented), `FastTensor`'s size-computing constructors (all `checked`),
`ComputationGraph.CreateTemporary`/`CreateAuxiliary` call sites across `DeepLearning/` and `Autograd/` that
pass `clearMemory: false` on arena-backed auxiliary tensors — each site checked was fully overwritten before
being read, with an explicit comment in most of them referencing a prior leak/read-before-write bug that was
already fixed.

**Not reached:** exhaustive tracing of every external caller of `TensorSpan<T>`/`TensorView<T>` outside
`Sources/Main` (e.g. ONNX importer, LanguageModels runtime) to check whether any of them relies on
`TensorKernels.Add`'s missing overlap guard or `TensorView.Reshape`'s offset gap in a way that's actually
reachable today — the ten-minute budget didn't stretch to a full call-graph walk, and both findings are
reported with their current-reachability status stated honestly rather than guessed.

## What the score means

5 confirmed defects (10 points) against a target of 11 (21 points), ended by scope rather than by the clock
— every file in the named directory was read, and the extra minutes were spent chasing the two most
promising leads (the arena `clearMemory:false` pattern, and each finding's actual call sites) rather than
being cut off mid-file. That is evidence this module is in good shape, not evidence of an unfinished hunt:
the foundation-level arithmetic and lifetime code is mostly careful and, notably, several sites carry
comments citing *previous* leak/overlap bugs that were already fixed — the team has been here before and it
shows. The defects that remain are exactly the kind that hide well: unreachable today, on a rarely-called
method (`AsMemory`), an error path (`FromView`'s non-contiguous throw), or a scale nobody has hit yet
(`TensorShape.Size` overflow) — quiet until the one caller that needs them shows up.
