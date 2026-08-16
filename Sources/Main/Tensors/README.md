# `Tensors` — storage, shapes and views

Where the numbers actually live. Everything above this directory eventually resolves to a
`Span<float>` over one of these buffers.

## Storage

| Type | Use |
|---|---|
| `TensorStorage<T>` (in `Core`) | Owned, poolable backing store; the normal choice. |
| `PooledBuffer<T>` | Scoped rental: `using var b = new PooledBuffer<T>(n, clearMemory: false)`. |
| `NativeBuffer` | Unmanaged allocation for buffers that must not move or be traced by the GC. |
| `FastTensor` | The general tensor: storage plus shape plus strides. |
| `TensorView` | A window onto someone else's storage. Owns nothing and disposes nothing. |

**`ArrayPool<T>.Shared` is banned** (`RS0030`, build error). Use `PooledBuffer<T>` for scoped
lifetimes, and for class-lifetime buffers hold one in a field — rented in the constructor, disposed by
the owner's `Dispose` (`TensorStorage<T>` and `FastTensor<T>` are the two examples). The wrapper exists
so rentals are visible and paired; raw pool calls have leaked here before. `OverfitPool<T>`, an earlier attempt at a
custom pool, was **deleted** after measuring 3× to ~3000× slower than `ArrayPool` — a negative result
worth keeping, since the idea looks obviously good.

**Jagged `float[][]` is banned in this project** (`OVERFIT033`, build error). Use a flat `float[]`
sliced per row — one allocation, cache-friendly — or one of the buffers above. `int[][]` and
`Parameter[][]` are unaffected.

## Shapes

`TensorShape` and `TensorStrides` are value types describing layout without touching data, so
reshaping, transposing and slicing are arithmetic rather than copies. `Array.Copy` is banned; use
`Span<T>.CopyTo`, which the JIT lowers better and which cannot silently box.

`TensorKernelGuards` holds the shape checks the kernels rely on, in one place, so a wrong shape fails
with a sentence rather than reading past the end of a buffer.
