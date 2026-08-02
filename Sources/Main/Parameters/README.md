# `Parameters` — one type, on purpose

`Parameter` is the only thing here, and a directory holding one type invites the question of why it is not
somewhere else. This file is the answer, so that the arrangement is a decision rather than a leftover.

## What the type is

A trainable tensor: storage, shape, an optional gradient, and the operations that only make sense for
something an optimiser steps — `ZeroGrad`, `LoadData`, `Save`/`Load`, and `AsNode()`, which is the bridge onto
the autograd tape.

It is the boundary between two halves of the engine that otherwise do not know about each other. A layer owns
`Parameter`s and enumerates them through `TrainableParameters()`; an optimiser takes
`IEnumerable<Parameter>` and never learns what a layer is. Neither side owns the concept, which is why it
does not live in either.

## Why it is not moved into `Autograd` or `Tensors`

Two reasons, and the second is the binding one.

**It is not a tensor and it is not a tape node.** `TensorStorage<float>` knows nothing about gradients;
`AutogradNode` is a tape entry with an ownership tag whose lifetime is a graph reset. A `Parameter` outlives
every graph, is disposed by the layer that owns it, and is the only thing in the system an optimiser is
allowed to mutate. Putting it under either neighbour would attach a long-lived concept to a short-lived one.

**Moving it is a breaking change to the public API for no functional gain.** `Parameter` appears in sixteen
public signatures — both optimiser constructors and `TrainableParameters()` on a dozen layers — so the
namespace is part of the contract, and C# has no way to soften that: `TypeForwardedTo` works across
assemblies, not across namespaces inside one. Every consumer with `using DevOnBike.Overfit.Parameters;`
breaks, and what they get for it is one fewer using directive. Under this repository's versioning policy that
is a MINOR bump, and it would have to be worth one on its own.

## What would change this

If a second concept genuinely arrives — parameter groups with per-group learning rates, an initialiser
family, a frozen/trainable partition that is more than a `bool` — this becomes an ordinary directory and the
question disappears. Until then the plural name is the only thing overstating the case, and renaming a
namespace costs exactly as much as moving one.

**Do not move it as tidying.** If it moves, it moves bundled with other breaking changes in a release that is
already taking a MINOR bump, and with a note in `CHANGELOG.md` naming the namespace consumers must update.
