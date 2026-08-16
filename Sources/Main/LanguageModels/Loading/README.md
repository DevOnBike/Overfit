# `LanguageModels/Loading` — reading model files, without Python

Every weight format Overfit understands is parsed here, in C#, with no Python, no ONNX Runtime and no
native dependency. `GgufReader` / `GgufLlamaLoader` for GGUF, `SafetensorsReader` and its sharded
variant for safetensors, plus the GPT-2 `.bin` path. `MemoryMappedModelFile` maps rather than reads
where it can.

## Direction is one-way

External format → Overfit. **There are no exporters and none are planned.** A request to convert an
Overfit model into GGUF or safetensors is out of scope, not a gap.

## Two conventions that produce fluent nonsense when confused

**RoPE pair layout.** HuggingFace rotates halves of the head dimension; GGUF rotates adjacent pairs.
The loaders permute rows at load time so the runtime only ever sees one convention. Get this wrong and
the model still generates confident, grammatical, meaningless text — there is no crash to find.

**Fused weights.** Phi-3 ships a fused QKV and a fused gate/up; Qwen-3 needs an explicit `head_dim` and
per-head QK-RMSNorm applied *before* RoPE; Gemma-2 scales embeddings by √d and applies both soft-caps.
Each of these is a per-architecture claim living in the loader, and each was found by output that was
subtly wrong rather than by an exception.

## Peak memory, not steady-state memory

The target includes low-end hardware, so the load path is written to minimise **peak** RAM, not just
what is retained afterwards. Concretely: prefer unpooled allocation for weights that live forever
(pooling a permanent buffer only fragments the pool), and do not stage a file through a scratch
`byte[]` when it can be read into its destination. `PooledBuffer<T>` in `../../Tensors` is the
`using`-shaped wrapper the `try/finally` rental sites here were swept onto.

`RepackedWeightsFile` reads the `*.gguf.repack` sidecar that the decode kernels use; note that its
presence short-circuits some kernel selection flags, which has invalidated an A/B before.
