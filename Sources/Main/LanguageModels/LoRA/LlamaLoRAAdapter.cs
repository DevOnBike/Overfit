// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    /// <summary>
    /// LoRA adapter for Llama/Qwen-family models (GQA architecture).
    ///
    /// Strategy: weight-merge. Enable() adds delta to TensorStorage in-place;
    /// Disable() subtracts it. Zero changes to inference kernels required.
    ///
    ///   W_eff = W_base + scale * (A @ B)
    ///
    /// Create via <see cref="LanguageModels.Runtime.CachedLlamaInferenceEngine.CreateLoRAAdapter"/>.
    ///
    /// Typical inference usage:
    ///   var adapter = engine.CreateLoRAAdapter("my-lora", options);
    ///   adapter.Load("my-lora.bin");
    ///   adapter.Enable();
    ///   session.Reset(prompt);
    ///   session.GenerateNextToken(in sampling);   // uses merged weights
    ///   adapter.Disable();
    ///
    /// Adapter file format (binary):
    ///   Magic    : uint32 = 0x524F4C4F ("LORA")
    ///   Version  : int32  = 1
    ///   Count    : int32  = number of LoRAWeight entries
    ///   For each:
    ///     LayerIdx  : int32
    ///     Module    : int32 (LoRATargetModules enum value)
    ///     HeadIdx   : int32
    ///     LoRAWeight: (inDim, outDim, rank, A[], B[])
    /// </summary>
    public sealed class LlamaLoRAAdapter : ILoRAAdapter
    {
        private const uint FileMagic = 0x524F4C4Fu;  // "LLOR"
        private const int FileVersion = 1;

        // key = (layerIdx, module, headOrKvIdx)
        private readonly Dictionary<(int, LoRATargetModules, int), LoRAWeight> _weights;
        private readonly Dictionary<(int, LoRATargetModules, int), TensorStorage<float>> _baseRefs;
        private readonly float[] _deltaScratch;

        private bool _disposed;
        private bool _enabled;

        /// <summary>Internal constructor — called by CachedLlamaInferenceEngine.CreateLoRAAdapter.</summary>
        internal LlamaLoRAAdapter(
            string name,
            LoRAOptions options,
            int nLayers,
            int nHeads,
            int nKvHeads,
            int dModel,
            int headDim,
            int dFF,
            Dictionary<(int, LoRATargetModules, int), TensorStorage<float>> baseWeightRefs)
        {
            Name = name;
            Options = options;
            _baseRefs = baseWeightRefs;
            _weights = new Dictionary<(int, LoRATargetModules, int), LoRAWeight>(
                nLayers * CountTargetedModules(options.TargetModules, nHeads, nKvHeads));

            var maxDelta = Math.Max(dModel * headDim,
                           Math.Max(dModel * dFF, dFF * dModel));
            _deltaScratch = new float[maxDelta];

            BuildWeights(options, nLayers, nHeads, nKvHeads, dModel, headDim, dFF);
        }

        // ── ILoRAAdapter ──────────────────────────────────────────────────────

        public string Name { get; }
        public LoRAOptions Options { get; }
        public bool IsEnabled => _enabled;

        public long TrainableParameterCount
        {
            get
            {
                var total = 0L;
                foreach (var w in _weights.Values)
                {
                    total += w.ParameterCount;
                }
                return total;
            }
        }

        /// <summary>
        /// Merges LoRA into base weights. After this call inference uses W + scale*(A@B).
        /// Idempotent — calling twice is a no-op.
        /// </summary>
        public void Enable()
        {
            ThrowIfDisposed();
            if (_enabled)
            {
                return;
            }
            ApplyDelta(+Options.Scale);
            _enabled = true;
        }

        /// <summary>
        /// Removes LoRA delta, restoring exact base weights.
        /// Safe to call when already disabled.
        /// </summary>
        public void Disable()
        {
            ThrowIfDisposed();
            if (!_enabled)
            {
                return;
            }
            ApplyDelta(-Options.Scale);
            _enabled = false;
        }

        public void ZeroGrad()
        {
            foreach (var w in _weights.Values)
            {
                w.ZeroGrad();
            }
        }

        // ── Forward (non-merged, for training) ───────────────────────────────

        /// <summary>
        /// Adds LoRA contribution for one projection during a training forward pass.
        /// Call AFTER the frozen base projection:
        ///   result (base output) += scale * B^T @ (A^T @ x)
        /// </summary>
        public void ForwardAdd(
            int layer,
            LoRATargetModules module,
            int headIdx,
            ReadOnlySpan<float> x,
            Span<float> result)
        {
            if (_weights.TryGetValue((layer, module, headIdx), out var w))
            {
                w.ForwardAdd(x, result, Options.Scale);
            }
        }

        // ── Save / Load ───────────────────────────────────────────────────────

        /// <summary>
        /// Saves A and B matrices for all LoRA weights to a binary file.
        /// Can be loaded into any adapter with matching architecture + LoRAOptions.
        /// </summary>
        public void Save(string path)
        {
            ThrowIfDisposed();
            using var fs = File.OpenWrite(path);
            using var bw = new BinaryWriter(fs);

            bw.Write(FileMagic);
            bw.Write(FileVersion);
            bw.Write(_weights.Count);

            foreach (var ((layer, module, headIdx), w) in _weights)
            {
                bw.Write(layer);
                bw.Write((int)module);
                bw.Write(headIdx);
                w.Save(bw);
            }
        }

        /// <summary>
        /// Loads A and B matrices from a saved file.
        /// The adapter must be disabled before loading.
        /// </summary>
        public void Load(string path)
        {
            ThrowIfDisposed();
            if (_enabled)
            {
                throw new OverfitRuntimeException(
                "Call Disable() before loading new LoRA weights.");
            }

            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);

            var magic = br.ReadUInt32();
            if (magic != FileMagic)
            {
                throw new OverfitFormatException(
                $"Not a LoRA file (magic={magic:#x}, expected {FileMagic:#x}).");
            }

            var version = br.ReadInt32();
            if (version != FileVersion)
            {
                throw new OverfitFormatException(
                $"Unsupported LoRA file version {version} (expected {FileVersion}).");
            }

            var count = br.ReadInt32();
            for (var i = 0; i < count; i++)
            {
                var layer = br.ReadInt32();
                var module = (LoRATargetModules)br.ReadInt32();
                var headIdx = br.ReadInt32();
                var key = (layer, module, headIdx);
                var loaded = LoRAWeight.Load(br);

                if (!_weights.TryGetValue(key, out var dst))
                {
                    continue;
                }

                if (loaded.InDim != dst.InDim || loaded.OutDim != dst.OutDim ||
                    loaded.Rank != dst.Rank)
                {
                    throw new OverfitFormatException(
                    $"Dimension mismatch at ({layer},{module},{headIdx}): " +
                    $"file=[{loaded.InDim}×{loaded.OutDim} r={loaded.Rank}], " +
                    $"adapter=[{dst.InDim}×{dst.OutDim} r={dst.Rank}]");
                }

                loaded.A.CopyTo(dst.AMutable);
                loaded.B.CopyTo(dst.BMutable);
            }
        }

        // ── IDisposable ───────────────────────────────────────────────────────

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            if (_enabled)
            {
                Disable();
            }
            _disposed = true;
        }

        // ── Private ───────────────────────────────────────────────────────────

        private void BuildWeights(
            LoRAOptions options,
            int nLayers, int nHeads, int nKvHeads,
            int dModel, int headDim, int dFF)
        {
            var rank = options.Rank;
            var mods = options.TargetModules;

            for (var l = 0; l < nLayers; l++)
            {
                if (mods.HasFlag(LoRATargetModules.Query))
                {
                    for (var h = 0; h < nHeads; h++)
                    {
                        Add(l, LoRATargetModules.Query, h,
                        new LoRAWeight(dModel, headDim, rank));
                    }
                }

                if (mods.HasFlag(LoRATargetModules.Key))
                {
                    for (var kv = 0; kv < nKvHeads; kv++)
                    {
                        Add(l, LoRATargetModules.Key, kv,
                        new LoRAWeight(dModel, headDim, rank));
                    }
                }

                if (mods.HasFlag(LoRATargetModules.Value))
                {
                    for (var kv = 0; kv < nKvHeads; kv++)
                    {
                        Add(l, LoRATargetModules.Value, kv,
                        new LoRAWeight(dModel, headDim, rank));
                    }
                }

                if (mods.HasFlag(LoRATargetModules.OutputProjection))
                {
                    for (var h = 0; h < nHeads; h++)
                    {
                        Add(l, LoRATargetModules.OutputProjection, h,
                        new LoRAWeight(headDim, dModel, rank));
                    }
                }

                if (mods.HasFlag(LoRATargetModules.FeedForwardUp))
                {
                    Add(l, LoRATargetModules.FeedForwardUp, 0,
                    new LoRAWeight(dModel, dFF, rank));
                }

                if (mods.HasFlag(LoRATargetModules.FeedForwardDown))
                {
                    Add(l, LoRATargetModules.FeedForwardDown, 0,
                    new LoRAWeight(dFF, dModel, rank));
                }
            }
        }

        private void Add(int l, LoRATargetModules m, int h, LoRAWeight w)
            => _weights[(l, m, h)] = w;

        /// <summary>
        /// Number of (layer, module, headIdx) entries that matched between
        /// _weights and _baseRefs during the last Enable/Disable call.
        /// Should equal TrainableParameterCount / (InDim+OutDim) / Rank.
        /// If 0: _baseRefs keys don't match _weights keys.
        /// </summary>
        public int LastApplyMatchCount { get; private set; }

        /// <summary>Number of entries in _baseRefs (should equal NLayers * targeted modules).</summary>
        public int BaseRefCount => _baseRefs.Count;

        /// <summary>
        /// Reads a single value from the base weight TensorStorage (for diagnostics).
        /// Returns NaN if the key is not in _baseRefs.
        /// </summary>
        public float ReadBaseWeight(int layer, LoRATargetModules module, int head, int index)
        {
            var key = (layer, module, head);
            if (!_baseRefs.TryGetValue(key, out var storage))
            {
                return float.NaN;
            }
            var span = storage.AsSpan();
            if (index < 0 || index >= span.Length)
            {
                return float.NaN;
            }
            return span[index];
        }

        /// <summary>L2 norm of the storage TensorStorage at given key (for diagnostics).</summary>
        public float ReadBaseWeightNorm(int layer, LoRATargetModules module, int head)
        {
            var key = (layer, module, head);
            if (!_baseRefs.TryGetValue(key, out var storage))
            {
                return float.NaN;
            }
            var sumSq = 0f;
            foreach (var v in storage.AsSpan())
            {
                sumSq += v * v;
            }
            return MathF.Sqrt(sumSq);
        }

        private void ApplyDelta(float scaleFactor)
        {
            var matched = 0;
            foreach (var ((layer, module, headIdx), w) in _weights)
            {
                var key = (layer, module, headIdx);
                if (!_baseRefs.TryGetValue(key, out var storage))
                {
                    continue;
                }

                matched++;
                var size = w.InDim * w.OutDim;
                var delta = _deltaScratch.AsSpan(0, size);
                w.ComputeDelta(delta);

                var target = storage.AsSpan().Slice(0, size);
                TensorPrimitives.MultiplyAdd(delta, scaleFactor, target, target);
            }
            LastApplyMatchCount = matched;
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
            {
                throw new ObjectDisposedException(nameof(LlamaLoRAAdapter));
            }
        }

        private static int CountTargetedModules(
            LoRATargetModules mods, int nHeads, int nKvHeads)
        {
            var n = 0;
            if (mods.HasFlag(LoRATargetModules.Query))
            {
                n += nHeads;
            }
            if (mods.HasFlag(LoRATargetModules.Key))
            {
                n += nKvHeads;
            }
            if (mods.HasFlag(LoRATargetModules.Value))
            {
                n += nKvHeads;
            }
            if (mods.HasFlag(LoRATargetModules.OutputProjection))
            {
                n += nHeads;
            }
            if (mods.HasFlag(LoRATargetModules.FeedForwardUp))
            {
                n += 1;
            }
            if (mods.HasFlag(LoRATargetModules.FeedForwardDown))
            {
                n += 1;
            }
            return n;
        }
    }
}
