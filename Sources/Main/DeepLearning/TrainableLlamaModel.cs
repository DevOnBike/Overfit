// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.LanguageModels.Rope;
using DevOnBike.Overfit.LanguageModels.Runtime;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// The frozen quantized weights of one decoder layer for <see cref="TrainableLlamaModel"/> — the
    /// combined-tensor projections plus the two RMSNorm gain initializers (copied into trainable γ).
    /// </summary>
    public sealed class LlamaLayerFrozenWeights
    {
        public required IDequantRowSource Wq { get; init; }
        public required IDequantRowSource Wk { get; init; }
        public required IDequantRowSource Wv { get; init; }
        public required IDequantRowSource Wo { get; init; }
        public required IDequantRowSource Gate { get; init; }
        public required IDequantRowSource Up { get; init; }
        public required IDequantRowSource Down { get; init; }
        public required float[] Ln1GammaInit { get; init; }
        public required float[] Ln2GammaInit { get; init; }
    }

    /// <summary>
    /// A full <b>trainable</b> Llama/Qwen language model assembled on the autograd graph: frozen
    /// quantized token embedding → N <see cref="TrainableLlamaBlock"/>s (each with optional LoRA, run
    /// under gradient checkpointing) → trainable final RMSNorm → frozen quantized LM head → logits.
    /// This is the QLoRA fine-tuner: every projection + embedding + LM head stays frozen 4-bit, only the
    /// LoRA adapters and RMSNorm gains train. Built either from explicit weights (tests) or straight from
    /// a loaded <see cref="CachedLlamaInferenceEngine"/> (<see cref="FromEngine"/>) — the GGUF→training
    /// bridge end-to-end. Scope: batch = 1 (one sequence per step), causal.
    /// </summary>
    public sealed class TrainableLlamaModel : IDisposable
    {
        private readonly int _dModel, _nQHeads, _nKVHeads, _dHead, _halfDim, _dFF, _vocab;
        private readonly float _eps;
        private readonly bool _splitHalf;
        private readonly DecodeWeight _embed;
        private readonly IDequantRowSource _lmHead;
        private readonly TrainableLlamaBlock[] _blocks;
        private readonly AutogradNode[] _ln1Gamma;
        private readonly AutogradNode[] _ln2Gamma;
        private readonly LlamaBlockLoRA?[] _lora;
        private readonly LoRAAdapter? _lmHeadLora;
        private readonly AutogradNode _finalNormGamma;
        private readonly RopeTable _ropeTable;
        private readonly List<IDisposable> _owned = new();
        private readonly List<IDisposable> _scratch = new();

        public TrainableLlamaModel(
            int dModel, int nQHeads, int nKVHeads, int vocab,
            DecodeWeight embed, IDequantRowSource lmHead,
            LlamaLayerFrozenWeights[] layers,
            float[] finalNormInit,
            float ropeTheta, bool ropeSplitHalf, float eps, int maxSeqLen, RopeScaling? ropeScaling,
            int loraRank, Random rng, bool loraOnLmHead = false)
        {
            _dModel = dModel; _nQHeads = nQHeads; _nKVHeads = nKVHeads; _vocab = vocab;
            _dHead = dModel / nQHeads; _halfDim = _dHead / 2; _eps = eps; _splitHalf = ropeSplitHalf;
            _embed = embed; _lmHead = lmHead;
            _dFF = layers[0].Gate.OutputSize;

            var nLayers = layers.Length;
            _blocks = new TrainableLlamaBlock[nLayers];
            _ln1Gamma = new AutogradNode[nLayers];
            _ln2Gamma = new AutogradNode[nLayers];
            _lora = new LlamaBlockLoRA?[nLayers];

            for (var l = 0; l < nLayers; l++)
            {
                var lw = layers[l];
                _blocks[l] = new TrainableLlamaBlock(
                    dModel, nQHeads, nKVHeads, lw.Wq, lw.Wk, lw.Wv, lw.Wo, lw.Gate, lw.Up, lw.Down,
                    eps, ropeSplitHalf);
                _ln1Gamma[l] = Param(lw.Ln1GammaInit);
                _ln2Gamma[l] = Param(lw.Ln2GammaInit);
                _lora[l] = loraRank > 0
                    ? Track(LlamaBlockLoRA.CreateAll(dModel, nQHeads * _dHead, nKVHeads * _dHead, _dFF, loraRank, rng))
                    : null;
            }

            _finalNormGamma = Param(finalNormInit);
            // LoRA on the LM head is opt-in: it adds direct output capacity but on a huge vocab it is
            // high-variance, so it is OFF by default (block LoRA + norms give a stable descent).
            _lmHeadLora = (loraRank > 0 && loraOnLmHead) ? Track(new LoRAAdapter(dModel, vocab, loraRank, rng)) : null;
            _ropeTable = new RopeTable(maxSeqLen, _dHead, ropeTheta, ropeScaling, ropeSplitHalf);
        }

        public int LayerCount => _blocks.Length;
        public int VocabSize => _vocab;

        /// <summary>Builds a trainable QLoRA model straight from a loaded quantized GGUF engine: every
        /// projection + embedding + LM head is the engine's frozen quantized handle (zero repack), with
        /// fresh LoRA adapters of the given rank and RMSNorm gains copied from the model.</summary>
        public static TrainableLlamaModel FromEngine(
            CachedLlamaInferenceEngine engine, int loraRank, Random rng, int maxSeqLen, bool loraOnLmHead = false)
        {
            var cfg = engine.Config;
            int dModel = cfg.DModel, nHeads = cfg.NHeads, kvHeads = cfg.KvHeads, headDim = dModel / nHeads;

            var layers = new LlamaLayerFrozenWeights[cfg.NLayers];
            for (var l = 0; l < cfg.NLayers; l++)
            {
                var lw = engine.GetTrainableLayer(l);
                layers[l] = new LlamaLayerFrozenWeights
                {
                    Wq = ConcatRows(lw.Wq, nHeads),
                    Wk = ConcatRows(lw.Wk, kvHeads),
                    Wv = ConcatRows(lw.Wv, kvHeads),
                    Wo = ConcatCols(lw.Wo, nHeads),
                    Gate = lw.FfnGate.AsRowSource(),
                    Up = lw.FfnUp.AsRowSource(),
                    Down = lw.FfnDown.AsRowSource(),
                    Ln1GammaInit = lw.AttnNormGamma.AsReadOnlySpan().ToArray(),
                    Ln2GammaInit = lw.FfnNormGamma.AsReadOnlySpan().ToArray(),
                };
            }

            return new TrainableLlamaModel(
                dModel, nHeads, kvHeads, cfg.VocabSize,
                engine.EmbeddingWeights, engine.LmHeadWeights.AsRowSource(),
                layers, engine.FinalNormGamma.AsReadOnlySpan().ToArray(),
                cfg.RoPETheta, cfg.RopeSplitHalf, 1e-6f, maxSeqLen, cfg.RopeScaling, loraRank, rng, loraOnLmHead);
        }

        /// <summary>Forward over a token sequence → logits <c>[T, vocab]</c>, recording the tape.
        /// <paramref name="useCheckpoint"/> runs each block under gradient checkpointing (≈2 blocks of
        /// activations live at once instead of all N) — the memory lever for big models.</summary>
        public AutogradNode Forward(ComputationGraph graph, int[] tokenIds, bool useCheckpoint)
        {
            DisposeScratch();
            var T = tokenIds.Length;

            // ── frozen token embedding lookup → [T, dModel] (no gradient to the table) ──
            var embStore = new TensorStorage<float>(T * _dModel, clearMemory: false);
            _scratch.Add(embStore);
            for (var t = 0; t < T; t++)
            {
                _embed.DequantizeRow(tokenIds[t], embStore.AsSpan().Slice(t * _dModel, _dModel));
            }
            // requiresGrad: true so checkpointing has a grad sink for its input node; the embedding
            // table itself is never updated (it is not an optimizer parameter).
            var hidden = new AutogradNode(embStore, new TensorShape(T, _dModel), requiresGrad: true);
            _scratch.Add(hidden);

            // ── RoPE cos/sin for positions 0..T-1 (constants) ──
            var cosStore = new TensorStorage<float>(T * _halfDim, clearMemory: false);
            var sinStore = new TensorStorage<float>(T * _halfDim, clearMemory: false);
            for (var t = 0; t < T; t++)
            {
                _ropeTable.CosAt(t).CopyTo(cosStore.AsSpan().Slice(t * _halfDim, _halfDim));
                _ropeTable.SinAt(t).CopyTo(sinStore.AsSpan().Slice(t * _halfDim, _halfDim));
            }
            var cosN = new AutogradNode(cosStore, new TensorShape(T, _halfDim), requiresGrad: false);
            var sinN = new AutogradNode(sinStore, new TensorShape(T, _halfDim), requiresGrad: false);
            _scratch.Add(cosStore); _scratch.Add(sinStore); _scratch.Add(cosN); _scratch.Add(sinN);

            var subArena = CheckpointArena(T);
            var h = hidden;
            for (var l = 0; l < _blocks.Length; l++)
            {
                var li = l; // capture a per-iteration copy for the deferred checkpoint recompute
                var block = _blocks[li];
                var g1 = _ln1Gamma[li]; var g2 = _ln2Gamma[li]; var lora = _lora[li];
                h = useCheckpoint
                    ? graph.Checkpoint((sub, x) => block.Forward(sub, x, cosN, sinN, g1, g2, lora), h, subArena)
                    : block.Forward(graph, h, cosN, sinN, g1, g2, lora);
            }

            var normed = TensorMath.RmsNorm(graph, h, _finalNormGamma, _eps);
            var logits = graph.FrozenQuantizedLinear(normed, _lmHead); // [T, vocab]
            return _lmHeadLora is null ? logits : graph.Add(logits, _lmHeadLora.Apply(graph, normed));
        }

        /// <summary>Greedy autoregressive generation from a prompt (no KV cache — recomputes the full
        /// sequence each step, fine for short demo generations). Returns the newly generated tokens,
        /// stopping at <paramref name="eosTokenId"/> or <paramref name="maxNewTokens"/>.</summary>
        public int[] Generate(ComputationGraph graph, int[] promptTokens, int maxNewTokens, int eosTokenId)
        {
            var tokens = new List<int>(promptTokens);
            var produced = new List<int>();
            for (var i = 0; i < maxNewTokens; i++)
            {
                graph.Reset();
                // Checkpointed forward: the per-block recompute runs without recording, so generation
                // allocates no gradient buffers and keeps the main arena tiny (only the layer outputs).
                var logits = Forward(graph, tokens.ToArray(), useCheckpoint: true);
                var T = tokens.Count;
                var lastRow = logits.DataView.AsReadOnlySpan().Slice((T - 1) * _vocab, _vocab);
                var next = 0; var bv = lastRow[0];
                for (var v = 1; v < _vocab; v++) { if (lastRow[v] > bv) { bv = lastRow[v]; next = v; } }
                if (next == eosTokenId) { break; }
                tokens.Add(next);
                produced.Add(next);
            }
            return produced.ToArray();
        }

        /// <summary>
        /// FAST greedy generation with a KV cache — processes only each new token through the layers
        /// (attending its query over the cache of all earlier positions) instead of re-running the whole
        /// sequence per token like <see cref="Generate"/>. No autograd, no graph. Same greedy result as
        /// <see cref="Generate"/>; the speedup grows with sequence length (O(n) work per token vs O(n²)
        /// total). Uses naive per-row dequant kernels — for llama.cpp-class speed see ROADMAP "Fast
        /// fine-tuned decode" (LoRA on the optimized inference engine).
        /// </summary>
        public int[] GenerateCached(int[] promptTokens, int maxNewTokens, int eosTokenId)
        {
            var nL = _blocks.Length;
            var kvWidth = _nKVHeads * _dHead;
            var maxLen = promptTokens.Length + maxNewTokens;

            // Per-layer KV caches as single CONTIGUOUS buffers (layer-major), not jagged — one allocation,
            // cache-friendly. Layer l's slice is [l*kvStride, kvStride) with kvStride = maxLen*kvWidth.
            var kvStride = maxLen * kvWidth;
            var cacheK = new float[nL * kvStride];
            var cacheV = new float[nL * kvStride];

            var tokens = new List<int>(promptTokens);
            var produced = new List<int>();

            using var hiddenB = new PooledBuffer<float>(_dModel, clearMemory: false);
            using var nextB = new PooledBuffer<float>(_dModel, false);
            using var normedB = new PooledBuffer<float>(_dModel, false);
            using var rowB = new PooledBuffer<float>(_dModel, false);
            var hidden = hiddenB.Span;
            var next = nextB.Span;

            var p = 0;
            while (true)
            {
                _embed.DequantizeRow(tokens[p], hidden);
                for (var l = 0; l < nL; l++)
                {
                    _blocks[l].DecodeStep(hidden, p, _ropeTable, _ln1Gamma[l], _ln2Gamma[l], _lora[l],
                        cacheK.AsSpan(l * kvStride, kvStride), cacheV.AsSpan(l * kvStride, kvStride), next);
                    next.CopyTo(hidden);
                }

                if (p == tokens.Count - 1)
                {
                    var token = LmHeadArgmax(hidden, normedB.Span, rowB.Span);
                    if (token == eosTokenId) { break; }
                    produced.Add(token);
                    if (produced.Count >= maxNewTokens) { break; }
                    tokens.Add(token);
                }
                p++;
            }
            return produced.ToArray();
        }

        private int LmHeadArgmax(ReadOnlySpan<float> hidden, Span<float> normed, Span<float> row)
        {
            var inv = 1f / MathF.Sqrt(TensorPrimitives.Dot(hidden, hidden) / _dModel + _eps);
            var g = _finalNormGamma.DataView.AsReadOnlySpan();
            for (var i = 0; i < _dModel; i++) { normed[i] = hidden[i] * inv * g[i]; }

            // LoRA pre-projection tmp[r] = Σ_i normed[i]·A[i,r], if an LM-head adapter is present.
            var rank = _lmHeadLora?.Rank ?? 0;
            ReadOnlySpan<float> bSpan = default;
            Span<float> tmp = stackalloc float[Math.Max(1, rank)];
            tmp.Clear();
            if (_lmHeadLora is not null)
            {
                var a = _lmHeadLora.A.DataView.AsReadOnlySpan();
                for (var i = 0; i < _dModel; i++)
                {
                    var xi = normed[i];
                    if (xi == 0f) { continue; }
                    var aRow = a.Slice(i * rank, rank);
                    for (var r = 0; r < rank; r++) { tmp[r] += xi * aRow[r]; }
                }
                bSpan = _lmHeadLora.B.DataView.AsReadOnlySpan(); // [rank, vocab]
            }

            int best = 0;
            var bestVal = float.NegativeInfinity;
            for (var o = 0; o < _vocab; o++)
            {
                _lmHead.DecodeRow(o, row);
                var logit = TensorPrimitives.Dot(row, normed);
                if (_lmHeadLora is not null)
                {
                    for (var r = 0; r < tmp.Length; r++) { logit += tmp[r] * bSpan[r * _vocab + o]; }
                }
                if (logit > bestVal) { bestVal = logit; best = o; }
            }
            return best;
        }

        /// <summary>All trainable parameters: every block's two RMSNorm gains + its LoRA adapters, plus
        /// the final RMSNorm gain. Feed to <c>Adam</c>.</summary>
        public IEnumerable<AutogradNode> TrainableParameters()
        {
            for (var l = 0; l < _blocks.Length; l++)
            {
                yield return _ln1Gamma[l];
                yield return _ln2Gamma[l];
                if (_lora[l] is { } lora)
                {
                    foreach (var p in lora.Parameters()) { yield return p; }
                }
            }
            yield return _finalNormGamma;
            if (_lmHeadLora is not null) { yield return _lmHeadLora.A; yield return _lmHeadLora.B; }
        }

        // ── Adapter save / load (the trained delta only; the frozen base is never written) ──

        private const uint AdapterMagic = 0x4F4C5241u; // "ARLO" — Overfit trainable-Llama adapter
        private const int AdapterVersion = 1;

        /// <summary>
        /// Saves ONLY the trained adapter — every LoRA A/B matrix + RMSNorm gain — to a small binary file
        /// (the frozen 4-bit base is never written; reload it from the original GGUF). For Qwen-3B rank-8
        /// this is tens of MB vs the 2 GB base. Round-trips into a model built with the SAME config via
        /// <see cref="LoadAdapter"/>.
        /// </summary>
        public void SaveAdapter(string path)
        {
            using var fs = File.Create(path);
            using var bw = new BinaryWriter(fs);
            bw.Write(AdapterMagic);
            bw.Write(AdapterVersion);
            bw.Write(_blocks.Length);
            bw.Write(_dModel);
            bw.Write(_vocab);

            var ps = MaterializeParams();
            bw.Write(ps.Count);
            foreach (var p in ps)
            {
                var s = p.DataView.AsReadOnlySpan();
                bw.Write(s.Length);
                bw.Write(MemoryMarshal.AsBytes(s));
            }
        }

        /// <summary>
        /// Loads an adapter saved by <see cref="SaveAdapter"/> into this model's trainable parameters
        /// (must be built with the same architecture/config). Overwrites the current LoRA + gains in place,
        /// so generation afterwards reflects the loaded fine-tune.
        /// </summary>
        public void LoadAdapter(string path)
        {
            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);
            if (br.ReadUInt32() != AdapterMagic) { throw new InvalidDataException("Not an Overfit Llama adapter file."); }
            var version = br.ReadInt32();
            if (version != AdapterVersion) { throw new InvalidDataException($"Unsupported adapter version {version}."); }
            var nLayers = br.ReadInt32();
            var dModel = br.ReadInt32();
            var vocab = br.ReadInt32();
            if (nLayers != _blocks.Length || dModel != _dModel || vocab != _vocab)
            {
                throw new InvalidDataException(
                    $"Adapter architecture mismatch (file {nLayers}L/{dModel}d/{vocab}v vs model {_blocks.Length}L/{_dModel}d/{_vocab}v).");
            }

            var ps = MaterializeParams();
            var count = br.ReadInt32();
            if (count != ps.Count)
            {
                throw new InvalidDataException($"Adapter parameter count mismatch ({count} vs {ps.Count}) — different LoRA targets/rank.");
            }
            foreach (var p in ps)
            {
                var len = br.ReadInt32();
                var dst = p.DataView.AsSpan();
                if (len != dst.Length)
                {
                    throw new InvalidDataException($"Adapter tensor length mismatch ({len} vs {dst.Length}).");
                }
                br.BaseStream.ReadExactly(MemoryMarshal.AsBytes(dst));
            }
        }

        private List<AutogradNode> MaterializeParams()
        {
            var ps = new List<AutogradNode>();
            foreach (var p in TrainableParameters()) { ps.Add(p); }
            return ps;
        }

        /// <summary>Next-token softmax cross-entropy: computes the mean loss over positions AND seeds
        /// <c>logits.Grad</c> with <c>softmax − onehot(target)</c> (normalized by T). Call
        /// <c>graph.BackwardFromGrad(logits)</c> afterwards. Mirrors the GPT-1 training loss.</summary>
        public static float CrossEntropyLossAndSeed(AutogradNode logits, int[] targets, int vocab)
        {
            var T = targets.Length;
            var data = logits.DataView.AsReadOnlySpan();
            var grad = logits.GradView.AsSpan();
            var total = 0.0;
            var scale = 1f / T;
            // Per-row softmax + grad seed, SIMD-batched via TensorPrimitives (vectorized Exp/Max/Sum —
            // far faster than a scalar MathF.Exp loop over a ~152k vocab). grad is fully overwritten
            // below (probs·invSum·scale per element), so no up-front Clear() is needed.
            Span<float> probs = vocab <= 4096 ? stackalloc float[vocab] : new float[vocab];

            for (var t = 0; t < T; t++)
            {
                var off = t * vocab;
                var row = data.Slice(off, vocab);
                var gradRow = grad.Slice(off, vocab);
                var probsRow = probs[..vocab];

                var max = TensorPrimitives.Max(row);
                TensorPrimitives.Subtract(row, max, probsRow);   // probs = row − max
                TensorPrimitives.Exp(probsRow, probsRow);        // probs = exp(row − max)
                var inv = 1f / TensorPrimitives.Sum(probsRow);

                var tgt = targets[t];
                total += -Math.Log(Math.Max(probsRow[tgt] * inv, 1e-30f));

                // grad = softmax·scale − onehot·scale = probs·(inv·scale), then subtract scale at target.
                TensorPrimitives.Multiply(probsRow, inv * scale, gradRow);
                gradRow[tgt] -= scale;
            }
            return (float)(total / T);
        }

        private int CheckpointArena(int T)
        {
            // One checkpointed block carves all its fwd+bwd tensors (data+grad) sequentially from this
            // sub-arena without intra-block reuse, so it must cover the SUM, not the live peak. Sized
            // generously (≈3× a tight estimate + slack) — it is transient (one block at a time, freed
            // after each), so over-provisioning costs nothing at steady state.
            var est = (long)T * (24L * _dModel + 8L * _dFF) + 3L * _nQHeads * T * T;
            return (int)Math.Min(3L * est + 4_000_000L, int.MaxValue - 16);
        }

        private AutogradNode Param(float[] init)
        {
            var store = new TensorStorage<float>(init.Length, clearMemory: false);
            init.CopyTo(store.AsSpan());
            var node = new AutogradNode(store, new TensorShape(init.Length), requiresGrad: true);
            _owned.Add(store); _owned.Add(node);
            return node;
        }

        private T Track<T>(T d) where T : IDisposable { _owned.Add(d); return d; }

        private static ConcatRowsDequantSource ConcatRows(DecodeWeight[] heads, int count)
        {
            var parts = new IDequantRowSource[count];
            for (var h = 0; h < count; h++) { parts[h] = heads[h].AsRowSource(); }
            return new ConcatRowsDequantSource(parts);
        }

        private static ConcatColsDequantSource ConcatCols(DecodeWeight[] heads, int count)
        {
            var parts = new IDequantRowSource[count];
            for (var h = 0; h < count; h++) { parts[h] = heads[h].AsRowSource(); }
            return new ConcatColsDequantSource(parts);
        }

        private void DisposeScratch()
        {
            for (var i = _scratch.Count - 1; i >= 0; i--) { _scratch[i].Dispose(); }
            _scratch.Clear();
        }

        public void Dispose()
        {
            DisposeScratch();
            for (var i = _owned.Count - 1; i >= 0; i--) { _owned[i].Dispose(); }
            _owned.Clear();
        }
    }
}
