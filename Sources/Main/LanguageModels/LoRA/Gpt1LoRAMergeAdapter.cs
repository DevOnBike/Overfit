// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics.CodeAnalysis;
using System.Numerics.Tensors;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    /// <summary>
    /// Inference-side counterpart of <see cref="Gpt1LoRAFineTuner"/> — applies a
    /// trained LoRA adapter to a <see cref="GPT1Model"/> by in-place weight-merge:
    ///
    ///   Enable():  W += scale * (A @ B)   for every entry in the .bin
    ///   Disable(): W -= scale * (A @ B)
    ///
    /// Entries are merged into the matching <see cref="TensorStorage{T}"/>:
    ///   <see cref="LoRATargetModules.LanguageModelHead"/> -> GPT1Model.LMHead.Data
    ///   <see cref="LoRATargetModules.FeedForwardUp"/>      -> Blocks[layer].FFN.W1.Data
    ///   <see cref="LoRATargetModules.FeedForwardDown"/>    -> Blocks[layer].FFN.W2.Data
    ///
    /// The merge is in place on the model's own weight storage. A KV-cached runtime
    /// created <i>after</i> <see cref="Enable"/> observes the merged weights — this
    /// is how a trained adapter (Stage 1 LM head and/or Stage 2 FFN) reaches
    /// GptAnomalyDetector. Multi-tenant use: keep one base model, Enable/Disable
    /// per-tenant adapters around each request.
    /// </summary>
    [SuppressMessage(
        "IDisposableAnalyzers.Correctness",
        "IDISP008:Don't assign member with injected and created disposables",
        Justification = "Borrowed zero-copy handles - the adapter never owns GPT1Model weight storage.")]
    public sealed class Gpt1LoRAMergeAdapter : IDisposable
    {
        private readonly MergeTarget[] _targets;
        private readonly float[] _deltaScratch;
        private readonly float _scale;

        private bool _enabled;
        private bool _disposed;

        private Gpt1LoRAMergeAdapter(MergeTarget[] targets, float scale)
        {
            _targets = targets;
            _scale = scale;

            var maxDelta = 0;
            foreach (var target in targets)
            {
                maxDelta = Math.Max(maxDelta, target.Weight.InDim * target.Weight.OutDim);
            }

            _deltaScratch = new float[maxDelta];
        }

        /// <summary>
        /// Loads a LoRA adapter saved by <see cref="Gpt1LoRAFineTuner.Save"/> and
        /// binds every entry to the matching weight matrix of <paramref name="model"/>.
        /// </summary>
        /// <param name="model">The model whose weight matrices the adapter merges into.</param>
        /// <param name="path">Path to a .bin written by <see cref="Gpt1LoRAFineTuner.Save"/>.</param>
        /// <param name="scale">
        /// LoRA scale (alpha/rank). Gpt1LoRAFineTuner trains with scale = 1, so the
        /// default merges the adapter exactly as it was trained.
        /// </param>
        public static Gpt1LoRAMergeAdapter Load(GPT1Model model, string path, float scale = 1f)
        {
            ArgumentNullException.ThrowIfNull(model);

            var entries = Gpt1LoRAFile.Load(path);
            var dModel = model.Config.DModel;
            var dFF = model.Config.DFF;
            var vocab = model.Config.VocabSize;
            var nLayers = model.Config.NLayers;

            var targets = new MergeTarget[entries.Count];

            for (var i = 0; i < entries.Count; i++)
            {
                var entry = entries[i];
                var weight = entry.Weight;

                TensorStorage<float> storage;
                int expectedIn;
                int expectedOut;

                switch (entry.Module)
                {
                    case LoRATargetModules.LanguageModelHead:
                        if (model.Config.TieWeights)
                        {
                            throw new NotSupportedException(
                                "LoRA on the LM head requires an untied head; the model must be built with TieWeights=false.");
                        }

                        storage = model.LMHead.Data;
                        expectedIn = dModel;
                        expectedOut = vocab;
                        break;

                    case LoRATargetModules.FeedForwardUp:
                        ValidateLayer(entry.Layer, nLayers, entry.Module);
                        storage = model.Blocks[entry.Layer].FFN.W1.Data;
                        expectedIn = dModel;
                        expectedOut = dFF;
                        break;

                    case LoRATargetModules.FeedForwardDown:
                        ValidateLayer(entry.Layer, nLayers, entry.Module);
                        storage = model.Blocks[entry.Layer].FFN.W2.Data;
                        expectedIn = dFF;
                        expectedOut = dModel;
                        break;

                    default:
                        throw new NotSupportedException(
                            $"Gpt1LoRAMergeAdapter cannot merge module {entry.Module}.");
                }

                if (weight.InDim != expectedIn || weight.OutDim != expectedOut)
                {
                    throw new InvalidDataException(
                        $"LoRA entry [{entry.Module} layer {entry.Layer}] dimensions " +
                        $"[{weight.InDim}x{weight.OutDim}] do not match the model [{expectedIn}x{expectedOut}].");
                }

                targets[i] = new MergeTarget(storage, weight);
            }

            return new Gpt1LoRAMergeAdapter(targets, scale);
        }

        /// <summary>True while the LoRA delta is merged into the base weights.</summary>
        public bool IsEnabled => _enabled;

        /// <summary>Number of weight matrices this adapter merges into.</summary>
        public int TargetCount => _targets.Length;

        public long TrainableParameterCount
        {
            get
            {
                var total = 0L;
                foreach (var target in _targets)
                {
                    total += target.Weight.ParameterCount;
                }

                return total;
            }
        }

        /// <summary>Merges every LoRA delta into the base weights. Idempotent.</summary>
        public void Enable()
        {
            ThrowIfDisposed();

            if (_enabled)
            {
                return;
            }

            ApplyDelta(_scale);
            _enabled = true;
        }

        /// <summary>Removes every LoRA delta, restoring the base weights. Idempotent.</summary>
        public void Disable()
        {
            ThrowIfDisposed();

            if (!_enabled)
            {
                return;
            }

            ApplyDelta(-_scale);
            _enabled = false;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            // Leave the base model clean — un-merge if still enabled.
            if (_enabled)
            {
                ApplyDelta(-_scale);
                _enabled = false;
            }

            _disposed = true;
        }

        private void ApplyDelta(float signedScale)
        {
            foreach (var target in _targets)
            {
                var size = target.Weight.InDim * target.Weight.OutDim;
                var delta = _deltaScratch.AsSpan(0, size);

                target.Weight.ComputeDelta(delta);   // delta = A @ B, row-major

                var storage = target.Storage.AsSpan().Slice(0, size);
                TensorPrimitives.MultiplyAdd(delta, signedScale, storage, storage);
            }
        }

        private static void ValidateLayer(int layer, int nLayers, LoRATargetModules module)
        {
            if (layer < 0 || layer >= nLayers)
            {
                throw new InvalidDataException(
                    $"LoRA entry for {module} references layer {layer}; the model has {nLayers} layers.");
            }
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }

        /// <summary>A borrowed base weight storage paired with its trained LoRA factors.</summary>
        private readonly record struct MergeTarget(TensorStorage<float> Storage, LoRAWeight Weight);
    }
}
