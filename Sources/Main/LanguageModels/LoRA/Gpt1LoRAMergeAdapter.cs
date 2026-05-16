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
    /// trained LM-head LoRA adapter to a <see cref="GPT1Model"/> by weight-merge:
    ///
    ///   Enable():  LMHead += scale * (A @ B)
    ///   Disable(): LMHead -= scale * (A @ B)
    ///
    /// The merge is done in place on <c>GPT1Model.LMHead.Data</c>. Because the
    /// cached runtime (<c>StackWeights</c> for an untied model) holds a zero-copy
    /// reference to that exact <see cref="TensorStorage{T}"/>, the merged weights
    /// are visible to KV-cached decode immediately — no inference-kernel changes,
    /// no re-binding. This is how a trained adapter reaches GptAnomalyDetector.
    ///
    /// Stage 1 scope: LM head only (the single module produced by
    /// <see cref="Gpt1LoRAFineTuner"/>). Multi-tenant use: keep one base model,
    /// Enable/Disable per-tenant adapters around each request.
    /// </summary>
    [SuppressMessage(
        "IDisposableAnalyzers.Correctness",
        "IDISP008:Don't assign member with injected and created disposables",
        Justification = "Borrowed zero-copy handle - the adapter never owns GPT1Model.LMHead storage.")]
    public sealed class Gpt1LoRAMergeAdapter : IDisposable
    {
        private readonly TensorStorage<float> _lmHead;   // borrowed — GPT1Model.LMHead.Data
        private readonly LoRAWeight _weight;
        private readonly float[] _deltaScratch;
        private readonly float _scale;

        private bool _enabled;
        private bool _disposed;

        private Gpt1LoRAMergeAdapter(TensorStorage<float> lmHead, LoRAWeight weight, float scale)
        {
            _lmHead = lmHead;
            _weight = weight;
            _scale = scale;
            _deltaScratch = new float[weight.InDim * weight.OutDim];
        }

        /// <summary>
        /// Loads a LoRA adapter saved by <see cref="Gpt1LoRAFineTuner.Save"/> and
        /// binds it to <paramref name="model"/>'s LM head.
        /// </summary>
        /// <param name="scale">
        /// LoRA scale (alpha/rank). Gpt1LoRAFineTuner trains with scale = 1, so the
        /// default merges the adapter exactly as it was trained.
        /// </param>
        public static Gpt1LoRAMergeAdapter Load(GPT1Model model, string path, float scale = 1f)
        {
            ArgumentNullException.ThrowIfNull(model);

            if (model.Config.TieWeights)
            {
                throw new NotSupportedException(
                    "Gpt1LoRAMergeAdapter targets the untied LM head; the model must be built with TieWeights=false.");
            }

            var weight = Gpt1LoRAFile.LoadLMHead(path);
            var dModel = model.Config.DModel;
            var vocab = model.Config.VocabSize;

            if (weight.InDim != dModel || weight.OutDim != vocab)
            {
                throw new InvalidDataException(
                    $"LoRA dimensions [{weight.InDim}x{weight.OutDim}] do not match " +
                    $"model LM head [{dModel}x{vocab}].");
            }

            return new Gpt1LoRAMergeAdapter(model.LMHead.Data, weight, scale);
        }

        /// <summary>True while the LoRA delta is merged into the base LM head.</summary>
        public bool IsEnabled => _enabled;

        public long TrainableParameterCount => _weight.ParameterCount;

        /// <summary>Merges the LoRA delta into the base LM head. Idempotent.</summary>
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

        /// <summary>Removes the LoRA delta, restoring the base LM head. Idempotent.</summary>
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
            var size = _weight.InDim * _weight.OutDim;
            var delta = _deltaScratch.AsSpan(0, size);

            _weight.ComputeDelta(delta);   // delta = A @ B  [dModel x vocab], row-major

            var target = _lmHead.AsSpan();
            TensorPrimitives.MultiplyAdd(delta, signedScale, target, target);
        }

        private void ThrowIfDisposed()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
        }
    }
}
