// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;
using System.IO;
using DevOnBike.Overfit.Anomalies.Gpt;
using DevOnBike.Overfit.Anomalies.Monitoring.Contracts;
using DevOnBike.Overfit.DeepLearning;
using DevOnBike.Overfit.LanguageModels.LoRA;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace DevOnBike.Overfit.Anomalies.Adaptive
{
    /// <summary>
    /// Per-pod adaptive anomaly monitor — the operator-facing lifecycle that turns the
    /// anomaly+LoRA capability into a deployment workflow. Over a single shared base model
    /// it keeps a detector per pod, watches for sustained false-positive pressure (the
    /// cross-pod base scoring a pod's benign regime as elevated), and lets the operator
    /// adapt that pod with a cheap LM-head LoRA fine-tuned on the pod's recent benign
    /// window. Per-pod adapters are stored as <c>.bin</c> on disk and swapped in/out of the
    /// shared model (merge is in-place + idempotent) so each pod is scored against its own
    /// adapter.
    ///
    /// Adaptation is OPERATOR-GATED: the monitor RECOMMENDS (sustained elevated-but-not-
    /// critical scores flag a pod via <see cref="PodsNeedingAdaptation"/>) but never adapts
    /// on its own — online, an elevated score can't be distinguished from a moderate real
    /// incident without a label, so the caller decides and supplies the benign window
    /// (here, the buffered sub-critical snapshots). The deployment recommendation is
    /// LM-head LoRA on a trained base (best + cheapest — see the production target A/B).
    ///
    /// Not thread-safe; drive from one loop. Disposing un-merges any active adapter so the
    /// base model is left clean.
    /// </summary>
    public sealed class AdaptiveAnomalyMonitor : IDisposable
    {
        private readonly GPT1Model _model;
        private readonly AdaptivePolicy _policy;
        private readonly MetricTokenizer _tokenizer = new();
        private readonly Dictionary<string, PodState> _pods = new(StringComparer.Ordinal);
        private string? _activeAdapterPod;
        private bool _disposed;

        public AdaptiveAnomalyMonitor(GPT1Model model, AdaptivePolicy policy)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _policy = policy ?? throw new ArgumentNullException(nameof(policy));
            ArgumentException.ThrowIfNullOrEmpty(policy.AdapterDirectory);
            Directory.CreateDirectory(policy.AdapterDirectory);
        }

        /// <summary>Pods seen so far.</summary>
        public IReadOnlyCollection<string> Pods => _pods.Keys;

        /// <summary>
        /// Scores one snapshot for its pod, with that pod's adapter active (if any). Returns
        /// the anomaly score; updates the pod's benign buffer and false-positive tracking.
        /// </summary>
        public AnomalyScore Observe(MetricSnapshot snapshot)
        {
            ThrowIfDisposed();
            var state = GetOrCreate(snapshot.PodName);
            ActivateAdapterFor(snapshot.PodName);

            var score = state.Detector.Score(snapshot);
            if (!score.IsWarmup)
            {
                Track(state, snapshot, score);
            }
            return score;
        }

        /// <summary>True once the pod has shown sustained false-positive pressure (and isn't yet adapted).</summary>
        public bool NeedsAdaptation(string podName) =>
            _pods.TryGetValue(podName, out var s) && s.AdaptationRecommended;

        /// <summary>True if the pod currently has a per-pod adapter loaded.</summary>
        public bool IsAdapted(string podName) =>
            _pods.TryGetValue(podName, out var s) && s.Adapter is not null;

        /// <summary>Pods the monitor recommends adapting.</summary>
        public IReadOnlyList<string> PodsNeedingAdaptation()
        {
            var list = new List<string>();
            foreach (var (pod, s) in _pods)
            {
                if (s.AdaptationRecommended) { list.Add(pod); }
            }
            return list;
        }

        /// <summary>
        /// Adapts a pod: fine-tunes an LM-head LoRA on its buffered benign window, stores the
        /// adapter <c>.bin</c>, and loads it for the pod (active on the next <see cref="Observe"/>).
        /// Throws if the pod is unknown or its benign buffer is below
        /// <see cref="AdaptivePolicy.MinBenignWindow"/>. Returns the saved adapter path.
        /// </summary>
        public string Adapt(string podName)
        {
            ThrowIfDisposed();
            if (!_pods.TryGetValue(podName, out var state))
            {
                throw new OverfitRuntimeException($"Unknown pod '{podName}' — Observe it before adapting.");
            }
            if (state.Benign.Count < _policy.MinBenignWindow)
            {
                throw new OverfitRuntimeException(
                    $"Pod '{podName}' has {state.Benign.Count} buffered benign snapshots; need ≥ {_policy.MinBenignWindow}.");
            }

            // Fine-tune against the CLEAN base — detach any active adapter first.
            DeactivateAdapter();

            var benign = state.Benign.ToArray();
            var corpus = _tokenizer.EncodeSequence(benign);
            var tps = MetricTokenizer.TokensPerSnapshot;
            var path = AdapterPath(podName);

            using (var tuner = new Gpt1LoRAFineTuner(_model, _policy.LoRARank, LoRATargetModules.LanguageModelHead))
            {
                // Fine-tune over the exact position range the detector exercises so the merge
                // lands where scores are read (see the contextLength gotcha in the LoRA tests).
                tuner.FineTune(corpus, _policy.LoRASteps, _policy.ContextSnapshots * tps, _policy.LoRALearningRate);
                tuner.Save(path);
            }
            // tuner disposed → provider detached; base model is plain again.

            state.Adapter?.Dispose();
            state.Adapter = Gpt1LoRAMergeAdapter.Load(_model, path);
            state.AdaptationRecommended = false;
            state.FalsePositiveStreak = 0;
            return path;
        }

        public void Dispose()
        {
            if (_disposed) { return; }
            _disposed = true;

            DeactivateAdapter();   // leave the base model clean
            foreach (var s in _pods.Values)
            {
                s.Adapter?.Dispose();
                s.Detector.Dispose();
            }
            _pods.Clear();
        }

        // ── Private ──────────────────────────────────────────────────────────

        private void Track(PodState state, MetricSnapshot snapshot, AnomalyScore score)
        {
            // Buffer sub-critical snapshots as the pod's benign window for adaptation.
            if (score.Score < _policy.CriticalThreshold)
            {
                state.Benign.Enqueue(snapshot);
                while (state.Benign.Count > _policy.BenignWindow) { state.Benign.Dequeue(); }
            }

            // Sustained elevated-but-not-critical = false-positive pressure (base miscalibration).
            if (score.Score >= _policy.AlertThreshold && score.Score < _policy.CriticalThreshold)
            {
                state.FalsePositiveStreak++;
            }
            else
            {
                state.FalsePositiveStreak = 0;
            }

            if (state.Adapter is null
                && state.FalsePositiveStreak >= _policy.AdaptAfterStreak
                && state.Benign.Count >= _policy.MinBenignWindow)
            {
                state.AdaptationRecommended = true;
            }
        }

        private PodState GetOrCreate(string podName)
        {
            if (_pods.TryGetValue(podName, out var existing)) { return existing; }

            var handle = SlmRuntimeFactory.CreateGpt1(_model);
            var state = new PodState
            {
                Detector = new GptAnomalyDetector(handle, _policy.ContextSnapshots),
                Benign = new Queue<MetricSnapshot>(_policy.BenignWindow + 1),
            };

            // Reload a previously-trained per-pod adapter from disk (pod restart / new monitor
            // instance over the same AdapterDirectory) so adaptation survives process restarts.
            var path = AdapterPath(podName);
            if (File.Exists(path))
            {
                state.Adapter = Gpt1LoRAMergeAdapter.Load(_model, path);
            }

            _pods[podName] = state;
            return state;
        }

        // Ensures the shared model carries the given pod's adapter (and only it).
        private void ActivateAdapterFor(string podName)
        {
            if (_activeAdapterPod == podName) { return; }

            DeactivateAdapter();
            if (_pods.TryGetValue(podName, out var s) && s.Adapter is { } adapter)
            {
                adapter.Enable();
                _activeAdapterPod = podName;
            }
        }

        private void DeactivateAdapter()
        {
            if (_activeAdapterPod is { } pod && _pods.TryGetValue(pod, out var s))
            {
                s.Adapter?.Disable();
            }
            _activeAdapterPod = null;
        }

        private string AdapterPath(string podName) =>
            Path.Combine(_policy.AdapterDirectory, $"{Sanitize(podName)}.lora.bin");

        private static string Sanitize(string podName)
        {
            var chars = podName.ToCharArray();
            for (var i = 0; i < chars.Length; i++)
            {
                if (Array.IndexOf(Path.GetInvalidFileNameChars(), chars[i]) >= 0) { chars[i] = '_'; }
            }
            return new string(chars);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) { throw new ObjectDisposedException(nameof(AdaptiveAnomalyMonitor)); }
        }

        // Per-pod monitoring state (nested/private — exempt from one-type-per-file).
        private sealed class PodState
        {
            public required GptAnomalyDetector Detector { get; init; }
            public required Queue<MetricSnapshot> Benign { get; init; }
            public Gpt1LoRAMergeAdapter? Adapter { get; set; }
            public int FalsePositiveStreak { get; set; }
            public bool AdaptationRecommended { get; set; }
        }
    }
}
