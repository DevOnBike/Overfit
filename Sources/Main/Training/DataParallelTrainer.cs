// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Threading.Tasks;
using DevOnBike.Overfit.Optimizers.Abstractions;
using DevOnBike.Overfit.Parameters;

namespace DevOnBike.Overfit.Training
{
    /// <summary>
    /// Synchronous single-machine data-parallel trainer: one master parameter set (owned by the
    /// optimizer) and <c>N</c> worker replicas that train disjoint micro-batches in parallel, their
    /// gradients <b>averaged</b> into the master each step (the standard "N cores → N replicas"
    /// throughput pattern). Turns a P-core box into a P× larger effective batch without touching the
    /// model or optimizer code.
    ///
    /// <para>The trainer is model-agnostic: it only manipulates <see cref="Parameter"/> data/grad
    /// buffers, so it works for any model exposing <c>TrainableParameters()</c>. The caller owns the
    /// models, their per-replica <c>ComputationGraph</c>s, and the task-specific forward+loss+backward,
    /// supplied as the <c>trainWorker</c> delegate to <see cref="Step"/>. The trainer owns only the
    /// gradient all-reduce, optional global grad-norm clip, the optimizer step, and the broadcast of
    /// the updated weights back to the replicas.</para>
    ///
    /// Per <see cref="Step"/>:
    /// <list type="number">
    ///   <item>master gradients are zeroed,</item>
    ///   <item><c>trainWorker(workerIndex)</c> runs for each replica in parallel — each computes its
    ///         local gradients into its own parameter set and returns its scalar loss,</item>
    ///   <item>worker gradients are averaged (×1/N) into the master,</item>
    ///   <item>an optional global grad-norm clip is applied to the master gradients,</item>
    ///   <item><c>optimizer.Step()</c> updates the master weights,</item>
    ///   <item>the updated master weights are broadcast back to every replica.</item>
    /// </list>
    ///
    /// <para>Averaging (not summing) the gradients keeps the effective learning rate comparable to a
    /// single replica's; scale the learning rate up (e.g. √N, the linear-scaling rule) if you want the
    /// larger global batch to translate into faster convergence.</para>
    /// </summary>
    public sealed class DataParallelTrainer
    {
        private readonly Parameter[] _master;
        private readonly Parameter[][] _workers;
        private readonly float _invWorkerCount;

        /// <param name="masterParameters">
        /// The authoritative parameters the optimizer updates (typically <c>master.TrainableParameters()</c>).
        /// </param>
        /// <param name="workerParameterSets">
        /// One parameter list per replica, each aligned 1:1 with <paramref name="masterParameters"/>
        /// (same count, same per-parameter length). Construct replicas with the same config as the master.
        /// </param>
        public DataParallelTrainer(
            IReadOnlyList<Parameter> masterParameters,
            IReadOnlyList<IReadOnlyList<Parameter>> workerParameterSets)
        {
            ArgumentNullException.ThrowIfNull(masterParameters);
            ArgumentNullException.ThrowIfNull(workerParameterSets);

            if (masterParameters.Count == 0)
            {
                throw new ArgumentException("Master has no parameters.", nameof(masterParameters));
            }
            if (workerParameterSets.Count == 0)
            {
                throw new ArgumentException("At least one worker replica is required.", nameof(workerParameterSets));
            }

            _master = new Parameter[masterParameters.Count];
            for (var i = 0; i < masterParameters.Count; i++)
            {
                _master[i] = masterParameters[i];
            }

            _workers = new Parameter[workerParameterSets.Count][];
            for (var w = 0; w < workerParameterSets.Count; w++)
            {
                var set = workerParameterSets[w];
                if (set.Count != _master.Length)
                {
                    throw new ArgumentException(
                        $"Worker {w} has {set.Count} parameters; master has {_master.Length}.",
                        nameof(workerParameterSets));
                }

                var replica = new Parameter[set.Count];
                for (var i = 0; i < set.Count; i++)
                {
                    if (set[i].DataSpan.Length != _master[i].DataSpan.Length)
                    {
                        throw new ArgumentException(
                            $"Worker {w} parameter {i} length {set[i].DataSpan.Length} " +
                            $"does not match master length {_master[i].DataSpan.Length}.",
                            nameof(workerParameterSets));
                    }
                    replica[i] = set[i];
                }
                _workers[w] = replica;
            }

            _invWorkerCount = 1f / _workers.Length;
        }

        /// <summary>Number of worker replicas.</summary>
        public int WorkerCount => _workers.Length;

        /// <summary>Number of trainable parameters per replica (== master parameter count).</summary>
        public int ParameterCount => _master.Length;

        /// <summary>
        /// Copies the master weights into every replica. Call once after the master is initialised /
        /// loaded and before the first <see cref="Step"/>; <see cref="Step"/> re-broadcasts after each
        /// update, so the replicas stay in lock-step with the master.
        /// </summary>
        public void BroadcastParameters()
        {
            var master = _master;
            var workers = _workers;
            Parallel.For(0, master.Length, p =>
            {
                var src = master[p].DataReadOnlySpan;
                for (var w = 0; w < workers.Length; w++)
                {
                    src.CopyTo(workers[w][p].DataSpan);
                }
            });
        }

        /// <summary>
        /// Runs one synchronous data-parallel optimization step (see the class remarks for the phase
        /// order) and returns the mean worker loss.
        /// </summary>
        /// <param name="optimizer">Optimizer over the master parameters (e.g. Adam / SGD).</param>
        /// <param name="trainWorker">
        /// Task-specific per-replica body: given a worker index, it must sample that replica's
        /// micro-batch, reset its graph, clear its own gradients, run forward + loss + backward into the
        /// replica's parameters, and return the scalar loss. Invoked once per replica, in parallel.
        /// </param>
        /// <param name="maxGradNorm">
        /// When &gt; 0, the master gradients are scaled so their global L2 norm does not exceed this
        /// value (gradient clipping). 0 disables clipping.
        /// </param>
        public float Step(IOptimizer optimizer, Func<int, float> trainWorker, float maxGradNorm = 0f)
        {
            ArgumentNullException.ThrowIfNull(optimizer);
            ArgumentNullException.ThrowIfNull(trainWorker);

            ClearMasterGradients();

            var workers = _workers;
            var losses = new float[workers.Length];
            if (workers.Length > 1)
            {
                Parallel.For(0, workers.Length, w => losses[w] = trainWorker(w));
            }
            else
            {
                losses[0] = trainWorker(0);
            }

            AverageWorkerGradientsIntoMaster();

            if (maxGradNorm > 0f)
            {
                ClipMasterGradientNorm(maxGradNorm);
            }

            optimizer.Step();
            BroadcastParameters();

            var sum = 0f;
            for (var w = 0; w < losses.Length; w++)
            {
                sum += losses[w];
            }
            return sum / losses.Length;
        }

        private void ClearMasterGradients()
        {
            for (var p = 0; p < _master.Length; p++)
            {
                _master[p].GradSpan.Clear();
            }
        }

        private void AverageWorkerGradientsIntoMaster()
        {
            var master = _master;
            var workers = _workers;
            var scale = _invWorkerCount;
            Parallel.For(0, master.Length, p =>
            {
                var masterGrad = master[p].GradSpan;
                for (var w = 0; w < workers.Length; w++)
                {
                    var workerGrad = workers[w][p].GradSpan;
                    for (var i = 0; i < masterGrad.Length; i++)
                    {
                        masterGrad[i] += workerGrad[i] * scale;
                    }
                }
            });
        }

        private void ClipMasterGradientNorm(float maxNorm)
        {
            var master = _master;

            var totalNormSq = 0.0;
            for (var p = 0; p < master.Length; p++)
            {
                var grad = master[p].GradSpan;
                for (var i = 0; i < grad.Length; i++)
                {
                    totalNormSq += (double)grad[i] * grad[i];
                }
            }

            var norm = (float)Math.Sqrt(totalNormSq);
            if (norm <= maxNorm)
            {
                return;
            }

            var scale = maxNorm / (norm + 1e-6f);
            for (var p = 0; p < master.Length; p++)
            {
                var grad = master[p].GradSpan;
                for (var i = 0; i < grad.Length; i++)
                {
                    grad[i] *= scale;
                }
            }
        }
    }
}
