// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Autograd
{
    /// <summary>
    /// A re-runnable forward segment for gradient checkpointing: maps an input node to an output node by
    /// recording its ops on the supplied (throwaway) graph. Must be deterministic and depend only on
    /// <paramref name="input"/> and captured parameters — it is run once for values in the forward pass
    /// and re-run for activations in the backward pass. See <see cref="ComputationGraph.Checkpoint"/>.
    /// </summary>
    public delegate AutogradNode CheckpointSegment(ComputationGraph graph, AutogradNode input);
}
