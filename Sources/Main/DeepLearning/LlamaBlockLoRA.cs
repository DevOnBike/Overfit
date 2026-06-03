// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// The trainable LoRA adapters for one <see cref="TrainableLlamaBlock"/> — one per linear projection
    /// (Q/K/V/O attention + gate/up/down SwiGLU). Each is added on top of its frozen quantized base.
    /// Any field may be null (that projection trains no LoRA). This is the trainable capacity in QLoRA
    /// fine-tuning; the base stays frozen 4-bit.
    /// </summary>
    public sealed class LlamaBlockLoRA : IDisposable
    {
        public LoRAAdapter? Q { get; init; }
        public LoRAAdapter? K { get; init; }
        public LoRAAdapter? V { get; init; }
        public LoRAAdapter? O { get; init; }
        public LoRAAdapter? Gate { get; init; }
        public LoRAAdapter? Up { get; init; }
        public LoRAAdapter? Down { get; init; }

        /// <summary>Creates LoRA on ALL seven projections at the given rank.</summary>
        public static LlamaBlockLoRA CreateAll(
            int dModel, int qDim, int kvDim, int dFF, int rank, Random rng)
        {
            return new LlamaBlockLoRA
            {
                Q = new LoRAAdapter(dModel, qDim, rank, rng),
                K = new LoRAAdapter(dModel, kvDim, rank, rng),
                V = new LoRAAdapter(dModel, kvDim, rank, rng),
                O = new LoRAAdapter(qDim, dModel, rank, rng),
                Gate = new LoRAAdapter(dModel, dFF, rank, rng),
                Up = new LoRAAdapter(dModel, dFF, rank, rng),
                Down = new LoRAAdapter(dFF, dModel, rank, rng),
            };
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            foreach (var a in new[] { Q, K, V, O, Gate, Up, Down })
            {
                if (a is not null) { yield return a.A; yield return a.B; }
            }
        }

        public void Dispose()
        {
            Q?.Dispose(); K?.Dispose(); V?.Dispose(); O?.Dispose();
            Gate?.Dispose(); Up?.Dispose(); Down?.Dispose();
        }
    }
}
