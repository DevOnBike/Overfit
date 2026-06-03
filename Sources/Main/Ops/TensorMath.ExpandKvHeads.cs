// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        /// <summary>
        /// Grouped-Query Attention (GQA) KV-head broadcast — the training equivalent of HF
        /// <c>repeat_kv</c>. The K (or V) tensor has fewer heads than Q: each KV head is shared by a
        /// group of <paramref name="groupSize"/> query heads. This op expands
        /// <c>[kvHeads, …]</c> → <c>[kvHeads·groupSize, …]</c> so the shared SDPA kernel (one head =
        /// one batch slice) can run unchanged.
        ///
        /// Forward: query head <c>qh</c> reads KV head <c>qh / groupSize</c> (HF mapping — KV head g
        /// feeds query heads <c>[g·groupSize, (g+1)·groupSize)</c>), a per-head block copy.
        ///
        /// Backward (the GQA-specific gradient): the upstream gradient of all <paramref name="groupSize"/>
        /// query heads in a group is <b>summed</b> back into the single shared KV head — because that
        /// KV head was read groupSize times in the forward.
        ///
        /// Layout-agnostic: dim 0 is the head axis (<paramref name="kvHeads"/>); every remaining axis
        /// forms the per-head block (<c>input.Size / kvHeads</c> floats). Works for <c>[kvHeads, T, d]</c>
        /// and <c>[kvHeads, T·d]</c> alike. <paramref name="groupSize"/> = 1 is a plain copy (MHA).
        /// </summary>
        public static AutogradNode ExpandKvHeads(
            ComputationGraph graph,
            AutogradNode input,
            int kvHeads,
            int groupSize)
        {
            if (kvHeads <= 0 || groupSize <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(groupSize), "kvHeads and groupSize must be positive.");
            }
            if (input.Shape.D0 != kvHeads)
            {
                throw new ArgumentException($"ExpandKvHeads: input.Shape.D0 ({input.Shape.D0}) must equal kvHeads ({kvHeads}).");
            }

            var qHeads = kvHeads * groupSize;
            var blockSize = input.Shape.Size / kvHeads;
            var output = AllocateNode(graph, input.Shape.WithD0(qHeads), input.RequiresGrad, clearMemory: false);

            var inS = input.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();

            for (var qh = 0; qh < qHeads; qh++)
            {
                var kvh = qh / groupSize;
                inS.Slice(kvh * blockSize, blockSize).CopyTo(outS.Slice(qh * blockSize, blockSize));
            }

            if (input.RequiresGrad)
            {
                graph?.Record(OpCode.ExpandKvHeads, output, input, i0: kvHeads, i1: groupSize);
            }

            return output;
        }

        /// <summary>GQA KV-head broadcast backward: sum the groupSize query-head gradient blocks
        /// back into each shared KV head's gradient (accumulating).</summary>
        public static void ExpandKvHeadsBackward(AutogradNode input, AutogradNode output, int kvHeads, int groupSize)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            var blockSize = input.Shape.Size / kvHeads;
            var dOut = output.GradView.AsReadOnlySpan();
            var dIn = input.GradView.AsSpan();

            for (var kvh = 0; kvh < kvHeads; kvh++)
            {
                var dInBlock = dIn.Slice(kvh * blockSize, blockSize);
                for (var g = 0; g < groupSize; g++)
                {
                    var qh = kvh * groupSize + g;
                    TensorPrimitives.Add(dInBlock, dOut.Slice(qh * blockSize, blockSize), dInBlock);
                }
            }
        }
    }
}
