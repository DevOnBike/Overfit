// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.Ops
{
    public static partial class TensorMath
    {
        /// <summary>
        /// Token embedding lookup.
        ///
        /// Looks up rows from a weight matrix (the embedding table):
        ///   output[t] = embeddings[tokenIds[t]]
        ///
        /// Input:
        ///   tokenIds  — int array of shape [seqLen], values in [0, vocabSize)
        ///   embeddings — AutogradNode of shape [vocabSize, embeddingDim] (the learnable table)
        ///
        /// Output:
        ///   AutogradNode of shape [seqLen, embeddingDim]
        ///
        /// Backward: gradient accumulates (scatter-add) into embedding rows
        ///   that were accessed. Rows not accessed in this batch get zero gradient.
        ///
        /// Note: tokenIds are stored on tape as an int[] context array.
        /// </summary>
        public static AutogradNode Embedding(
            ComputationGraph graph,
            int[] tokenIds,
            AutogradNode embeddings)
        {
            if (embeddings.Shape.Rank < 2)
            {
                throw new ArgumentException(
                    $"Embedding table must be rank 2 [vocabSize, embDim], got rank {embeddings.Shape.Rank}.",
                    nameof(embeddings));
            }

            var seqLen = tokenIds.Length;
            var embDim = embeddings.Shape.D1;
            var vocabSize = embeddings.Shape.D0;

            var output = AllocateNode(
                graph,
                new TensorShape(seqLen, embDim),
                embeddings.RequiresGrad,
                clearMemory: false);

            var embS = embeddings.DataView.AsReadOnlySpan();
            var outS = output.DataView.AsSpan();

            for (var t = 0; t < seqLen; t++)
            {
                var id = tokenIds[t];

                if ((uint)id >= (uint)vocabSize)
                {
                    throw new ArgumentOutOfRangeException(
                        nameof(tokenIds),
                        $"Token id {id} at position {t} is out of range [0, {vocabSize}).");
                }

                embS.Slice(id * embDim, embDim).CopyTo(outS.Slice(t * embDim, embDim));
            }

            if (output.RequiresGrad)
            {
                // Store tokenIds as context for backward.
                // Tape uses contextData (int[]) to carry token ids.
                graph?.RecordEmbedding(OpCode.Embedding, output, embeddings, tokenIds);
            }

            return output;
        }

        /// <summary>
        /// Embedding backward: scatter-add output gradient into embedding table rows.
        ///   dEmbeddings[tokenId] += dOutput[t]  for each t
        /// </summary>
        public static void EmbeddingBackward(
            AutogradNode embeddings,
            AutogradNode output,
            int[] tokenIds)
        {
            if (!embeddings.RequiresGrad)
            {
                return;
            }

            var seqLen = tokenIds.Length;
            var embDim = embeddings.Shape.D1;

            var dOut = output.GradView.AsReadOnlySpan();
            var dEmb = embeddings.GradView.AsSpan();

            for (var t = 0; t < seqLen; t++)
            {
                var id = tokenIds[t];
                var dOutRow = dOut.Slice(t * embDim, embDim);
                var dEmbRow = dEmb.Slice(id * embDim, embDim);

                TensorPrimitives.Add(dOutRow, dEmbRow, dEmbRow);
            }
        }
    }
}
