// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Numerics.Tensors;
using DevOnBike.Overfit.Tensors;

namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    /// <summary>
    /// A single LoRA adapter for one weight matrix W [inDim × outDim].
    ///
    /// LoRA decomposition:
    ///   W_eff = W + scale * (A @ B)
    ///
    ///   A : float[inDim  × rank]  — initialized N(0, 1/sqrt(rank))
    ///   B : float[rank   × outDim] — initialized to zeros (standard LoRA init)
    ///
    /// Forward (used without merging, e.g. in training):
    ///   r     = A^T @ x        [rank]
    ///   delta = B^T @ r        [outDim]
    ///   return scale * delta
    ///
    /// Memory: (inDim + outDim) * rank floats — ~2–8% of base weight.
    /// </summary>
    public sealed class LoRAWeight
    {
        private readonly float[] _a;      // [inDim × rank]   row-major
        private readonly float[] _b;      // [rank  × outDim] row-major
        private readonly float[] _gradA;  // same shape as A
        private readonly float[] _gradB;  // same shape as B

        public LoRAWeight(int inDim, int outDim, int rank, Random? rng = null)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inDim);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outDim);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rank);

            InDim = inDim;
            OutDim = outDim;
            Rank = rank;

            _a = new float[(long)inDim * rank];
            _b = new float[(long)rank * outDim];
            _gradA = new float[(long)inDim * rank];
            _gradB = new float[(long)rank * outDim];

            InitializeA(rng ?? new Random(42));
            // B stays zero — standard LoRA initialization
        }

        public int InDim
        {
            get;
        }
        public int OutDim
        {
            get;
        }
        public int Rank
        {
            get;
        }
        public long ParameterCount => (long)(InDim + OutDim) * Rank;

        public ReadOnlySpan<float> A => _a.AsSpan();
        public ReadOnlySpan<float> B => _b.AsSpan();
        public Span<float> GradA => _gradA.AsSpan();
        public Span<float> GradB => _gradB.AsSpan();

        // Internal mutable access for load/copy
        internal Span<float> AMutable => _a.AsSpan();
        internal Span<float> BMutable => _b.AsSpan();

        /// <summary>
        /// Computes full delta matrix: delta[inDim × outDim] = A @ B  (row-major).
        /// delta[i,j] = sum_k( A[i,k] * B[k,j] )
        /// Caller adds: W += scale * delta.
        /// </summary>
        public void ComputeDelta(Span<float> delta)
        {
            if (delta.Length != InDim * OutDim)
            {
                throw new ArgumentException(
                $"delta must be {InDim * OutDim} floats, got {delta.Length}");
            }

            delta.Clear();
            for (var i = 0; i < InDim; i++)
            {
                var aRow = _a.AsSpan(i * Rank, Rank);
                var dRow = delta.Slice(i * OutDim, OutDim);
                for (var k = 0; k < Rank; k++)
                {
                    var a_ik = aRow[k];
                    if (a_ik == 0f)
                    {
                        continue;
                    }
                    TensorPrimitives.MultiplyAdd(
                        _b.AsSpan(k * OutDim, OutDim),
                        a_ik, dRow, dRow);
                }
            }
        }

        /// <summary>
        /// Adds LoRA delta to an output vector for a single input x.
        ///   result += scale * B^T @ (A^T @ x)
        /// Used in non-merged training forward pass.
        /// </summary>
        public void ForwardAdd(ReadOnlySpan<float> x, Span<float> result, float scale)
        {
            // r[] is used as accumulator below (MultiplyAdd reads r[k] then writes).
            // With `[module: SkipLocalsInit]` stackalloc no longer pre-zeroes —
            // explicit Clear() is required so the accumulation starts from 0.
            using var rBuffer = Rank <= 256 ? default : new PooledBuffer<float>(Rank, clearMemory: false);
#pragma warning disable OVERFIT026 // BOUND: guarded at 256 floats = 1 KB. LoRA rank is caller-supplied and validated only as positive upstream, so the bound lives here; ranks above 256 (unheard of in practice) take the pooled branch instead of the stack.
            Span<float> r = Rank <= 256 ? stackalloc float[Rank] : rBuffer.Span;
#pragma warning restore OVERFIT026
            r.Clear();

            // r[k] = sum_i( A[i,k] * x[i] )
            for (var i = 0; i < InDim; i++)
            {
                if (x[i] == 0f)
                {
                    continue;
                }
                TensorPrimitives.MultiplyAdd(
                    _a.AsSpan(i * Rank, Rank), x[i], r, r);
            }

            // result[j] += scale * sum_k( B[k,j] * r[k] )
            for (var k = 0; k < Rank; k++)
            {
                if (r[k] == 0f)
                {
                    continue;
                }
                TensorPrimitives.MultiplyAdd(
                    _b.AsSpan(k * OutDim, OutDim),
                    scale * r[k],
                    result.Slice(0, OutDim),
                    result.Slice(0, OutDim));
            }
        }

        public void ZeroGrad()
        {
            _gradA.AsSpan().Clear();
            _gradB.AsSpan().Clear();
        }

        // ── Serialization ─────────────────────────────────────────────────────

        public void Save(BinaryWriter w)
        {
            w.Write(InDim);
            w.Write(OutDim);
            w.Write(Rank);
            foreach (var v in _a)
            {
                w.Write(v);
            }
            foreach (var v in _b)
            {
                w.Write(v);
            }
        }

        public static LoRAWeight Load(BinaryReader r)
        {
            var inDim = r.ReadInt32();
            var outDim = r.ReadInt32();
            var rank = r.ReadInt32();

            // Three unvalidated file fields going straight into a constructor that allocates
            // (inDim*rank + rank*outDim) floats four times over. The constructor's
            // ThrowIfNegativeOrZero catches the degenerate values and nothing catches the large ones:
            // 30000 x 30000 x 8 is a well-formed-looking header and 28 GB of allocation.
            //
            // Bounded against the bytes left in the file, which is what the loop below actually reads —
            // the same guard shape as RepackedWeightsFile and LlamaLoRAAdapter, both of which sit one
            // directory away. Found by OVERFIT038 on its first inventory over the tree.
            RequireDeclaredShapeFitsInFile(r, inDim, outDim, rank);

            var lw = new LoRAWeight(inDim, outDim, rank);
            for (var i = 0; i < lw._a.Length; i++)
            {
                lw._a[i] = r.ReadSingle();
            }
            for (var i = 0; i < lw._b.Length; i++)
            {
                lw._b[i] = r.ReadSingle();
            }
            return lw;
        }

        /// <summary>
        /// Rejects a declared A/B shape that the remaining bytes cannot hold.
        ///
        /// <para>The floats this record contains are exactly what <see cref="Load"/> reads next, so the
        /// arithmetic is not an estimate: <c>(inDim*rank + rank*outDim) * 4</c> bytes have to be there. A
        /// non-seekable stream has no length to compare against, and the check degrades rather than
        /// guessing.</para>
        /// </summary>
        private static void RequireDeclaredShapeFitsInFile(BinaryReader r, int inDim, int outDim, int rank)
        {
            if (!r.BaseStream.CanSeek)
            {
                return;
            }

            var remaining = r.BaseStream.Length - r.BaseStream.Position;
            var needed = ((long)inDim * rank + (long)rank * outDim) * sizeof(float);

            if (inDim > 0 && outDim > 0 && rank > 0 && needed <= remaining)
            {
                return;
            }

            throw new OverfitFormatException(
                $"LoRA weight declares {inDim}x{rank} and {rank}x{outDim}, needing {needed} bytes, "
                + $"and {remaining} bytes remain in the file.");
        }

        // ── Private ───────────────────────────────────────────────────────────

        private void InitializeA(Random rng)
        {
            // Kaiming uniform: U(-bound, bound) where bound = 1/sqrt(rank)
            var bound = 1f / MathF.Sqrt(Rank);
            for (var i = 0; i < _a.Length; i++)
            {
                _a[i] = ((float)rng.NextDouble() * 2f - 1f) * bound;
            }
        }
    }
}
