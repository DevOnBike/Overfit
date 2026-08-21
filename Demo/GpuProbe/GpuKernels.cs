// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using ILGPU;

namespace DevOnBike.Overfit.GpuProbe
{
    /// <summary>
    /// The device kernels. All three are FP32 and none of them dequantizes: the weight arrives already
    /// decoded to F32, which is why the honest CPU counterpart for the forward is arm C3 and not C1.
    /// <para>
    /// Layouts, fixed for the whole probe: <c>input[n, k]</c> row-major, <c>weight[m, k]</c> row-major
    /// (output-major, one row per output feature — the layout <c>IDequantRowSource.DecodeRow</c> hands
    /// back), <c>output[n, m]</c> row-major.
    /// </para>
    /// </summary>
    internal static class GpuKernels
    {
        /// <summary>Tile edge. The group is <c>TileSize * TileSize</c> threads.</summary>
        public const int TileSize = 16;

        /// <summary>
        /// Shared-memory row pitch. One float wider than the tile so that a column walk
        /// (<c>tile[t * Padded + j]</c> with t varying) does not land every thread on the same bank.
        /// </summary>
        private const int Padded = TileSize + 1;

        /// <summary>
        /// G1: one thread per output element. The floor — what a first port gets before anyone thinks
        /// about memory. Each thread streams a whole weight row, so the row is re-read once per batch
        /// element and nothing is shared.
        /// </summary>
        public static void NaiveForward(
            Index2D index,
            ArrayView<float> input,
            ArrayView<float> weight,
            ArrayView<float> output,
            int n,
            int k,
            int m)
        {
            var o = index.X;
            var b = index.Y;
            if (b >= n || o >= m)
            {
                return;
            }

            var inBase = (long)b * k;
            var wBase = (long)o * k;
            var acc = 0f;
            for (var i = 0; i < k; i++)
            {
                acc += input[inBase + i] * weight[wBase + i];
            }

            output[(long)b * m + o] = acc;
        }

        /// <summary>
        /// G2: shared-memory tiled forward, <c>out[b,o] = sum_i in[b,i] * W[o,i]</c>. Both operands are
        /// contiguous along the contraction dimension, so a tile load with the thread's X index walking
        /// <c>i</c> is coalesced on both sides.
        /// </summary>
        public static void TiledForward(
            ArrayView<float> input,
            ArrayView<float> weight,
            ArrayView<float> output,
            int n,
            int k,
            int m)
        {
            var tx = Group.IdxX;
            var ty = Group.IdxY;
            var oBase = Grid.IdxX * TileSize;
            var bBase = Grid.IdxY * TileSize;

            var tileIn = SharedMemory.Allocate<float>(TileSize * Padded);
            var tileW = SharedMemory.Allocate<float>(TileSize * Padded);

            var acc = 0f;
            for (var kt = 0; kt < k; kt += TileSize)
            {
                var i = kt + tx;
                var bRow = bBase + ty;
                var oRow = oBase + ty;

                tileIn[ty * Padded + tx] = bRow < n && i < k ? input[(long)bRow * k + i] : 0f;
                tileW[ty * Padded + tx] = oRow < m && i < k ? weight[(long)oRow * k + i] : 0f;
                Group.Barrier();

                for (var j = 0; j < TileSize; j++)
                {
                    acc += tileIn[ty * Padded + j] * tileW[tx * Padded + j];
                }

                Group.Barrier();
            }

            var b = bBase + ty;
            var o = oBase + tx;
            if (b < n && o < m)
            {
                output[(long)b * m + o] = acc;
            }
        }

        /// <summary>
        /// G3: the transposed direction, <c>dx[b,i] = sum_o dy[b,o] * W[o,i]</c>. Section 1 fact 2 of the
        /// plan: on the CPU both directions walk the same output-major rows, on a device the backward
        /// contracts ACROSS rows of a row-major weight, which is a different access pattern and can cost
        /// several times the forward at the same FLOP count.
        /// </summary>
        public static void TiledBackward(
            ArrayView<float> outputGrad,
            ArrayView<float> weight,
            ArrayView<float> inputGrad,
            int n,
            int k,
            int m)
        {
            var tx = Group.IdxX;
            var ty = Group.IdxY;
            var iBase = Grid.IdxX * TileSize;
            var bBase = Grid.IdxY * TileSize;

            var tileDy = SharedMemory.Allocate<float>(TileSize * Padded);
            var tileW = SharedMemory.Allocate<float>(TileSize * Padded);

            var acc = 0f;
            for (var ot = 0; ot < m; ot += TileSize)
            {
                var bRow = bBase + ty;
                var oCol = ot + tx;
                tileDy[ty * Padded + tx] = bRow < n && oCol < m ? outputGrad[(long)bRow * m + oCol] : 0f;

                var oRow = ot + ty;
                var iCol = iBase + tx;
                tileW[ty * Padded + tx] = oRow < m && iCol < k ? weight[(long)oRow * k + iCol] : 0f;
                Group.Barrier();

                for (var j = 0; j < TileSize; j++)
                {
                    acc += tileDy[ty * Padded + j] * tileW[j * Padded + tx];
                }

                Group.Barrier();
            }

            var b = bBase + ty;
            var i = iBase + tx;
            if (b < n && i < k)
            {
                inputGrad[(long)b * k + i] = acc;
            }
        }
    }
}
