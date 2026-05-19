// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.LanguageModels.Runtime
{
    /// <summary>
    /// Q8_0-style 8-bit quantized dot product — the compute primitive for the
    /// quantized decode matmul (step 2 of the decode-kernel plan, ROADMAP
    /// "Slot 2b").
    ///
    /// A vector is quantized in blocks of <see cref="BlockSize"/>: each block is
    /// 32 <see cref="sbyte"/> quants plus one <see cref="float"/> scale, so a
    /// value reconstructs as <c>scale * quant</c>. <see cref="Dot"/> contracts
    /// two quantized vectors with an INT8 SIMD inner product (AVX2
    /// <c>vpmaddubsw</c> + <c>vpmaddwd</c>), accumulating INT32 within a block
    /// and applying the F32 scales once per block — the scheme llama.cpp's ggml
    /// uses for Q8_0.
    ///
    /// Standalone and F32-parity-tested; wiring it into the decode weight path
    /// (output-major quantized storage, the GGUF loader) is a later sub-step.
    /// </summary>
    public static unsafe class Q8DotKernel
    {
        /// <summary>Elements per quantization block — one 256-bit SIMD lane of sbytes.</summary>
        public const int BlockSize = 32;

        /// <summary>
        /// Quantizes <paramref name="source"/> (length a multiple of
        /// <see cref="BlockSize"/>) into per-element <paramref name="quants"/>
        /// and per-block <paramref name="scales"/>. Symmetric Q8_0:
        /// <c>scale = absmax / 127</c>, <c>quant = round(value / scale)</c>.
        /// </summary>
        public static void Quantize(
            ReadOnlySpan<float> source,
            Span<sbyte> quants,
            Span<float> scales)
        {
            if (source.Length % BlockSize != 0)
            {
                throw new ArgumentException(
                    $"Source length ({source.Length}) must be a multiple of {BlockSize}.",
                    nameof(source));
            }

            var blocks = source.Length / BlockSize;

            if (quants.Length < source.Length)
            {
                throw new ArgumentException("Quants span is smaller than source.", nameof(quants));
            }

            if (scales.Length < blocks)
            {
                throw new ArgumentException("Scales span is smaller than the block count.", nameof(scales));
            }

            for (var blk = 0; blk < blocks; blk++)
            {
                var offset = blk * BlockSize;
                var block = source.Slice(offset, BlockSize);

                var absMax = 0f;
                for (var i = 0; i < BlockSize; i++)
                {
                    var a = MathF.Abs(block[i]);
                    if (a > absMax)
                    {
                        absMax = a;
                    }
                }

                var scale = absMax / 127f;
                scales[blk] = scale;

                var inverse = scale > 0f ? 1f / scale : 0f;
                for (var i = 0; i < BlockSize; i++)
                {
                    var q = Math.Clamp(MathF.Round(block[i] * inverse), -127f, 127f);
                    quants[offset + i] = (sbyte)q;
                }
            }
        }

        /// <summary>
        /// Contracts two Q8-quantized vectors of <paramref name="length"/>
        /// elements (a multiple of <see cref="BlockSize"/>):
        /// <c>Σ_b (aScales[b]·bScales[b]) · INT8dot(aBlock[b], bBlock[b])</c>.
        /// Approximates <c>Σ a[i]·b[i]</c> within Q8 quantization noise.
        /// </summary>
        public static float Dot(
            ReadOnlySpan<sbyte> aQuants,
            ReadOnlySpan<float> aScales,
            ReadOnlySpan<sbyte> bQuants,
            ReadOnlySpan<float> bScales,
            int length)
        {
            if (length % BlockSize != 0)
            {
                throw new ArgumentException(
                    $"Length ({length}) must be a multiple of {BlockSize}.", nameof(length));
            }

            var blocks = length / BlockSize;

            if (aQuants.Length < length || bQuants.Length < length)
            {
                throw new ArgumentException("A quant span is smaller than length.");
            }

            if (aScales.Length < blocks || bScales.Length < blocks)
            {
                throw new ArgumentException("A scale span is smaller than the block count.");
            }

            ref var aRef = ref MemoryMarshal.GetReference(aQuants);
            ref var bRef = ref MemoryMarshal.GetReference(bQuants);

            var result = 0f;
            for (var blk = 0; blk < blocks; blk++)
            {
                var offset = blk * BlockSize;

                var blockDot = Int8BlockDot(
                    Vector256.LoadUnsafe(ref Unsafe.Add(ref aRef, offset)),
                    Vector256.LoadUnsafe(ref Unsafe.Add(ref bRef, offset)));

                result += aScales[blk] * bScales[blk] * blockDot;
            }

            return result;
        }

        /// <summary>
        /// Quantized single-token projection: <c>output = bias + input @ W</c>,
        /// where <paramref name="weightQuants"/> / <paramref name="weightScales"/>
        /// hold W as Q8_0 in <b>output-major</b> layout — row <c>o</c> is the
        /// <paramref name="inputSize"/>-long contraction vector for output
        /// <c>o</c>, quantized in <c>inputSize / BlockSize</c> blocks.
        ///
        /// The F32 <paramref name="input"/> is quantized once into the
        /// caller-owned scratch, then each output is a single <see cref="Dot"/>.
        /// Approximates the F32 projection within Q8 quantization noise. This is
        /// the sequential reference shape; the decode path parallelises the
        /// output loop over <c>OverfitParallelFor</c>.
        /// </summary>
        public static void Project(
            ReadOnlySpan<float> input,
            ReadOnlySpan<sbyte> weightQuants,
            ReadOnlySpan<float> weightScales,
            ReadOnlySpan<float> bias,
            Span<float> output,
            int inputSize,
            int outputSize,
            Span<sbyte> inputQuants,
            Span<float> inputScales)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputSize);

            if (inputSize % BlockSize != 0)
            {
                throw new ArgumentException(
                    $"inputSize ({inputSize}) must be a multiple of {BlockSize}.", nameof(inputSize));
            }

            var blocksPerRow = inputSize / BlockSize;

            if (input.Length < inputSize)
            {
                throw new ArgumentException("Input span is smaller than inputSize.", nameof(input));
            }
            if (weightQuants.Length < (long)outputSize * inputSize)
            {
                throw new ArgumentException("Weight quants span is smaller than outputSize * inputSize.", nameof(weightQuants));
            }
            if (weightScales.Length < (long)outputSize * blocksPerRow)
            {
                throw new ArgumentException("Weight scales span is smaller than outputSize * blocksPerRow.", nameof(weightScales));
            }
            if (output.Length < outputSize)
            {
                throw new ArgumentException("Output span is smaller than outputSize.", nameof(output));
            }
            if (!bias.IsEmpty && bias.Length < outputSize)
            {
                throw new ArgumentException("Bias span is smaller than outputSize.", nameof(bias));
            }
            if (inputQuants.Length < inputSize || inputScales.Length < blocksPerRow)
            {
                throw new ArgumentException("Input quantization scratch is too small.");
            }

            // Quantize the activation once — reused for every output dot.
            Quantize(input.Slice(0, inputSize), inputQuants, inputScales);

            for (var o = 0; o < outputSize; o++)
            {
                var dot = Dot(
                    weightQuants.Slice(o * inputSize, inputSize),
                    weightScales.Slice(o * blocksPerRow, blocksPerRow),
                    inputQuants,
                    inputScales,
                    inputSize);

                output[o] = bias.IsEmpty ? dot : bias[o] + dot;
            }
        }

        /// <summary>
        /// Parallel quantized projection — <see cref="Project"/> with the output
        /// loop split across the zero-allocation <c>OverfitParallelFor</c> worker
        /// pool. The activation is quantized once (sequentially) into the
        /// caller-owned scratch, then each worker computes a disjoint band of
        /// output dots. Bit-identical to <see cref="Project"/>.
        /// </summary>
        public static void ProjectParallel(
            ReadOnlySpan<float> input,
            Q8Weight weight,
            ReadOnlySpan<float> bias,
            Span<float> output,
            Span<sbyte> inputQuants,
            Span<float> inputScales)
        {
            ArgumentNullException.ThrowIfNull(weight);

            var inputSize = weight.InputSize;
            var outputSize = weight.OutputSize;
            var blocksPerRow = inputSize / BlockSize;

            if (input.Length < inputSize)
            {
                throw new ArgumentException("Input span is smaller than the weight's input size.", nameof(input));
            }
            if (output.Length < outputSize)
            {
                throw new ArgumentException("Output span is smaller than the weight's output size.", nameof(output));
            }
            if (!bias.IsEmpty && bias.Length < outputSize)
            {
                throw new ArgumentException("Bias span is smaller than outputSize.", nameof(bias));
            }
            if (inputQuants.Length < inputSize || inputScales.Length < blocksPerRow)
            {
                throw new ArgumentException("Input quantization scratch is too small.");
            }

            // Quantize the activation once — read-only for every worker.
            Quantize(input.Slice(0, inputSize), inputQuants, inputScales);

            fixed (sbyte* weightQuants = weight.Quants)
            fixed (float* weightScales = weight.Scales)
            fixed (sbyte* inQuants = inputQuants)
            fixed (float* inScales = inputScales)
            fixed (float* biasPtr = bias)
            fixed (float* outputPtr = output)
            {
                var context = new Q8ProjectContext
                {
                    WeightQuants = weightQuants,
                    WeightScales = weightScales,
                    InputQuants = inQuants,
                    InputScales = inScales,
                    Bias = biasPtr,
                    BiasLength = bias.Length,
                    Output = outputPtr,
                    InputSize = inputSize,
                    BlocksPerRow = blocksPerRow,
                };

                OverfitParallelFor.For(0, outputSize, &ProjectChunk, &context);
            }
        }

        /// <summary>Worker body for <see cref="ProjectParallel"/> — one disjoint band of output rows.</summary>
        private static void ProjectChunk(int chunkStart, int chunkEnd, void* context)
        {
            ref var ctx = ref Unsafe.AsRef<Q8ProjectContext>(context);

            var activationQuants = new ReadOnlySpan<sbyte>(ctx.InputQuants, ctx.InputSize);
            var activationScales = new ReadOnlySpan<float>(ctx.InputScales, ctx.BlocksPerRow);

            for (var o = chunkStart; o < chunkEnd; o++)
            {
                var rowQuants = new ReadOnlySpan<sbyte>(
                    ctx.WeightQuants + (long)o * ctx.InputSize, ctx.InputSize);
                var rowScales = new ReadOnlySpan<float>(
                    ctx.WeightScales + (long)o * ctx.BlocksPerRow, ctx.BlocksPerRow);

                var dot = Dot(rowQuants, rowScales, activationQuants, activationScales, ctx.InputSize);
                ctx.Output[o] = ctx.BiasLength == 0 ? dot : ctx.Bias[o] + dot;
            }
        }

        private struct Q8ProjectContext
        {
            public sbyte* WeightQuants;
            public float* WeightScales;
            public sbyte* InputQuants;
            public float* InputScales;
            public float* Bias;
            public int BiasLength;
            public float* Output;
            public int InputSize;
            public int BlocksPerRow;
        }

        /// <summary>
        /// <c>Σ a[i]·b[i]</c> over one 32-lane block as INT32. The AVX2 path uses
        /// <c>vpmaddubsw</c> (unsigned×signed) reached via the <c>vpsignb</c>
        /// trick — <c>|a|·sign(b,a) == a·b</c> elementwise — then <c>vpmaddwd</c>
        /// to widen the 16-bit pair-sums to INT32. Q8 quants are in [-127,127],
        /// so the 16-bit pair-sums (≤ 2·127² = 32258) never saturate.
        /// </summary>
        private static int Int8BlockDot(Vector256<sbyte> a, Vector256<sbyte> b)
        {
            if (Avx2.IsSupported)
            {
                var absA = Avx2.Abs(a);                                  // |a|, as bytes
                var signedB = Avx2.Sign(b, a);                           // b · sign(a)
                var pairs16 = Avx2.MultiplyAddAdjacent(absA, signedB);   // 16 × int16 pair sums
                var pairs32 = Avx2.MultiplyAddAdjacent(pairs16, Vector256.Create((short)1));
                return Vector256.Sum(pairs32);
            }

            var sum = 0;
            for (var i = 0; i < Vector256<sbyte>.Count; i++)
            {
                sum += a.GetElement(i) * b.GetElement(i);
            }

            return sum;
        }
    }
}
