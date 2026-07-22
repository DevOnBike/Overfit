// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Benchmarks.Helpers;
using DevOnBike.Overfit.LanguageModels.Loading;
using DevOnBike.Overfit.LanguageModels.Runtime;

namespace Benchmarks
{
    /// <summary>
    /// Settles one question for the decode path: is the Q4_K GEMV bound by <b>raw memory bandwidth</b> or by
    /// <b>dequantization compute</b>?
    ///
    /// <para>Decode reads the entire weight matrix once per token and does ~one MAC per weight byte, so its
    /// arithmetic intensity is tiny and it should be bandwidth-bound. But a whole-model estimate put decode at
    /// ≈46 GB/s against a measured 90 GB/s DRAM read ceiling — only ~51%. If the GEMV kernel itself streams its
    /// weight near the ceiling, that shortfall is per-token overhead (attention, LM head, sampling) and a wider
    /// kernel buys nothing. If the GEMV runs well under the ceiling, the dequant compute cannot consume bytes
    /// as fast as memory delivers them, and an AVX-512 / VNNI decode kernel could raise utilisation — the one
    /// case where the reverted "AVX-512 decode port" negative would not apply.</para>
    ///
    /// <para>The <c>GB/s</c> column (from <see cref="WorkAmount.Memory"/>) reports the repacked weight bytes
    /// streamed per call; compare it directly against <c>MachineRooflineBenchmark.ReadBandwidth</c> (~90 GB/s).
    /// The shape is one FFN projection at Qwen-3B dimensions; decode processes a single activation row.</para>
    ///
    /// Run:
    ///   dotnet run -c Release --project Sources/Benchmark -- --filter "*DecodeGemvRoofline*"
    /// </summary>
    [Config(typeof(BenchmarkConfig))]
    public class DecodeGemvRooflineBenchmark
    {
        private const int InputSize = 2048;

        /// <summary>
        /// 11008 keeps the repacked weight (~12.7 MB) inside this box's 128 MB L3, so the kernel runs
        /// compute-bound with hot data — that number is the kernel's own throughput ceiling. 176128 makes the
        /// weight ~203 MB, past L3, so the kernel streams from DRAM — that number is what decode actually sees.
        /// The pair separates the kernel's compute rate from the memory rate it is fed.
        /// </summary>
        [Params(11008, 176128)]
        public int OutputSize
        {
            get; set;
        }

        private Q4KWeight _q4k = null!;
        private byte[] _repacked = null!;
        private sbyte[] _quants = null!;
        private float[] _scales = null!;
        private short[] _bsums = null!;
        private float[] _output = null!;

        public static WorkAmount GetWorkAmount(BenchmarkCase benchmarkCase)
        {
            // Bytes actually streamed: the repacked weight, read once per GEMV.
            var outputSize = (int)benchmarkCase.Parameters["OutputSize"];
            var repackedBytes = (long)(outputSize / 8) * (InputSize / 256) * Q4KRepack.BlockKx8Bytes;

            return WorkAmount.Memory(repackedBytes);
        }

        [GlobalSetup]
        public void Setup()
        {
            var rng = new Random(20260723);

            var f32 = new float[(long)OutputSize * InputSize];
            for (var i = 0; i < f32.Length; i++)
            {
                f32[i] = (float)((rng.NextDouble() * 2.0) - 1.0);
            }

            _q4k = new Q4KWeight(GgmlQuant.QuantizeQ4_K(f32, InputSize, OutputSize), InputSize, OutputSize);
            _repacked = _q4k.EnsureRepacked().ToArray();

            var spr = _q4k.SuperBlocksPerRow;
            var input = new float[InputSize];
            for (var i = 0; i < input.Length; i++)
            {
                input[i] = (float)((rng.NextDouble() * 2.0) - 1.0);
            }

            _quants = new sbyte[InputSize];
            _scales = new float[spr];
            _bsums = new short[spr * Q4KDotKernel.GroupsPerSuperBlock];
            Q4KDotKernel.QuantizeActivationQ8K(input, _quants, _scales, _bsums);

            _output = new float[OutputSize];
        }

        /// <summary>The production decode kernel: AVX2 8×8 GEMV, one activation row, parallel over row-groups.</summary>
        [Benchmark]
        public void DecodeGemvParallel()
        {
            Q4KGemvKernel.GemvParallel(_repacked, OutputSize, InputSize, _quants, _scales, _bsums, _output);
        }
    }
}
