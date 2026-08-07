// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests
{
    /// <summary>
    /// Named prerequisites a <see cref="FixtureFact"/> can require.
    ///
    /// <para><b>Why an enum and not a path.</b> Attribute arguments must be compile-time constants, and
    /// every one of these is produced at runtime — <c>Path.Combine(TestModelPaths.Qwen3B.Dir, ...)</c>
    /// reads an environment variable, <c>LocateOverfitExe()</c> walks the directory tree. An enum member
    /// *is* a constant, so it can travel in the attribute while the resolution stays in one place next to
    /// <see cref="TestModelPaths"/>. The alternative — copying a path literal into the attribute — is how a
    /// test comes to skip for one reason and fail for another.</para>
    /// </summary>
    internal enum TestFixture
    {
        /// <summary>Qwen2.5 weights in safetensors form, beside the GGUF.</summary>
        QwenSafetensors,

        /// <summary>The <c>tokenizer.json</c> the Qwen tokenizer tests load.</summary>
        QwenTokenizerJson,

        /// <summary>MNIST training images — the idx file, not the directory.</summary>
        MnistTrainingImages,

        /// <summary>Reference vectors produced by sentence-transformers, for MiniLM parity.</summary>
        MiniLmReferenceEmbeddings,

        /// <summary>Reference vectors for BGE parity.</summary>
        BgeReferenceEmbeddings,

        /// <summary>Reference vectors for E5 parity.</summary>
        E5ReferenceEmbeddings,

        /// <summary>The Qwen-3B binary checkpoint plus the tokenizer the diagnostics load beside it.</summary>
        Qwen3BBinaryAndTokenizer,

        /// <summary>
        /// GPT-2 Small in both forms at once — <c>model.safetensors</c> and <c>gpt2_small.bin</c>.
        ///
        /// <para>Both, because the test that needs it compares one loader against the other. With only one
        /// present there is nothing to compare, and a pass would report that the two agree.</para>
        /// </summary>
        Gpt2SafetensorsAndBinary,

        /// <summary>The MiniLM sentence-transformer weights.</summary>
        MiniLmSafetensors,

        /// <summary>
        /// The Qwen1.5-MoE GGUF <b>and</b> a reference Qwen tokenizer to cross-check it against —
        /// either <c>tokenizer.json</c> or <c>vocab.json</c> will do.
        ///
        /// <para>The either-or is why this needs its own entry rather than a list of required files:
        /// <see cref="ModelFact"/> requires every path it is given, and here one of two suffices.</para>
        /// </summary>
        QwenMoeGgufAndReferenceTokenizer,

        /// <summary>The Qwen-3B Q4_K_M GGUF.</summary>
        Qwen3BQ4KmGguf,

        /// <summary>The Qwen-3B <c>tokenizer.json</c> on its own.</summary>
        Qwen3BTokenizerJson,

        /// <summary>
        /// AVX2 and FMA on the running CPU.
        ///
        /// <para><b>Not a fixture at all, and that is why it is worth naming here.</b> These tests measure
        /// or verify x86 SIMD kernels; on a machine without the instructions there is nothing to measure,
        /// and the honest outcome is "not run" rather than a pass. Grouping it with the model fixtures is
        /// deliberate: the question a reader asks is the same — <i>was this actually checked?</i> — and it
        /// deserves the same answer in the same place, rather than a second mechanism nobody remembers.</para>
        /// </summary>
        Avx2AndFma,

        /// <summary>
        /// A built <c>overfit</c> CLI under <c>Sources/Cli/bin</c>.
        ///
        /// <para>Different in kind from the others: it is not a downloaded artifact but a build output, so
        /// its absence usually means "nobody built the CLI in this configuration" rather than "this box
        /// lacks a fixture". The skip message says so, because the two need different responses.</para>
        /// </summary>
        OverfitCli,
    }
}
