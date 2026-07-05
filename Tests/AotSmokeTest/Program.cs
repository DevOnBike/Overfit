// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

// AOT smoketest entry point. Touches a slice of the DevOnBike.Overfit public
// surface that is safe to exercise without IO so that ILCompiler has a real
// reachable call graph to analyse. Extend cautiously: each new touched type or
// method widens the trim/AOT verification scope (good), but may surface trim
// warnings that block publish until fixed in the library (also good — that's
// the point of the smoketest).
//
// Exit code is the AOT verdict: 0 = pass, 1 = unexpected exception.

using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Contracts;

try
{
    Console.WriteLine("Overfit AOT smoketest — start");

    // Force the trim/AOT analyser to look at the OverfitClient type metadata.
    var clientType = typeof(OverfitClient);
    Console.WriteLine($"  OverfitClient type loaded: {clientType.FullName}");

    // Exercise the GenerationOptions value type and a static SamplingOptions
    // entry point without performing any IO. These are the smallest non-trivial
    // public surface paths that surface trim/AOT warnings if a struct, default
    // value, or sampling preset uses a dynamic-code-requiring API.
    var sampling = SamplingOptions.Greedy;
    var options = new GenerationOptions(
        maxNewTokens: 16,
        maxContextLength: 64,
        sampling: sampling,
        stopOnEndOfTextToken: true,
        endOfTextTokenId: -1);

    Console.WriteLine($"  GenerationOptions ok — maxNew={options.MaxNewTokens}, ctx={options.MaxContextLength}");

    Console.WriteLine("Overfit AOT smoketest — ok");
    return 0;
}
catch (Exception ex)
{
    Console.Error.WriteLine($"Overfit AOT smoketest — FAILED: {ex}");
    return 1;
}
