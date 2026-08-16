// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// How much a difference obliges the reader to do, worst last.
    ///
    /// <para><b>The ordering is the product.</b> A flat list of differences is not usable — every rebuild
    /// produces one. An ordered classification is, because the only question anyone actually asks is "what is
    /// the worst thing in here", and <see cref="AssemblyComparison.HighestLevel"/> answers it in one value.</para>
    ///
    /// <para><b>Why <see cref="SilentBehaviourChange"/> ranks above <see cref="BinaryBreaking"/></b>, which
    /// looks wrong at first. A binary break is loud: the consumer gets a
    /// <see cref="MissingMethodException"/> or a <see cref="TypeLoadException"/> the first time it runs, and
    /// somebody investigates. A changed <c>const</c> or a changed optional-parameter default is baked into the
    /// consumer's own IL at <i>their</i> compile time — they do not fail, they do not rebuild, they are simply
    /// already wrong, and nothing anywhere reports it. That is the same failure shape the rest of this
    /// repository is organised against: a defect whose signature is silence.</para>
    /// </summary>
    internal enum ChangeLevel
    {
        /// <summary>Nothing differs, not even the build stamps.</summary>
        None = 0,

        /// <summary>
        /// Present on every rebuild of identical source and therefore never evidence of anything: MVID, PE
        /// timestamp and checksum, <c>AssemblyInformationalVersion</c>, strong-name signature, Authenticode
        /// certificate table, PDB checksum.
        /// </summary>
        Inert = 1,

        /// <summary>Compiled logic moved; the externally visible surface did not. <i>Retest, tell nobody.</i></summary>
        InternalOnly = 2,

        /// <summary>New surface that breaks nobody: a new type, a new method on a class.</summary>
        Additive = 3,

        /// <summary>
        /// The consumer fails to compile when it next rebuilds; already-compiled binaries keep working.
        /// The least disruptive break — the consumer can fix their own source.
        /// </summary>
        SourceBreaking = 4,

        /// <summary>
        /// Already-compiled consumers fail at run time with <see cref="MissingMethodException"/> or
        /// <see cref="TypeLoadException"/>, whether or not they rebuild.
        /// </summary>
        BinaryBreaking = 5,

        /// <summary>
        /// The consumer's behaviour changes with no error, no exception and no rebuild — a changed
        /// <c>const</c>, optional-parameter default, or enum member value, each of which is copied into the
        /// consumer's IL at their compile time.
        /// </summary>
        SilentBehaviourChange = 6,
    }
}
