// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// A difference that is present on every rebuild of identical source and says nothing about behaviour.
    ///
    /// <para><b>This category is the product.</b> A tool that reports "the files differ" is worthless, because
    /// two builds of the same commit differ: the MVID is a fresh GUID each time, the PE header carries a
    /// timestamp, the checksum follows the timestamp, a signed package carries a certificate table sized by
    /// the signing service, and a source-linked build stamps its commit hash into
    /// <c>AssemblyInformationalVersion</c>. Every one of those changed between System.Numerics.Tensors 10.0.10
    /// and 10.0.11 while the IL region differed by zero bytes.</para>
    ///
    /// <para><b>Inert means "cannot change execution", not "uninteresting".</b> A changed
    /// <c>AssemblyInformationalVersion</c> is exactly how you tell that two artefacts are different builds at
    /// all — which is the evidence that the comparison was run on the two files intended, rather than twice on
    /// the same one. It is reported, and it is kept out of the verdict.</para>
    /// </summary>
    internal sealed class InertDifference
    {
        internal InertDifference(string kind, string left, string right)
        {
            Kind = kind;
            Left = left;
            Right = right;
        }

        /// <summary>What differs: <c>Mvid</c>, <c>PE.TimeDateStamp</c>, <c>AssemblyInformationalVersion</c>, …</summary>
        internal string Kind
        {
            get;
        }

        internal string Left
        {
            get;
        }

        internal string Right
        {
            get;
        }

        public override string ToString()
        {
            return Kind + ": " + Left + " -> " + Right;
        }
    }
}
