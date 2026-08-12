// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>One difference in what consumers of the assembly can see and bind to.</summary>
    internal sealed class ApiDifference
    {
        internal ApiDifference(DifferenceKind kind, ApiMember left, ApiMember right)
        {
            Kind = kind;
            Left = left;
            Right = right;
        }

        internal DifferenceKind Kind { get; }

        /// <summary>The member as it was, or <see langword="null"/> for an addition.</summary>
        internal ApiMember Left { get; }

        /// <summary>The member as it is now, or <see langword="null"/> for a removal.</summary>
        internal ApiMember Right { get; }

        /// <summary>The member key both sides agreed on, or the one side that exists.</summary>
        internal string MatchKey
        {
            get { return (Right ?? Left).MatchKey; }
        }

        public override string ToString()
        {
            return Kind switch
            {
                DifferenceKind.Added => "Added: " + Right.Descriptor,
                DifferenceKind.Removed => "Removed: " + Left.Descriptor,
                _ => "Changed: " + Left.Descriptor + "  ->  " + Right.Descriptor,
            };
        }
    }
}
