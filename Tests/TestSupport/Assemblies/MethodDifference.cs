// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// One method whose compiled body is not the same on both sides.
    ///
    /// <para>This is the answer to "if the IL changed, <i>where</i>". "The IL changed" tells a reader to go and
    /// benchmark everything; a list of method keys tells them whether the change is anywhere near the path
    /// they care about.</para>
    /// </summary>
    internal sealed class MethodDifference
    {
        internal MethodDifference(string method, DifferenceKind kind)
        {
            Method = method;
            Kind = kind;
        }

        /// <summary>The method key: declaring type, name, generic arity, parameter types and return type.</summary>
        internal string Method
        {
            get;
        }

        internal DifferenceKind Kind
        {
            get;
        }

        public override string ToString()
        {
            return Kind + ": " + Method;
        }
    }
}
