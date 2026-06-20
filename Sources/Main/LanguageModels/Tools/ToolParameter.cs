// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Tools
{
    /// <summary>
    /// One required argument of a <see cref="ToolDefinition"/>. When a tool declares parameters, the
    /// <see cref="ToolCallConstraint"/> forces the <c>arguments</c> object to contain exactly these keys,
    /// in this order, each with a value of the declared <see cref="Kind"/> — so the model cannot invent
    /// key names, omit a required argument, or emit the wrong value type. (Parameters are required and
    /// ordered; optional/defaulted arguments are a follow-on.)
    /// </summary>
    public sealed class ToolParameter
    {
        public ToolParameter(string name, ToolParameterKind kind = ToolParameterKind.String)
        {
            ArgumentException.ThrowIfNullOrEmpty(name);
            Name = name;
            Kind = kind;
        }

        /// <summary>The argument's JSON key — emitted verbatim by the constraint.</summary>
        public string Name
        {
            get;
        }

        /// <summary>The JSON type the value is constrained to.</summary>
        public ToolParameterKind Kind
        {
            get;
        }
    }
}
