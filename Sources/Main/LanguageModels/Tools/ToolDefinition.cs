// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Tools
{
    /// <summary>
    /// A tool the model may call: its <see cref="Name"/> (the value constrained generation will
    /// force the model to pick from) and a human-readable <see cref="Description"/> (for the system
    /// prompt — it tells the model when to use the tool; it is not enforced at decode time).
    ///
    /// When <see cref="Parameters"/> is supplied, the <see cref="ToolCallConstraint"/> additionally
    /// forces the <c>arguments</c> object to contain exactly those keys, in order, each with a value
    /// of the declared type — the model cannot invent key names, omit an argument, or emit the wrong
    /// type. When it is empty, the constraint only guarantees a well-formed JSON arguments value and
    /// the C# handler validates the shape (the original behaviour, preserved for backward compatibility).
    /// </summary>
    public sealed class ToolDefinition
    {
        private static readonly ToolParameter[] _noParameters = [];

        public ToolDefinition(string name, string description)
            : this(name, description, _noParameters)
        {
        }

        public ToolDefinition(string name, string description, IReadOnlyList<ToolParameter> parameters)
        {
            ArgumentException.ThrowIfNullOrEmpty(name);
            ArgumentNullException.ThrowIfNull(parameters);
            Name = name;
            Description = description ?? string.Empty;
            Parameters = parameters;
        }

        /// <summary>The tool's identifier — the model is forced to emit exactly one of these.</summary>
        public string Name { get; }

        /// <summary>What the tool does / when to use it. Used in the prompt, not enforced.</summary>
        public string Description { get; }

        /// <summary>
        /// The required, ordered arguments. When non-empty, the constraint forces the exact key set,
        /// order and value types; when empty, the arguments are only constrained to well-formed JSON.
        /// </summary>
        public IReadOnlyList<ToolParameter> Parameters { get; }
    }
}
