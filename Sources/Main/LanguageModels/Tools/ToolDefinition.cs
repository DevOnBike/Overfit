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
    /// Argument typing is left to the handler: the <see cref="ToolCallConstraint"/> guarantees a
    /// well-formed JSON arguments object, and the C# side validates/deserializes it. Schema-typed
    /// argument enforcement is the JSON-Schema follow-on.
    /// </summary>
    public sealed class ToolDefinition
    {
        public ToolDefinition(string name, string description)
        {
            ArgumentException.ThrowIfNullOrEmpty(name);
            Name = name;
            Description = description ?? string.Empty;
        }

        /// <summary>The tool's identifier — the model is forced to emit exactly one of these.</summary>
        public string Name { get; }

        /// <summary>What the tool does / when to use it. Used in the prompt, not enforced.</summary>
        public string Description { get; }
    }
}
