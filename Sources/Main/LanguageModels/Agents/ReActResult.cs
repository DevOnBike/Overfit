// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Generic;

namespace DevOnBike.Overfit.LanguageModels.Agents
{
    /// <summary>Result of running a <c>ReActAgent</c> loop: final answer + per-step trace + exit reason.</summary>
    public sealed class ReActResult
    {
        public ReActResult(string answer, IReadOnlyList<ReActStep> steps, ReActCompletion completion)
        {
            Answer = answer;
            Steps = steps;
            Completion = completion;
        }

        /// <summary>The final answer (from the <c>finish</c> tool), or an exhaustion message on step-cap exit.</summary>
        public string Answer { get; }

        /// <summary>Per-turn trace: each tool call + observation. Useful for debugging / audit logs.</summary>
        public IReadOnlyList<ReActStep> Steps { get; }

        public ReActCompletion Completion { get; }
    }
}
