// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Agents
{
    /// <summary>Why a <c>ReActAgent.Run</c> loop stopped.</summary>
    public enum ReActCompletion
    {
        /// <summary>The model called the synthetic <c>finish</c> tool with a final answer.</summary>
        Finish = 0,

        /// <summary>The loop hit its step cap without finishing — answer reflects exhaustion.</summary>
        StepCap = 1
    }
}
