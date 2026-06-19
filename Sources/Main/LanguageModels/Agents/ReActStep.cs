// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Agents
{
    /// <summary>
    /// One turn of a ReAct loop: the tool the model called, the JSON arguments it supplied, and the
    /// observation (handler output or extracted final answer) that came back.
    /// </summary>
    public sealed class ReActStep
    {
        public ReActStep(string toolName, string argumentsJson, string observation, bool finished)
        {
            ToolName = toolName;
            ArgumentsJson = argumentsJson;
            Observation = observation;
            Finished = finished;
        }

        public string ToolName
        {
            get;
        }

        /// <summary>The raw JSON object the model emitted as the tool's arguments.</summary>
        public string ArgumentsJson
        {
            get;
        }

        /// <summary>
        /// What the handler returned (the next-turn user message), or — for the <c>finish</c> step —
        /// the extracted final answer.
        /// </summary>
        public string Observation
        {
            get;
        }

        /// <summary>True only for the final step (the model called the synthetic <c>finish</c> tool).</summary>
        public bool Finished
        {
            get;
        }
    }
}
