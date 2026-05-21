// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Chat
{
    /// <summary>The prompt formats Overfit can render without a Jinja engine.</summary>
    public enum ChatTemplateFormat
    {
        /// <summary>ChatML — Qwen, Yi, many others: <c>&lt;|im_start|&gt;role\n…&lt;|im_end|&gt;</c>.</summary>
        ChatML,

        /// <summary>Llama-3: <c>&lt;|start_header_id|&gt;role&lt;|end_header_id|&gt;…&lt;|eot_id|&gt;</c>.</summary>
        Llama3,

        /// <summary>Mistral / Llama-2 instruct: <c>[INST] … [/INST]</c>.</summary>
        Mistral,
    }
}
