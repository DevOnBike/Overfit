// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Skills.Optimization
{
    /// <summary>A training case the current skill got wrong — the prompt, what the skill actually produced, and
    /// <b>why that counted as a failure</b>. The optimizer reasons over these to propose a targeted edit.
    ///
    /// <para><paramref name="Reason"/> is what makes an edit DIRECTED rather than random. Measured on
    /// Qwen2.5-3B as the editor, identical failing case, the reason being the only difference:</para>
    /// <code>
    /// without reason -> "Answer the question in the format 'The [subject] is [answer]'."   // WRONG DIRECTION
    /// with reason    -> "Answer the question in at most 4 words."                          // correct
    /// </code>
    /// <para>Given only (prompt, output) the editor sees an answer that reads perfectly fine, so it codifies that
    /// answer as the desired format — optimizing AWAY from the grader. Which check failed, and its note, is the
    /// only gradient information the editor ever gets. Optional so existing two-argument callers keep compiling;
    /// an empty reason simply restores the old, blind behaviour.</para>
    /// </summary>
    public sealed record CaseFailure(string Prompt, string Output, string Reason = "");
}
