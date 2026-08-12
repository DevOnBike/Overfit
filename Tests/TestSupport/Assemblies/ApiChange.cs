// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// One classified finding: what changed, how bad it is, and which published rule says so.
    ///
    /// <para><b><see cref="Message"/> has to stand on its own.</b> Whoever reads this is deciding whether to
    /// ship, and "CP0006" is not a decision. The message names the type, the member, and the consequence in the
    /// terms the consumer will experience it — a compile error on rebuild, an exception at run time, or a
    /// wrong answer with no symptom at all.</para>
    /// </summary>
    internal sealed class ApiChange
    {
        internal ApiChange(ChangeLevel level, string ruleId, string target, string message)
        {
            Level = level;
            RuleId = ruleId;
            Target = target;
            Message = message;
        }

        internal ChangeLevel Level { get; }

        /// <summary>
        /// This tool's rule id, with the equivalent <c>Microsoft.DotNet.ApiCompat</c> diagnostic in brackets
        /// where one exists — so a disagreement with the tool Microsoft ships can be looked up rather than
        /// argued about.
        /// </summary>
        internal string RuleId { get; }

        /// <summary>The type or member the finding is about.</summary>
        internal string Target { get; }

        internal string Message { get; }

        public override string ToString()
        {
            return "[" + (int)Level + " " + Level + "] " + RuleId + " " + Target + " — " + Message;
        }
    }
}
