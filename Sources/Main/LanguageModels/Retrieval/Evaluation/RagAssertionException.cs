// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.Retrieval.Evaluation
{
    /// <summary>
    /// Thrown by <see cref="RagAssert"/> when a RAG quality gate fails. Framework-agnostic: xUnit / NUnit /
    /// MSTest all treat a thrown exception as a failed test, and the actionable detail (which queries missed,
    /// which groups are brittle, which traps sprung) lives in the message.
    /// </summary>
    public sealed class RagAssertionException : OverfitException
    {
        public RagAssertionException(string message)
            : base(message)
        {
        }
    }
}
