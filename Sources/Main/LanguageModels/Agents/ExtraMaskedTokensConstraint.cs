// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.LanguageModels.Agents
{
    /// <summary>
    /// Wraps an inner <see cref="ITokenConstraint"/> and additionally masks a set of token ids while
    /// the inner constraint reports <see cref="ITokenConstraint.IsComplete"/> = false. The motivating
    /// case: <c>ToolCallConstraint</c> masks its tokenizer's <c>EndOfTextTokenId</c>, but a chat
    /// runtime may stop on a DIFFERENT token (e.g. Qwen's <c>&lt;|im_end|&gt;</c> ≠ <c>&lt;|endoftext|&gt;</c>).
    /// Without this wrapper the model could emit that other id mid-envelope, the chat would stop, and
    /// the reply would be truncated and unparseable.
    /// </summary>
    internal sealed class ExtraMaskedTokensConstraint : ITokenConstraint
    {
        private readonly ITokenConstraint _inner;
        private readonly int[] _extra;

        public ExtraMaskedTokensConstraint(ITokenConstraint inner, params int[] extraTokenIds)
        {
            ArgumentNullException.ThrowIfNull(inner);
            ArgumentNullException.ThrowIfNull(extraTokenIds);
            _inner = inner;
            _extra = extraTokenIds;
        }

        public bool IsComplete => _inner.IsComplete;

        public void ApplyMask(Span<float> logits)
        {
            _inner.ApplyMask(logits);
            if (_inner.IsComplete)
            {
                return;
            }

            foreach (var id in _extra)
            {
                if (id >= 0 && (uint)id < (uint)logits.Length)
                {
                    logits[id] = float.NegativeInfinity;
                }
            }
        }

        public void Accept(int token) => _inner.Accept(token);
    }
}
