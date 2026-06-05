// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Exceptions
{
    /// <summary>
    /// An operation is invalid in the current state (e.g. the KV cache is full, a session is empty, a
    /// normalizer isn't fitted yet) or a feature / format / operator is not supported by the runtime. The
    /// Overfit-domain replacement for <see cref="System.InvalidOperationException"/> and
    /// <see cref="System.NotSupportedException"/> at the runtime's operation sites.
    /// </summary>
    public class OverfitRuntimeException : OverfitException
    {
        public OverfitRuntimeException()
        {
        }

        public OverfitRuntimeException(string message)
            : base(message)
        {
        }

        public OverfitRuntimeException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}
