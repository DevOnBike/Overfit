// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Redaction
{
    /// <summary>
    /// What the gateway does when a category is detected in an outbound payload. Ordered by severity so a request's
    /// overall verdict is the max action across its matches.
    /// </summary>
    public enum RedactionAction
    {
        /// <summary>Leave the value in the payload, untouched and unlogged.</summary>
        Allow = 0,

        /// <summary>Leave the value in the payload but record it in the audit (visibility without blocking).</summary>
        Alert = 1,

        /// <summary>Replace the value with a placeholder before forwarding; restore on the response.</summary>
        Redact = 2,

        /// <summary>Refuse the whole request — the value must never leave the box, not even redacted.</summary>
        Block = 3
    }
}
