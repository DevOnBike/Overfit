// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Demo.LocalAgent.Chat
{
    /// <summary>Request body for <c>POST /chat</c> and <c>POST /chat/json</c>. <paramref name="Schema"/> is an
    /// optional JSON-Schema (text) for <c>/chat/json</c>: when supplied, the reply is constrained to conform to
    /// it (typed/required/enum fields), not merely to be well-formed JSON.</summary>
    public record ChatRequest(string Message, string? Schema = null);
}
