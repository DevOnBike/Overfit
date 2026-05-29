// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Demo.LocalAgent.Chat
{
    /// <summary>Request body for <c>POST /chat</c> and <c>POST /chat/json</c>.</summary>
    public record ChatRequest(string Message);
}
