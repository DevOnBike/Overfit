// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Monitoring
{
    /// <summary>
    /// Destination for alert events. Implement to integrate with PagerDuty,
    /// Teams webhooks, Slack, or any other notification channel.
    /// </summary>
    public interface IAlertSink
    {
        /// <summary>
        /// Sends the alert event to the destination.
        /// Implementations must not throw — catch internally and log/swallow.
        /// </summary>
        Task SendAsync(AlertEvent alert, CancellationToken ct = default);
    }
}