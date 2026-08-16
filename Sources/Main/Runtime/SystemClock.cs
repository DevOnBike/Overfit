// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Runtime
{
    /// <summary>
    /// The real clock: the one place in the library that is allowed to read the wall clock.
    ///
    /// <para>Stateless and shared through <see cref="Instance"/>, so taking an <see cref="IClock"/> costs a
    /// field and nothing per call. This is also the single adapter point if the platform's own time
    /// abstraction changes again — see <see cref="IClock"/> for why that is not hypothetical.</para>
    /// </summary>
    public sealed class SystemClock : IClock
    {
        /// <summary>The shared instance. There is no state, so there is no reason to have a second one.</summary>
        public static readonly SystemClock Instance = new();

        private SystemClock()
        {
        }

        /// <inheritdoc />
#pragma warning disable RS0030 // THE adapter: this one line is what the ban on static clock reads redirects to.
        public DateTimeOffset UtcNow => DateTimeOffset.UtcNow;
#pragma warning restore RS0030
    }
}
