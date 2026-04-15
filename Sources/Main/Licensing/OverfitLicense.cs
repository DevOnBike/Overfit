// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace DevOnBike.Overfit.Licensing
{
    public static class OverfitLicense
    {
        private static readonly Lock _lock = new();
        private volatile static bool _noticeShown;

        public static OverfitLicenseMode Mode { get; private set; } = OverfitLicenseMode.OpenSource;

        /// <summary>
        /// Suppresses the open-source notice completely.
        /// Useful for tests, benchmarks and other technical hosts.
        /// </summary>
        public static bool SuppressNotice { get; set; }

        /// <summary>
        /// Optional custom sink for license/info messages.
        /// If not set, Console + Debug will be used.
        /// </summary>
        public static Action<string>? MessageSink { get; set; }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static void EnsureNotified()
        {
            if (SuppressNotice || Mode == OverfitLicenseMode.Commercial || _noticeShown)
            {
                return;
            }

            ShowOpenSourceNoticeOnce();
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ShowOpenSourceNoticeOnce()
        {
            lock (_lock)
            {
                if (SuppressNotice || _noticeShown)
                {
                    return;
                }

                _noticeShown = true;

                var msg =
                    "==========================================================================\n" +
                    "[OVERFIT ML ENGINE] Running in Open-Source mode.\n" +
                    "This build is available under AGPLv3.\n" +
                    "If you use Overfit inside a closed-source commercial product,\n" +
                    "please obtain a commercial license.\n" +
                    "Project page: https://github.com/DevOnBike/Overfit\n" +
                    "==========================================================================";

                WriteMessage(msg, ConsoleColor.DarkYellow);
            }
        }

        public static void UseCommercialMode()
        {
            Mode = OverfitLicenseMode.Commercial;
        }

        private static void WriteMessage(string message, ConsoleColor color)
        {
            if (MessageSink is not null)
            {
                try
                {
                    MessageSink(message);
                    return;
                }
                catch
                {
                    // fallback do Console/Debug
                }
            }

            var originalColor = Console.ForegroundColor;

            try
            {
                Console.ForegroundColor = color;
                Console.WriteLine(message);
            }
            finally
            {
                Console.ForegroundColor = originalColor;
            }

            Debug.WriteLine(message);
        }
    }
}