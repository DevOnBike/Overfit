// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.OverfitChat
{
    /// <summary>
    /// Tiny on-device error log: appends timestamped lines to a file in the app's files dir (and mirrors to
    /// Android logcat under the "OverThink" tag). The "Report a problem" action shares this file's tail so a
    /// user can send a crash/error report without needing adb. Best-effort and self-bounding.
    /// </summary>
    internal static class AppLog
    {
        private const long MaxBytes = 256 * 1024;
        private static readonly object Gate = new();
        private static string? _path;

        public static string? Path => _path;

        public static void Init(string filesDir)
        {
            _path = System.IO.Path.Combine(filesDir, "overthink.log");
            Write($"--- session start ({System.Runtime.InteropServices.RuntimeInformation.OSArchitecture}) ---");
        }

        public static void Write(string message, Exception? error = null)
        {
            Android.Util.Log.Info("OverThink", error is null ? message : message + ": " + error);

            if (_path is null)
            {
                return;
            }

            try
            {
                lock (Gate)
                {
                    var line = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss}  {message}";
                    if (error is not null)
                    {
                        line += Environment.NewLine + error;
                    }
                    System.IO.File.AppendAllText(_path, line + Environment.NewLine);

                    if (new System.IO.FileInfo(_path).Length > MaxBytes)
                    {
                        // Keep the most recent half so the file never grows unbounded.
                        var text = System.IO.File.ReadAllText(_path);
                        System.IO.File.WriteAllText(_path, text[(text.Length / 2)..]);
                    }
                }
            }
            catch
            {
                // logging must never throw
            }
        }

        public static string ReadTail(int maxChars = 8000)
        {
            try
            {
                if (_path is not null && System.IO.File.Exists(_path))
                {
                    var text = System.IO.File.ReadAllText(_path);
                    return text.Length > maxChars ? text[^maxChars..] : text;
                }
            }
            catch
            {
                // ignore
            }

            return "(no log)";
        }
    }
}
