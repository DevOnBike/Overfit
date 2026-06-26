// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using Android.App;
using Android.Content.PM;
using Android.OS;
using Android.Views;
using Android.Widget;
using DevOnBike.Overfit.AndroidBench;

namespace DevOnBike.OverfitBench
{
    /// <summary>
    /// Single-screen bench launcher: a RUN button kicks <see cref="DecodeBench.Run"/> on a background
    /// thread and streams its log lines to a scrollable TextView (and to logcat tag "OverfitBench").
    /// The model is read from the app's external files dir — push it there with
    /// <c>adb push model.gguf /sdcard/Android/data/com.devonbike.overfitbench/files/model.gguf</c>.
    /// </summary>
    [Activity(
        Label = "Overfit Bench",
        MainLauncher = true,
        ConfigurationChanges = ConfigChanges.Orientation | ConfigChanges.ScreenSize
            | ConfigChanges.KeyboardHidden | ConfigChanges.UiMode)]
    public class MainActivity : Activity
    {
        private const string ModelFileName = "model.gguf";
        private const string LogTag = "OverfitBench";

        // The bench flips the global Q4KDotKernel.ForceScalar, so two concurrent runs corrupt each other.
        // Guard so it runs exactly once per process even if the Activity is recreated.
        private static bool _benchStarted;

        protected override void OnCreate(Bundle? savedInstanceState)
        {
            base.OnCreate(savedInstanceState);

            // Keep the screen on so a screen-off doesn't recreate the Activity mid-run.
            Window?.AddFlags(WindowManagerFlags.KeepScreenOn);

            var output = new TextView(this)
            {
                Text = "Overfit decode bench\nPush a Q4_K_M GGUF to the app files dir, then tap RUN.\n",
            };
            var runButton = new Button(this) { Text = "RUN BENCH" };

            var layout = new LinearLayout(this) { Orientation = Orientation.Vertical };
            layout.AddView(runButton);
            layout.AddView(output);

            var scroll = new ScrollView(this);
            scroll.AddView(layout);
            SetContentView(scroll);

            void Log(string line)
            {
                Android.Util.Log.Info(LogTag, line);
                RunOnUiThread(() => output.Text += line + "\n");
            }

            void StartBench()
            {
                runButton.Enabled = false;
                new System.Threading.Thread(() =>
                {
                    try
                    {
                        var filesDir = GetExternalFilesDir(null)?.AbsolutePath
                            ?? throw new System.IO.DirectoryNotFoundException("No external files dir.");
                        var modelPath = System.IO.Path.Combine(filesDir, ModelFileName);
                        Log($"model: {modelPath}");

                        if (!System.IO.File.Exists(modelPath))
                        {
                            Log($"!! model not found. adb push <model>.gguf {modelPath}");
                            return;
                        }

                        DecodeBench.Run(modelPath, genTokens: 32, repeats: 1, warmup: 2, log: Log);
                    }
                    catch (System.Exception ex)
                    {
                        Log($"ERROR: {ex.GetType().Name}: {ex.Message}");
                    }
                    finally
                    {
                        RunOnUiThread(() => runButton.Enabled = true);
                    }
                }).Start();
            }

            runButton.Click += (_, _) => StartBench();

            // Auto-run once per process on launch — the button sits under the status bar on edge-to-edge
            // displays, so a headless `adb shell monkey ... LAUNCHER` is enough to kick the bench. The guard
            // stops an Activity recreate from starting a second, interfering run (results go to logcat).
            if (!_benchStarted)
            {
                _benchStarted = true;
                StartBench();
            }
        }
    }
}
