// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Diagnostics;
using System.Globalization;
using DevOnBike.Overfit.Trees;

namespace DevOnBike.Overfit.Tests.Trees
{
    /// <summary>
    /// End-to-end check of the <c>overfit score</c> CLI: runs the built executable over a CSV and asserts
    /// its predictions match the in-process <see cref="BoostedTreeModel"/> (the same predictor the parity
    /// tests pin against XGBoost). Integration — needs the built <c>overfit</c> exe, so it is a [LongFact]
    /// and skips cleanly when the exe is absent (e.g. a Main-only build).
    /// </summary>
    public sealed class XgboostScoreCliTests
    {
        [LongFact]
        public void ScoreCommand_MatchesInProcessPredictor()
        {
            var exe = LocateOverfitExe();
            if (exe is null)
            {
                return; // exe not built — nothing to exercise
            }

            var modelPath = Path.Combine(AppContext.BaseDirectory, "test_fixtures", "xgboost", "clf_model.json");
            var model = XgboostModelLoader.Load(modelPath);

            // Four deterministic rows (one with a missing value) → expected probabilities from the library.
            var rows = new[]
            {
                new[] { 0.5f, -0.3f, 1.2f, 0.1f, -0.7f, 0.4f, 0.0f, 0.9f, -1.1f, 0.2f, 0.6f, -0.5f },
                new[] { -1.0f, 0.8f, -0.2f, 1.5f, 0.3f, -0.9f, 0.7f, -0.4f, 0.1f, 1.0f, -0.6f, 0.5f },
                new[] { float.NaN, 0.2f, 0.4f, -0.8f, 1.1f, 0.0f, -0.3f, 0.6f, 0.9f, -1.2f, 0.5f, 0.1f },
                new[] { 0.9f, -0.9f, 0.9f, -0.9f, 0.9f, -0.9f, 0.9f, -0.9f, 0.9f, -0.9f, 0.9f, -0.9f }
            };

            var expected = new float[rows.Length];
            for (var r = 0; r < rows.Length; r++)
            {
                expected[r] = model.Predict(rows[r]);
            }

            var csvPath = Path.Combine(Path.GetTempPath(), $"overfit_score_{Guid.NewGuid():N}.csv");
            try
            {
                WriteCsv(csvPath, rows);

                var stdout = RunScore(exe, modelPath, csvPath);
                var lines = stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries);

                Assert.Equal("prediction", lines[0].Trim());
                Assert.Equal(rows.Length, lines.Length - 1);

                for (var r = 0; r < rows.Length; r++)
                {
                    var got = float.Parse(lines[r + 1].Trim(), CultureInfo.InvariantCulture);
                    Assert.Equal(expected[r], got, 5);
                }
            }
            finally
            {
                File.Delete(csvPath);
            }
        }

        private static void WriteCsv(string path, float[][] rows)
        {
            using var writer = new StreamWriter(path);
            foreach (var row in rows)
            {
                for (var c = 0; c < row.Length; c++)
                {
                    if (c > 0)
                    {
                        writer.Write(',');
                    }
                    // Missing value ⇒ empty cell, which the CLI maps to NaN.
                    writer.Write(float.IsNaN(row[c]) ? string.Empty : row[c].ToString(CultureInfo.InvariantCulture));
                }
                writer.Write('\n');
            }
        }

        private static string RunScore(string exe, string modelPath, string csvPath)
        {
            var psi = new ProcessStartInfo(exe)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };
            psi.ArgumentList.Add("score");
            psi.ArgumentList.Add(modelPath);
            psi.ArgumentList.Add("--input");
            psi.ArgumentList.Add(csvPath);

            using var process = Process.Start(psi)!;
            var stdout = process.StandardOutput.ReadToEnd();
            process.WaitForExit(30_000);

            Assert.Equal(0, process.ExitCode);
            return stdout;
        }

        private static string? LocateOverfitExe()
        {
            // Walk up to the repo root (the folder with Overfit.sln), then look under Sources/Cli/bin.
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "Overfit.sln")))
            {
                dir = dir.Parent;
            }

            if (dir is null)
            {
                return null;
            }

            var cliBin = Path.Combine(dir.FullName, "Sources", "Cli", "bin");
            if (!Directory.Exists(cliBin))
            {
                return null;
            }

            var name = OperatingSystem.IsWindows() ? "overfit.exe" : "overfit";
            var matches = Directory.GetFiles(cliBin, name, SearchOption.AllDirectories);
            return matches.Length > 0 ? matches[0] : null;
        }
    }
}
