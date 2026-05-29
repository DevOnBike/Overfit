// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Demo.LocalAgent.Infrastructure
{
    internal static class ModelPathResolver
    {
        // Returns either a *.gguf file path or a directory containing model.safetensors (HuggingFace).
        public static string Resolve(IConfiguration config)
        {
            // 1) appsettings `ModelPath` — a *.gguf file OR a directory with model.safetensors.
            var fromSettings = config.GetValue<string>("ModelPath");
            if (!string.IsNullOrWhiteSpace(fromSettings))
            {
                if (File.Exists(fromSettings) && fromSettings.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
                {
                    return fromSettings;
                }
                if (Directory.Exists(fromSettings) && File.Exists(Path.Combine(fromSettings, "model.safetensors")))
                {
                    return fromSettings;
                }
            }

            // 2) Env var `OVERFIT_MODEL_DIR` — a directory; prefer a *.gguf inside, else model.safetensors.
            var fromEnv = Environment.GetEnvironmentVariable("OVERFIT_MODEL_DIR");
            if (!string.IsNullOrWhiteSpace(fromEnv) && Directory.Exists(fromEnv))
            {
                var ggufs = Directory.GetFiles(fromEnv, "*.gguf");
                if (ggufs.Length > 0) { return ggufs[0]; }
                if (File.Exists(Path.Combine(fromEnv, "model.safetensors"))) { return fromEnv; }
            }

            throw new InvalidOperationException(
            "Could not locate a model. Set 'ModelPath' in appsettings to either an absolute *.gguf file " +
            "(e.g. C:\\qwen3b\\qwen.q4km.gguf) OR a HuggingFace directory containing model.safetensors " +
            "(e.g. C:\\qwen3b for an unpacked Qwen2.5-0.5B). Or set OVERFIT_MODEL_DIR to such a directory. " +
            "See Demo/LocalAgentAspNetDemo/README.md.");
        }
    }
}