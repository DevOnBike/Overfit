// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

// Overfit as a drop-in local model for MICROSOFT AGENT FRAMEWORK.
//
// Agent Framework (Microsoft.Agents.AI) is the successor that merged Semantic Kernel and AutoGen. It is built
// on Microsoft.Extensions.AI's IChatClient — and Overfit implements IChatClient — so a ChatClientAgent runs on
// a local GGUF with NO adapter code, no Ollama, no Docker, no cloud key. This file is that proof, end to end.
//
//   dotnet run --project Demo/AgentFrameworkDemo -- <path-to-model.gguf>
//
// Verified against Microsoft.Agents.AI 1.13.0 + DevOnBike.Overfit 10.0.x on a Qwen2.5-0.5B Q4_K_M.

using DevOnBike.Overfit.Extensions.AI;
using DevOnBike.Overfit.LanguageModels;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;

namespace DevOnBike.Overfit.Demo.AgentFramework
{
    public static class Program
    {
        public static async Task<int> Main(string[] args)
        {
            var modelPath = args.Length > 0
                ? args[0]
                : Environment.GetEnvironmentVariable("OVERFIT_MODEL_PATH");

            if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
            {
                Console.Error.WriteLine(
                    "Usage: dotnet run --project Demo/AgentFrameworkDemo -- <path-to-model.gguf>\n" +
                    "   or: set OVERFIT_MODEL_PATH to a .gguf file.\n\n" +
                    "Any small instruct GGUF works, e.g. Qwen2.5-0.5B-Instruct-Q4_K_M.");
                return 1;
            }

            Console.WriteLine($"Loading {Path.GetFileName(modelPath)} in-process (CPU, mmap) ...");
            using var overfit = OverfitClient.LoadGguf(modelPath, mmap: true, maxNewTokens: 128);

            // The entire integration. Overfit's IChatClient IS an Agent Framework model provider.
            IChatClient chat = overfit.AsChatClient();
            AIAgent agent = new ChatClientAgent(chat, instructions: "You are a concise assistant. Answer briefly.");

            foreach (var question in new[]
            {
                "What is the capital of France?",
                "Name one benefit of running a language model locally.",
            })
            {
                Console.WriteLine($"\n> {question}");
                var response = await agent.RunAsync(question);
                Console.WriteLine(response);
            }

            Console.WriteLine("\nThat agent ran entirely on your CPU — no server, no cloud, no data egress.");
            return 0;
        }
    }
}
