// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    /// <summary>
    /// Binary .bin format for a GPT1 LM-head LoRA adapter — the contract shared
    /// between <see cref="Gpt1LoRAFineTuner"/> (writer) and
    /// <see cref="Gpt1LoRAMergeAdapter"/> (reader).
    ///
    /// Layout (matches <see cref="LlamaLoRAAdapter"/> so a future multi-module
    /// reader can consume it): magic "LORA", version, entry count, then per
    /// entry — layer index, module, head index, <see cref="LoRAWeight"/> blob.
    /// Stage 1 writes exactly one entry keyed
    /// <see cref="LoRATargetModules.LanguageModelHead"/>.
    /// </summary>
    internal static class Gpt1LoRAFile
    {
        private const uint Magic = 0x524F4C4Fu;   // "LORA"
        private const int Version = 1;

        public static void SaveLMHead(string path, LoRAWeight weight)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);
            ArgumentNullException.ThrowIfNull(weight);

            using var fs = File.Create(path);
            using var bw = new BinaryWriter(fs);

            bw.Write(Magic);
            bw.Write(Version);
            bw.Write(1);                                        // entry count
            bw.Write(0);                                        // layer index
            bw.Write((int)LoRATargetModules.LanguageModelHead); // module
            bw.Write(0);                                        // head index
            weight.Save(bw);
        }

        public static LoRAWeight LoadLMHead(string path)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);

            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);

            var magic = br.ReadUInt32();
            if (magic != Magic)
            {
                throw new InvalidDataException($"Not a LoRA file (magic=0x{magic:X8}).");
            }

            var version = br.ReadInt32();
            if (version != Version)
            {
                throw new InvalidDataException($"Unsupported LoRA file version {version}.");
            }

            var count = br.ReadInt32();
            if (count != 1)
            {
                throw new InvalidDataException($"Expected 1 LoRA entry for the LM head, got {count}.");
            }

            _ = br.ReadInt32();                              // layer index
            var module = (LoRATargetModules)br.ReadInt32();
            _ = br.ReadInt32();                              // head index

            if (module != LoRATargetModules.LanguageModelHead)
            {
                throw new InvalidDataException($"Expected a LanguageModelHead entry, got {module}.");
            }

            return LoRAWeight.Load(br);
        }
    }
}
