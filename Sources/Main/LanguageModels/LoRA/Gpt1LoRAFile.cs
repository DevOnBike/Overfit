// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.LanguageModels.LoRA
{
    /// <summary>
    /// One LoRA adapter entry: which weight matrix it adapts (module + layer) and
    /// the trained low-rank factors. <see cref="Layer"/> is -1 for the LM head.
    /// </summary>
    internal readonly record struct Gpt1LoRAEntry(int Layer, LoRATargetModules Module, LoRAWeight Weight);

    /// <summary>
    /// Binary .bin format for a GPT1 LoRA adapter — the contract shared between
    /// <see cref="Gpt1LoRAFineTuner"/> (writer) and <see cref="Gpt1LoRAMergeAdapter"/>
    /// (reader).
    ///
    /// Layout (matches <see cref="LlamaLoRAAdapter"/>): magic "LORA", version,
    /// entry count, then per entry — layer index, module, head index (always 0 for
    /// GPT1), <see cref="LoRAWeight"/> blob. A Stage-1 LM-head adapter is simply a
    /// one-entry file; a Stage-2 FFN adapter carries FeedForwardUp/Down entries for
    /// every block.
    /// </summary>
    internal static class Gpt1LoRAFile
    {
        private const uint Magic = 0x524F4C4Fu;   // "LORA"
        private const int Version = 1;

        public static void Save(string path, IReadOnlyList<Gpt1LoRAEntry> entries)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);
            ArgumentNullException.ThrowIfNull(entries);

            if (entries.Count == 0)
            {
                throw new ArgumentException("A LoRA adapter must have at least one entry.", nameof(entries));
            }

            using var fs = File.Create(path);
            using var bw = new BinaryWriter(fs);

            bw.Write(Magic);
            bw.Write(Version);
            bw.Write(entries.Count);

            foreach (var entry in entries)
            {
                bw.Write(entry.Layer);
                bw.Write((int)entry.Module);
                bw.Write(0);                       // head index — unused for GPT1
                entry.Weight.Save(bw);
            }
        }

        public static IReadOnlyList<Gpt1LoRAEntry> Load(string path)
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
            if (count <= 0)
            {
                throw new InvalidDataException($"LoRA file declares {count} entries.");
            }

            var entries = new Gpt1LoRAEntry[count];
            for (var i = 0; i < count; i++)
            {
                var layer = br.ReadInt32();
                var module = (LoRATargetModules)br.ReadInt32();
                _ = br.ReadInt32();                // head index — unused for GPT1
                entries[i] = new Gpt1LoRAEntry(layer, module, LoRAWeight.Load(br));
            }

            return entries;
        }
    }
}
