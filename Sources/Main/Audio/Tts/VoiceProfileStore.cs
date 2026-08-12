// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.Audio.Tts
{
    /// <summary>
    /// Persists a <see cref="VoiceProfile"/> to a directory as a small text manifest (<c>&lt;id&gt;.voice</c>) plus a
    /// raw little-endian float32 blob (<c>&lt;id&gt;.embedding.bin</c>) when a speaker embedding is present. Manual
    /// format on purpose — no reflection-based JSON, so it stays Native-AOT-clean. This is the on-disk side of
    /// <c>overfit voice enroll</c>; the embedding is filled in once a cloning backend exists.
    /// </summary>
    // OVERFIT040 for this whole type, type-scoped because every method on it is disk I/O — Save, Load and
    // the private ReadId are the three flagged, and there is nothing else here to protect.
    //
    // THE CONSTRAINT: this is the on-disk side of `overfit voice enroll` — a manifest of five short lines and
    // a small float blob, read or written once when a voice is enrolled, listed or loaded. It runs on the
    // caller's own thread (a CLI verb's main thread, or a model-construction path), not on a pool thread, and
    // nothing is queued behind it.
    //
    // WHAT IS GIVEN UP: these are public API of the shipped `DevOnBike.Overfit` package. Returning tasks
    // would break every caller and force the CLI verbs above them async too, for an operation whose cost is
    // one small file.
#pragma warning disable OVERFIT040
    public static class VoiceProfileStore
    {
        private const string ManifestHeader = "overfit-voice 1";

        public static void Save(VoiceProfile profile, string directory)
        {
            ArgumentNullException.ThrowIfNull(profile);
            ArgumentException.ThrowIfNullOrWhiteSpace(directory);
            Directory.CreateDirectory(directory);

            var baseName = Sanitize(profile.Id);
            var embedding = profile.SpeakerEmbedding;
            var embeddingDim = embedding?.Length ?? 0;

            var sb = new StringBuilder();
            sb.Append(ManifestHeader).Append('\n');
            sb.Append("id=").Append(profile.Id).Append('\n');
            sb.Append("language=").Append(profile.Language).Append('\n');
            sb.Append("referenceAudio=").Append(profile.ReferenceAudioPath ?? string.Empty).Append('\n');
            sb.Append("embeddingDim=").Append(embeddingDim).Append('\n');
            File.WriteAllText(Path.Combine(directory, baseName + ".voice"), sb.ToString());

            var binPath = Path.Combine(directory, baseName + ".embedding.bin");
            if (embeddingDim > 0)
            {
                using var bw = new BinaryWriter(File.Create(binPath));
                for (var k = 0; k < embeddingDim; k++)
                {
                    bw.Write(embedding![k]);
                }
            }
            if (embeddingDim <= 0 && File.Exists(binPath))
            {
                // Gated on embeddingDim: the branch above CREATES binPath, so an ungated File.Exists
                // here would delete the file that was just written.
                File.Delete(binPath);   // a preset voice has no embedding — drop a stale one
            }
        }

        public static VoiceProfile Load(string id, string directory)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(id);
            var baseName = Sanitize(id);
            var path = Path.Combine(directory, baseName + ".voice");
            if (!File.Exists(path))
            {
                throw new OverfitFormatException($"Voice '{id}' not found in '{directory}'.");
            }

            string realId = id, language = "en", reference = string.Empty;
            var embeddingDim = 0;
            foreach (var line in File.ReadAllLines(path))
            {
                var eq = line.IndexOf('=');
                if (eq <= 0)
                {
                    continue;
                }
                var key = line.Substring(0, eq);
                var value = line.Substring(eq + 1);
                switch (key)
                {
                    case "id":
                        realId = value;
                        break;
                    case "language":
                        language = value;
                        break;
                    case "referenceAudio":
                        reference = value;
                        break;
                    case "embeddingDim":
                        _ = int.TryParse(value, out embeddingDim);
                        break;
                }
            }

            float[]? embedding = null;
            var binPath = Path.Combine(directory, baseName + ".embedding.bin");
            if (embeddingDim > 0 && File.Exists(binPath))
            {
                // OVERFIT001: load-time — reads a persisted voice-profile embedding from disk once; the array
                // is stored on the returned VoiceProfile, not per-call scratch.
#pragma warning disable OVERFIT001
                embedding = new float[embeddingDim];
#pragma warning restore OVERFIT001
                using var br = new BinaryReader(File.OpenRead(binPath));
                for (var k = 0; k < embeddingDim; k++)
                {
                    embedding[k] = br.ReadSingle();
                }
            }

            return new VoiceProfile(realId, language, embedding, string.IsNullOrEmpty(reference) ? null : reference);
        }

        /// <summary>The ids of every voice manifest in <paramref name="directory"/> (read from each manifest).</summary>
        public static IReadOnlyList<string> List(string directory)
        {
            var ids = new List<string>();
            if (!Directory.Exists(directory))
            {
                return ids;
            }
            foreach (var file in Directory.GetFiles(directory, "*.voice"))
            {
                ids.Add(ReadId(file));
            }
            return ids;
        }

        public static bool Exists(string id, string directory)
            => File.Exists(Path.Combine(directory, Sanitize(id) + ".voice"));

        private static string ReadId(string manifestPath)
        {
            foreach (var line in File.ReadAllLines(manifestPath))
            {
                if (line.StartsWith("id=", StringComparison.Ordinal))
                {
                    return line.Substring(3);
                }
            }
            return Path.GetFileNameWithoutExtension(manifestPath);
        }

        private static string Sanitize(string id)
        {
            var chars = id.ToCharArray();
            for (var k = 0; k < chars.Length; k++)
            {
                var c = chars[k];
                if (!(char.IsLetterOrDigit(c) || c is '.' or '_' or '-'))
                {
                    chars[k] = '_';
                }
            }
            return new string(chars);
        }
    }
#pragma warning restore OVERFIT040
}
