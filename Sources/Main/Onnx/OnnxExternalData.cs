// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Exceptions;
using DevOnBike.Overfit.Onnx.Schema;
using DevOnBike.Overfit.Runtime;

namespace DevOnBike.Overfit.Onnx
{
    /// <summary>
    /// Resolving an initializer that lives in a separate <c>.data</c> file — the PyTorch ≥ 2.x default.
    ///
    /// <para><b>One implementation, and that is the whole point of the file.</b> There were two: the linear
    /// importer's, which rejects an empty location, rejects an absolute one and proves the result stays
    /// inside the model directory, and the graph importer's, which did none of the three. The second opened
    /// with the comment <i>"To avoid duplication we just re-implement the minimal version here"</i>, and
    /// "the minimal version" is exactly where the three checks went. The importer without the checks is the
    /// one required for skip-connection models — ResNet, DenseNet — so the guarded path was the less-used
    /// one.</para>
    ///
    /// <para><b>Why an unvalidated path is not a small thing.</b>
    /// <see cref="Path.Combine(string, string)"/> returns its second argument whole when that argument is
    /// rooted; this is documented behaviour, not a quirk. A model naming <c>/etc/shadow</c> or
    /// <c>C:\Users\...\id_rsa</c> as its external data location is therefore not combined with the model
    /// directory at all — the process simply reads that file and hands its bytes back as weights. A relative
    /// <c>../../..</c> escapes just as easily when nothing checks containment.</para>
    ///
    /// <para><b>Only the requested slice is read.</b> The previous code read whole external data files into
    /// memory and cached them. For a multi-gigabyte <c>.data</c> file that is gigabytes of peak RAM to
    /// extract a few megabytes of one tensor, and with an unvalidated path it was an out-of-memory kill that
    /// a model file could ask for. Seeking to the offset costs one open handle per file and reads exactly
    /// what the initializer claims.</para>
    /// </summary>
    internal static class OnnxExternalData
    {
        /// <summary>
        /// Replaces every externally-stored initializer in <paramref name="model"/> with its bytes.
        /// </summary>
        /// <param name="model">Parsed model; its initializer list is rewritten in place.</param>
        /// <param name="externalDataDir">
        /// Directory the model was loaded from. Every location is resolved relative to it and must stay
        /// inside it. Null or empty is an error <b>only if</b> something actually references external data,
        /// so a model with none loads from a stream as before.
        /// </param>
        public static void Resolve(OnnxModel model, string? externalDataDir)
        {
            ArgumentNullException.ThrowIfNull(model);

            var handles = new Dictionary<string, FileStream>(StringComparer.OrdinalIgnoreCase);

            try
            {
                for (var i = 0; i < model.Graph.Initializers.Count; i++)
                {
                    var initializer = model.Graph.Initializers[i];

                    if (initializer.ExternalData is null)
                    {
                        continue;
                    }

                    model.Graph.Initializers[i] = ResolveOne(initializer, externalDataDir, handles);
                }
            }
            finally
            {
                foreach (var handle in handles.Values)
                {
                    handle.Dispose();
                }
            }
        }

        private static OnnxTensor ResolveOne(
            OnnxTensor initializer,
            string? externalDataDir,
            Dictionary<string, FileStream> handles)
        {
            var external = initializer.ExternalData!;

            if (string.IsNullOrEmpty(externalDataDir))
            {
                throw new OverfitRuntimeException(
                    $"Initializer '{initializer.Name}' references external data '{external.Location}', but "
                    + "no external data directory was provided. Use the Load(path) overload, which resolves "
                    + "it automatically.");
            }

            var fullPath = ResolvePath(externalDataDir, external.Location, initializer.Name);

            if (!handles.TryGetValue(fullPath, out var file))
            {
                if (!File.Exists(fullPath))
                {
                    throw new FileNotFoundException(
                        $"External data file not found: {fullPath} (referenced by initializer "
                        + $"'{initializer.Name}').",
                        fullPath);
                }

                file = new FileStream(
                    fullPath, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: 0,
                    FileOptions.RandomAccess);

                handles[fullPath] = file;
            }

            var offset = CheckedToInt64(
                external.Offset, $"External data offset for initializer '{initializer.Name}'");

            // The offset is proved to be inside the file BEFORE anything is computed from it. The
            // subtraction below expresses the "zero means read to the end" convention, and it is where a
            // negative allocation used to be reachable, because nothing had established that the offset was
            // inside the file. A comment here previously claimed this order and the code had the other one.
            if (offset < 0 || offset > file.Length)
            {
                throw new OverfitFormatException(
                    $"External data for '{initializer.Name}' starts at byte {offset}, but "
                    + $"'{Path.GetFileName(fullPath)}' is {file.Length} bytes.");
            }

            var available = file.Length - offset;

            var length = external.Length == 0
                ? available
                : CheckedToInt64(
                    external.Length, $"External data length for initializer '{initializer.Name}'");

            if (length < 0 || length > available)
            {
                throw new OverfitFormatException(
                    $"External data for '{initializer.Name}' requests bytes [{offset}, {offset + length}) "
                    + $"but '{Path.GetFileName(fullPath)}' is only {file.Length} bytes.");
            }

            if (length > Array.MaxLength)
            {
                throw new OverfitFormatException(
                    $"External data for '{initializer.Name}' is {length} bytes, which cannot be held in a "
                    + $"single array (the limit is {Array.MaxLength}).");
            }

            var raw = new byte[length];

            file.Seek(offset, SeekOrigin.Begin);
            file.ReadExactly(raw);

            return new OnnxTensor
            {
                Name = initializer.Name,
                DataType = initializer.DataType,
                Dims = initializer.Dims,
                RawData = raw,
                FloatData = initializer.FloatData,
                Int64Data = initializer.Int64Data,
                ExternalData = null,
            };
        }

        /// <summary>
        /// The model directory plus a location the model supplied, proved to still be inside it.
        ///
        /// <para>All three checks matter and each closes a different door: an empty location resolves to the
        /// directory itself, a rooted one bypasses <see cref="Path.Combine(string, string)"/> entirely, and
        /// a relative one with enough <c>..</c> segments walks out. Containment is tested after
        /// <see cref="Path.GetFullPath(string)"/> has normalised the segments away, because testing the
        /// unnormalised string is the version of this check that does not work.</para>
        /// </summary>
        internal static string ResolvePath(string externalDataDir, string location, string initializerName)
        {
            // The checks themselves now live in ContainedPath, because this was the only correct copy of
            // them in the repository and a second loader joining a model-supplied name to a directory had
            // none at all. Same three checks, same order, same messages in substance.
            return ContainedPath.Resolve(
                externalDataDir, location, $"External data for initializer '{initializerName}'");
        }

        /// <summary>
        /// Rejects a file-supplied count that cannot be a byte position, naming what carried it.
        ///
        /// <para>Kept at 64 bits rather than narrowing to <see cref="int"/> here. The previous code cast to
        /// <c>int</c> before any bounds check, so a value above <see cref="int.MaxValue"/> wrapped to a
        /// negative one and surfaced as <see cref="ArgumentOutOfRangeException"/> — not the
        /// <see cref="OverfitFormatException"/> every other malformed-model path produces, so a caller that
        /// handles bad models did not handle this one.</para>
        /// </summary>
        private static long CheckedToInt64(long value, string name)
        {
            if (value < 0)
            {
                throw new OverfitFormatException($"{name} is negative ({value}).");
            }

            return value;
        }
    }
}
