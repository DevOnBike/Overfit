// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Exceptions;

namespace DevOnBike.Overfit.Runtime
{
    /// <summary>
    /// A model directory plus a path the model itself supplied, proved to still be inside it.
    ///
    /// <para><b>Every format this engine loads has a file that names other files.</b> ONNX has external-data
    /// locations, sharded safetensors has <c>weight_map</c>, GGUF has sidecars. In each case the name is read
    /// out of a file the customer downloaded from a public hub, and joining it to a directory without a
    /// containment test lets that file choose which file the process opens.</para>
    ///
    /// <para><b>All three checks matter and each closes a different door.</b> An empty location resolves to
    /// the directory itself; a <b>rooted</b> one bypasses <see cref="Path.Combine(string, string)"/> entirely,
    /// which is the one people miss because the combine <i>looks</i> like it constrains the result; and a
    /// relative one with enough <c>..</c> segments walks out. Containment is tested <b>after</b>
    /// <see cref="Path.GetFullPath(string)"/> has normalised the segments away — testing the unnormalised
    /// string is the version of this check that does not work.</para>
    ///
    /// <para><b>Rooted covers more than a local absolute path.</b> On Windows a UNC name is rooted, so
    /// rejecting rooted paths is also what stops a model file pointing the loader at a remote share and
    /// making the host authenticate to it.</para>
    ///
    /// <para>Extracted from <c>OnnxExternalData.ResolvePath</c>, which had the only correct copy of this
    /// check in the repository while a second loader had none. A guard that exists in one loader and not the
    /// next is the shape this type exists to prevent.</para>
    /// </summary>
    internal static class ContainedPath
    {
        /// <summary>
        /// Resolves <paramref name="location"/> against <paramref name="baseDirectory"/>, or throws.
        /// </summary>
        /// <param name="baseDirectory">Directory the result must stay inside.</param>
        /// <param name="location">Path as the model file supplied it.</param>
        /// <param name="describeSubject">
        /// What carried the location, for the exception — e.g. <c>"initializer 'weight'"</c> or
        /// <c>"weight_map entry 'model.embed'"</c>. The message is the only thing an operator staring at a
        /// refused model has to go on.
        /// </param>
        /// <returns>The absolute, normalised path.</returns>
        internal static string Resolve(string baseDirectory, string location, string describeSubject)
        {
            ArgumentNullException.ThrowIfNull(baseDirectory);
            ArgumentNullException.ThrowIfNull(describeSubject);

            if (string.IsNullOrWhiteSpace(location))
            {
                throw new OverfitFormatException($"{describeSubject} has an empty path.");
            }

            if (Path.IsPathRooted(location))
            {
                throw new OverfitFormatException(
                    $"{describeSubject} references an absolute path: '{location}'.");
            }

            var root = Path.GetFullPath(baseDirectory);
            var full = Path.GetFullPath(Path.Combine(root, location));

            var comparison = OperatingSystem.IsWindows()
                ? StringComparison.OrdinalIgnoreCase
                : StringComparison.Ordinal;

            if (!IsInside(full, root, comparison))
            {
                throw new OverfitFormatException(
                    $"{describeSubject} escapes the model directory: '{location}'.");
            }

            return full;
        }

        private static bool IsInside(string path, string directory, StringComparison comparison)
        {
            var normalized = directory;

            if (!normalized.EndsWith(Path.DirectorySeparatorChar))
            {
                normalized += Path.DirectorySeparatorChar;
            }

            return path.StartsWith(normalized, comparison);
        }
    }
}
