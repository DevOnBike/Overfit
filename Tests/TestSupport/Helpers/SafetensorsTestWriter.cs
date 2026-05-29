// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Buffers.Binary;
using System.Collections.Generic;
using System.Text;

namespace DevOnBike.Overfit.Tests.TestSupport.Helpers
{
    /// <summary>
    /// Minimal test-only writer for the safetensors format (F32 tensors only), used to fabricate
    /// HuggingFace-layout fixtures so loaders can be exercised offline without a real model file.
    /// This is deliberately a test helper, not a product exporter (Overfit loading is one-directional).
    /// </summary>
    public sealed class SafetensorsTestWriter
    {
        private readonly List<(string Name, long[] Shape, float[] Data)> _tensors = new();

        public SafetensorsTestWriter Add(string name, long[] shape, float[] data)
        {
            _tensors.Add((name, shape, data));
            return this;
        }

        public void Write(string path)
        {
            // 1. Build the JSON header with contiguous data offsets in insertion order.
            var sb = new StringBuilder();
            sb.Append('{');
            long offset = 0;
            for (var i = 0; i < _tensors.Count; i++)
            {
                var (name, shape, data) = _tensors[i];
                if (i > 0) { sb.Append(','); }
                sb.Append('"').Append(name).Append("\":{\"dtype\":\"F32\",\"shape\":[");
                for (var s = 0; s < shape.Length; s++)
                {
                    if (s > 0) { sb.Append(','); }
                    sb.Append(shape[s]);
                }

                var begin = offset;
                var end = offset + (long)data.Length * 4;
                sb.Append("],\"data_offsets\":[").Append(begin).Append(',').Append(end).Append("]}");
                offset = end;
            }

            sb.Append('}');
            var headerBytes = Encoding.UTF8.GetBytes(sb.ToString());

            using var fs = File.Create(path);
            Span<byte> len = stackalloc byte[8];
            BinaryPrimitives.WriteUInt64LittleEndian(len, (ulong)headerBytes.Length);
            fs.Write(len);
            fs.Write(headerBytes);

            // 2. Raw little-endian F32 payload, same order as the header.
            Span<byte> four = stackalloc byte[4];
            foreach (var (_, _, data) in _tensors)
            {
                foreach (var v in data)
                {
                    BinaryPrimitives.WriteSingleLittleEndian(four, v);
                    fs.Write(four);
                }
            }
        }
    }
}
