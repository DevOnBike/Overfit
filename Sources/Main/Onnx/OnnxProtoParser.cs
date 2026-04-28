// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Onnx.Protobuf;
using DevOnBike.Overfit.Onnx.Schema;

namespace DevOnBike.Overfit.Onnx
{
    /// <summary>
    /// Parses ONNX protobuf into in-memory <see cref="OnnxModel"/>.
    /// 
    /// Field numbers reference the official onnx.proto3 schema.
    /// We deliberately ignore fields not needed for inference import (e.g., training_info,
    /// metadata_props, doc_string) by skipping them via SkipField.
    /// </summary>
    internal static class OnnxProtoParser
    {
        public static OnnxModel ParseModel(ReadOnlySpan<byte> data)
        {
            // ModelProto fields:
            //   1: ir_version (int64)
            //   2: producer_name (string)
            //   3: producer_version (string)
            //   4: domain (string) - skip
            //   5: model_version (int64) - skip
            //   6: doc_string (string) - skip
            //   7: graph (GraphProto)
            //   8: opset_import (OperatorSetIdProto, repeated)
            //   14: metadata_props - skip

            long irVersion = 0;
            var producerName = "";
            var producerVersion = "";
            var opsetImports = new List<OnnxOpsetImport>();
            OnnxGraph? graph = null;

            var reader = new ProtoReader(data);

            while (!reader.IsEnd)
            {
                var tag = reader.ReadTag();
                var fieldNum = ProtoReader.GetFieldNumber(tag);
                var wireType = ProtoReader.GetWireType(tag);

                switch (fieldNum)
                {
                    case 1: // ir_version
                        irVersion = reader.ReadInt64();
                        break;
                    case 2: // producer_name
                        producerName = reader.ReadString();
                        break;
                    case 3: // producer_version
                        producerVersion = reader.ReadString();
                        break;
                    case 7: // graph
                        var graphSub = reader.ReadSubMessage();
                        graph = ParseGraph(ref graphSub);
                        break;
                    case 8: // opset_import (repeated)
                        var opsetSub = reader.ReadSubMessage();
                        opsetImports.Add(ParseOpsetImport(ref opsetSub));
                        break;
                    default:
                        reader.SkipField(wireType);
                        break;
                }
            }

            return new OnnxModel
            {
                IrVersion = irVersion,
                ProducerName = producerName,
                ProducerVersion = producerVersion,
                OpsetImports = opsetImports,
                Graph = graph ?? new OnnxGraph(),
            };
        }

        private static OnnxOpsetImport ParseOpsetImport(ref ProtoReader reader)
        {
            // OperatorSetIdProto: 1=domain (string), 2=version (int64)
            var domain = "";
            long version = 0;

            while (!reader.IsEnd)
            {
                var tag = reader.ReadTag();
                var fieldNum = ProtoReader.GetFieldNumber(tag);
                var wireType = ProtoReader.GetWireType(tag);

                switch (fieldNum)
                {
                    case 1: domain = reader.ReadString(); break;
                    case 2: version = reader.ReadInt64(); break;
                    default: reader.SkipField(wireType); break;
                }
            }

            return new OnnxOpsetImport { Domain = domain, Version = version };
        }

        private static OnnxGraph ParseGraph(ref ProtoReader reader)
        {
            // GraphProto fields:
            //   1: node (NodeProto, repeated)
            //   2: name (string)
            //   5: initializer (TensorProto, repeated)
            //   10: doc_string - skip
            //   11: input (ValueInfoProto, repeated)
            //   12: output (ValueInfoProto, repeated)
            //   13: value_info (ValueInfoProto, repeated) - shape inference, skip for MVP

            var name = "";
            var nodes = new List<OnnxNode>();
            var inputs = new List<OnnxValueInfo>();
            var outputs = new List<OnnxValueInfo>();
            var initializers = new List<OnnxTensor>();

            while (!reader.IsEnd)
            {
                var tag = reader.ReadTag();
                var fieldNum = ProtoReader.GetFieldNumber(tag);
                var wireType = ProtoReader.GetWireType(tag);

                switch (fieldNum)
                {
                    case 1: // node
                        var nodeSub = reader.ReadSubMessage();
                        nodes.Add(ParseNode(ref nodeSub));
                        break;
                    case 2: // name
                        name = reader.ReadString();
                        break;
                    case 5: // initializer
                        var initSub = reader.ReadSubMessage();
                        initializers.Add(ParseTensor(ref initSub));
                        break;
                    case 11: // input
                        var inSub = reader.ReadSubMessage();
                        inputs.Add(ParseValueInfo(ref inSub));
                        break;
                    case 12: // output
                        var outSub = reader.ReadSubMessage();
                        outputs.Add(ParseValueInfo(ref outSub));
                        break;
                    default:
                        reader.SkipField(wireType);
                        break;
                }
            }

            return new OnnxGraph
            {
                Name = name,
                Nodes = nodes,
                Inputs = inputs,
                Outputs = outputs,
                Initializers = initializers,
            };
        }

        private static OnnxNode ParseNode(ref ProtoReader reader)
        {
            // NodeProto fields:
            //   1: input (string, repeated)
            //   2: output (string, repeated)
            //   3: name (string)
            //   4: op_type (string)
            //   5: domain (string) - skip
            //   6: attribute (AttributeProto, repeated)
            //   6 was attribute - check schema again
            // Actually:
            //   1: input
            //   2: output
            //   3: name
            //   4: op_type
            //   5: attribute (AttributeProto, repeated)
            //   6: doc_string - skip
            //   7: domain - skip

            var inputs = new List<string>();
            var outputs = new List<string>();
            var name = "";
            var opType = "";
            var attributes = new Dictionary<string, OnnxAttribute>();

            while (!reader.IsEnd)
            {
                var tag = reader.ReadTag();
                var fieldNum = ProtoReader.GetFieldNumber(tag);
                var wireType = ProtoReader.GetWireType(tag);

                switch (fieldNum)
                {
                    case 1: inputs.Add(reader.ReadString()); break;
                    case 2: outputs.Add(reader.ReadString()); break;
                    case 3: name = reader.ReadString(); break;
                    case 4: opType = reader.ReadString(); break;
                    case 5:
                        var attrSub = reader.ReadSubMessage();
                        var attr = ParseAttribute(ref attrSub);
                        attributes[attr.Name] = attr;
                        break;
                    default: reader.SkipField(wireType); break;
                }
            }

            return new OnnxNode
            {
                Name = name,
                OpType = opType,
                Inputs = inputs,
                Outputs = outputs,
                Attributes = attributes,
            };
        }

        private static OnnxAttribute ParseAttribute(ref ProtoReader reader)
        {
            // AttributeProto fields:
            //   1: name (string)
            //   2: f (float)
            //   3: i (int64)
            //   4: s (bytes)
            //   5: t (TensorProto)
            //   7: floats (float, repeated)
            //   8: ints (int64, repeated)
            //   9: strings (bytes, repeated)
            //   10: tensors (TensorProto, repeated) - skip
            //   13: doc_string - skip
            //   20: type (AttributeType enum)

            var name = "";
            var type = OnnxAttributeType.Undefined;
            float f = 0;
            long i = 0;
            var s = "";
            OnnxTensor? t = null;
            var floats = new List<float>();
            var ints = new List<long>();

            while (!reader.IsEnd)
            {
                var tag = reader.ReadTag();
                var fieldNum = ProtoReader.GetFieldNumber(tag);
                var wireType = ProtoReader.GetWireType(tag);

                switch (fieldNum)
                {
                    case 1: name = reader.ReadString(); break;
                    case 2: f = reader.ReadFloat(); break;
                    case 3: i = reader.ReadInt64(); break;
                    case 4: s = reader.ReadString(); break;
                    case 5:
                        var tSub = reader.ReadSubMessage();
                        t = ParseTensor(ref tSub);
                        break;
                    case 7:
                        // floats can be packed (length-delimited) or unpacked (repeated fixed32)
                        if (wireType == WireType.LengthDelimited)
                        {
                            floats.AddRange(reader.ReadPackedFloat());
                        }
                        else
                        {
                            floats.Add(reader.ReadFloat());
                        }
                        break;
                    case 8:
                        if (wireType == WireType.LengthDelimited)
                        {
                            ints.AddRange(reader.ReadPackedInt64());
                        }
                        else
                        {
                            ints.Add(reader.ReadInt64());
                        }
                        break;
                    case 20: type = (OnnxAttributeType)reader.ReadInt32(); break;
                    default: reader.SkipField(wireType); break;
                }
            }

            return new OnnxAttribute
            {
                Name = name,
                Type = type,
                FloatValue = f,
                IntValue = i,
                StringValue = s,
                TensorValue = t,
                FloatArray = floats.ToArray(),
                IntArray = ints.ToArray(),
            };
        }

        private static OnnxTensor ParseTensor(ref ProtoReader reader)
        {
            // TensorProto fields (subset relevant for our parser):
            //   1: dims (int64, repeated)
            //   2: data_type (int32)
            //   3: segment - skip
            //   4: float_data (float, repeated, packed)
            //   5: int32_data - not for MVP
            //   6: string_data - not for MVP
            //   7: int64_data (int64, repeated, packed)
            //   8: name (string)
            //   9: raw_data (bytes)
            //   10: double_data - not for MVP
            //   13: external_data (StringStringEntryProto, repeated)
            //   14: data_location (enum: 0=DEFAULT, 1=EXTERNAL)

            var dims = new List<long>();
            var dataType = OnnxDataType.Undefined;
            var name = "";
            byte[] rawData = Array.Empty<byte>();
            float[]? floatData = null;
            long[]? int64Data = null;
            var externalEntries = new Dictionary<string, string>();
            var dataLocation = 0; // 0 = DEFAULT (inline), 1 = EXTERNAL

            while (!reader.IsEnd)
            {
                var tag = reader.ReadTag();
                var fieldNum = ProtoReader.GetFieldNumber(tag);
                var wireType = ProtoReader.GetWireType(tag);

                switch (fieldNum)
                {
                    case 1:
                        if (wireType == WireType.LengthDelimited)
                        {
                            dims.AddRange(reader.ReadPackedInt64());
                        }
                        else
                        {
                            dims.Add(reader.ReadInt64());
                        }
                        break;
                    case 2: dataType = (OnnxDataType)reader.ReadInt32(); break;
                    case 4:
                        if (wireType == WireType.LengthDelimited)
                        {
                            floatData = reader.ReadPackedFloat();
                        }
                        else
                        {
                            // unpacked
                            floatData ??= Array.Empty<float>();
                            var newArr = new float[floatData.Length + 1];
                            floatData.CopyTo(newArr, 0);
                            newArr[^1] = reader.ReadFloat();
                            floatData = newArr;
                        }
                        break;
                    case 7:
                        if (wireType == WireType.LengthDelimited)
                        {
                            int64Data = reader.ReadPackedInt64().ToArray();
                        }
                        else
                        {
                            int64Data ??= Array.Empty<long>();
                            var newArr = new long[int64Data.Length + 1];
                            int64Data.CopyTo(newArr, 0);
                            newArr[^1] = reader.ReadInt64();
                            int64Data = newArr;
                        }
                        break;
                    case 8: name = reader.ReadString(); break;
                    case 9: rawData = reader.ReadBytes().ToArray(); break;
                    case 13: // external_data: StringStringEntryProto { 1=key, 2=value }
                        var entrySub = reader.ReadSubMessage();
                        var (key, value) = ParseStringStringEntry(ref entrySub);
                        externalEntries[key] = value;
                        break;
                    case 14: dataLocation = reader.ReadInt32(); break;
                    default: reader.SkipField(wireType); break;
                }
            }

            OnnxExternalDataInfo? extInfo = null;
            if (dataLocation == 1 && externalEntries.Count > 0)
            {
                long offset = 0, length = 0;
                if (externalEntries.TryGetValue("offset", out var offsetStr))
                {
                    long.TryParse(offsetStr, out offset);
                }
                if (externalEntries.TryGetValue("length", out var lengthStr))
                {
                    long.TryParse(lengthStr, out length);
                }
                externalEntries.TryGetValue("location", out var location);

                extInfo = new OnnxExternalDataInfo
                {
                    Location = location ?? "",
                    Offset = offset,
                    Length = length,
                };
            }

            return new OnnxTensor
            {
                Name = name,
                DataType = dataType,
                Dims = dims.ToArray(),
                RawData = rawData,
                FloatData = floatData,
                Int64Data = int64Data,
                ExternalData = extInfo,
            };
        }

        private static (string Key, string Value) ParseStringStringEntry(ref ProtoReader reader)
        {
            // StringStringEntryProto: 1=key (string), 2=value (string)
            var key = "";
            var value = "";

            while (!reader.IsEnd)
            {
                var tag = reader.ReadTag();
                var fieldNum = ProtoReader.GetFieldNumber(tag);
                var wireType = ProtoReader.GetWireType(tag);

                switch (fieldNum)
                {
                    case 1: key = reader.ReadString(); break;
                    case 2: value = reader.ReadString(); break;
                    default: reader.SkipField(wireType); break;
                }
            }

            return (key, value);
        }

        private static OnnxValueInfo ParseValueInfo(ref ProtoReader reader)
        {
            // ValueInfoProto fields:
            //   1: name (string)
            //   2: type (TypeProto)
            //   3: doc_string - skip
            // TypeProto:
            //   1: tensor_type (TypeProto.Tensor)
            // TypeProto.Tensor:
            //   1: elem_type (int32)
            //   2: shape (TensorShapeProto)
            // TensorShapeProto:
            //   1: dim (Dimension, repeated)
            // Dimension:
            //   1: dim_value (int64)
            //   2: dim_param (string) - symbolic, treat as null

            var name = "";
            var dataType = OnnxDataType.Undefined;
            long?[] shape = Array.Empty<long?>();

            while (!reader.IsEnd)
            {
                var tag = reader.ReadTag();
                var fieldNum = ProtoReader.GetFieldNumber(tag);
                var wireType = ProtoReader.GetWireType(tag);

                switch (fieldNum)
                {
                    case 1: name = reader.ReadString(); break;
                    case 2:
                        var typeSub = reader.ReadSubMessage();
                        (dataType, shape) = ParseTypeProto(ref typeSub);
                        break;
                    default: reader.SkipField(wireType); break;
                }
            }

            return new OnnxValueInfo
            {
                Name = name,
                DataType = dataType,
                Shape = shape,
            };
        }

        private static (OnnxDataType, long?[]) ParseTypeProto(ref ProtoReader reader)
        {
            var dataType = OnnxDataType.Undefined;
            long?[] shape = Array.Empty<long?>();

            while (!reader.IsEnd)
            {
                var tag = reader.ReadTag();
                var fieldNum = ProtoReader.GetFieldNumber(tag);
                var wireType = ProtoReader.GetWireType(tag);

                if (fieldNum == 1) // tensor_type
                {
                    var sub = reader.ReadSubMessage();

                    while (!sub.IsEnd)
                    {
                        var t = sub.ReadTag();
                        var fn = ProtoReader.GetFieldNumber(t);
                        var wt = ProtoReader.GetWireType(t);

                        switch (fn)
                        {
                            case 1: dataType = (OnnxDataType)sub.ReadInt32(); break;
                            case 2:
                                var shapeSub = sub.ReadSubMessage();
                                shape = ParseTensorShape(ref shapeSub);
                                break;
                            default: sub.SkipField(wt); break;
                        }
                    }
                }
                else
                {
                    reader.SkipField(wireType);
                }
            }

            return (dataType, shape);
        }

        private static long?[] ParseTensorShape(ref ProtoReader reader)
        {
            var dims = new List<long?>();

            while (!reader.IsEnd)
            {
                var tag = reader.ReadTag();
                var fieldNum = ProtoReader.GetFieldNumber(tag);
                var wireType = ProtoReader.GetWireType(tag);

                if (fieldNum == 1) // dim
                {
                    var sub = reader.ReadSubMessage();
                    long? dim = null;

                    while (!sub.IsEnd)
                    {
                        var t = sub.ReadTag();
                        var fn = ProtoReader.GetFieldNumber(t);
                        var wt = ProtoReader.GetWireType(t);

                        switch (fn)
                        {
                            case 1: dim = sub.ReadInt64(); break;     // dim_value
                            case 2: sub.ReadString(); dim = null; break; // dim_param (symbolic)
                            default: sub.SkipField(wt); break;
                        }
                    }

                    dims.Add(dim);
                }
                else
                {
                    reader.SkipField(wireType);
                }
            }

            return dims.ToArray();
        }
    }
}