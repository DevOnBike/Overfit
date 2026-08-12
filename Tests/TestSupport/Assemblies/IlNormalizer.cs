// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Reflection;
using System.Reflection.Emit;
using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;
using System.Text;

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// Renders one method body as text in which nothing depends on where its neighbours ended up.
    ///
    /// <para><b>The decision, stated plainly because the task turns on it.</b> Raw IL bytes are NOT compared.
    /// Every operand that is a metadata token is replaced by the resolved name of the thing it points at, and
    /// the resulting text is what gets hashed. Comparing raw bytes is cheaper and wrong for the use this
    /// exists for: adding a method to a type shifts row indices, which shifts tokens, which changes the bytes
    /// of methods whose source was not touched. On a real servicing bump that produces a page of false
    /// differences and the reader learns to ignore the tool.</para>
    ///
    /// <para><b>What that costs, and it is not nothing.</b> Two things are now invisible to this comparison
    /// that a raw byte compare would have caught:</para>
    /// <list type="bullet">
    ///   <item>A change to metadata <i>layout</i> alone — different row ordering with identical logic. That is
    ///   correct to ignore for "did the behaviour change", and wrong to ignore for "is this bit-identical".
    ///   Use <see cref="AssemblyFacts.MetadataHeapSizes"/> for the latter question.</item>
    ///   <item>A rename that this resolver prints identically. Type parameters are printed positionally
    ///   (<c>!0</c>), so renaming <c>T</c> to <c>TItem</c> shows no IL difference — which is right, since the
    ///   compiled call is unchanged — and assembly reference <i>versions</i> are excluded from printed type
    ///   names, so a rebind of a dependency shows up in
    ///   <see cref="AssemblyFacts.ReferencedAssemblies"/> and not here.</item>
    /// </list>
    ///
    /// <para><b>Branch targets are kept as the raw relative offsets the IL carries</b>, not resolved to labels.
    /// They are already position-independent with respect to the rest of the assembly, and every token operand
    /// is a fixed four bytes, so normalising an operand never moves a later instruction.</para>
    ///
    /// <para><b>The opcode table is built from <see cref="OpCodes"/> by reflection</b> rather than typed out.
    /// A hand-written table of 226 entries is a place to make a silent transcription error, and the failure
    /// mode of a wrong operand width is not an exception — it is a decoder that runs off into the middle of an
    /// instruction and produces a stable, meaningless answer. Reflection is fine here; the ban lives in
    /// <c>Sources/Main</c>, not in the test project.</para>
    /// </summary>
    internal static class IlNormalizer
    {
        private static readonly Dictionary<short, OpCode> Opcodes = BuildOpcodeTable();

        /// <summary>
        /// The method body as normalised text, or <see langword="null"/> when the method has no managed body —
        /// abstract, extern, P/Invoke or runtime-implemented.
        ///
        /// <para><b>Null is not "empty".</b> An abstract method and a method with an empty body are different
        /// facts, and collapsing them would let a method that lost its body compare equal to one that never
        /// had one.</para>
        /// </summary>
        internal static string Normalize(PEReader peReader, MetadataNames names, MethodDefinitionHandle handle)
        {
            ArgumentNullException.ThrowIfNull(peReader);
            ArgumentNullException.ThrowIfNull(names);

            var definition = names.Reader.GetMethodDefinition(handle);
            var address = definition.RelativeVirtualAddress;

            if (address == 0)
            {
                return null;
            }

            var body = peReader.GetMethodBody(address);
            var builder = new StringBuilder();

            AppendLocals(builder, names, body);
            AppendInstructions(builder, names, body.GetILBytes());
            AppendExceptionRegions(builder, names, body);

            return builder.ToString();
        }

        private static void AppendLocals(StringBuilder builder, MetadataNames names, MethodBodyBlock body)
        {
            if (body.LocalSignature.IsNil)
            {
                return;
            }

            var signature = names.Reader.GetStandaloneSignature(body.LocalSignature);
            var locals = signature.DecodeLocalSignature(names.Provider, null);

            builder.Append(".locals");

            if (body.LocalVariablesInitialized)
            {
                builder.Append(" init");
            }

            builder.Append(" (").Append(string.Join(", ", locals)).Append(")\n");
        }

        private static void AppendInstructions(StringBuilder builder, MetadataNames names, byte[] il)
        {
            var offset = 0;

            // BOUND: `offset` strictly increases by at least one byte per iteration (every opcode is one or
            // two bytes and every operand length is non-negative), so the walk is bounded by il.Length.
            while (offset < il.Length)
            {
                var code = il[offset];
                var key = (short)code;
                var size = 1;

                if (code == 0xFE && offset + 1 < il.Length)
                {
                    key = unchecked((short)(0xFE00 | il[offset + 1]));
                    size = 2;
                }

                if (!Opcodes.TryGetValue(key, out var opcode))
                {
                    // An undecodable byte means the walk has lost sync, and everything after it would be
                    // fabricated. Emitting the remaining bytes verbatim keeps the comparison honest — two
                    // bodies that really are identical still compare equal — and makes the failure visible.
                    builder.Append("<undecodable 0x").Append(code.ToString("x2"))
                        .Append(" at ").Append(offset).Append("> ")
                        .Append(Convert.ToHexString(il, offset, il.Length - offset)).Append('\n');

                    return;
                }

                offset += size;
                builder.Append(opcode.Name);
                offset = AppendOperand(builder, names, il, offset, opcode);
                builder.Append('\n');
            }
        }

        private static int AppendOperand(
            StringBuilder builder,
            MetadataNames names,
            byte[] il,
            int offset,
            OpCode opcode)
        {
            switch (opcode.OperandType)
            {
                case OperandType.InlineNone:
                    return offset;

                case OperandType.ShortInlineBrTarget:
                    builder.Append(' ').Append((sbyte)il[offset]);

                    return offset + 1;

                case OperandType.ShortInlineI:
                    builder.Append(' ').Append((sbyte)il[offset]);

                    return offset + 1;

                case OperandType.ShortInlineVar:
                    builder.Append(' ').Append(il[offset]);

                    return offset + 1;

                case OperandType.InlineVar:
                    builder.Append(' ').Append(BitConverter.ToUInt16(il, offset));

                    return offset + 2;

                case OperandType.InlineBrTarget:
                case OperandType.InlineI:
                    builder.Append(' ').Append(BitConverter.ToInt32(il, offset));

                    return offset + 4;

                case OperandType.ShortInlineR:
                    builder.Append(' ').Append(BitConverter.ToSingle(il, offset)
                        .ToString("R", System.Globalization.CultureInfo.InvariantCulture));

                    return offset + 4;

                case OperandType.InlineI8:
                    builder.Append(' ').Append(BitConverter.ToInt64(il, offset));

                    return offset + 8;

                case OperandType.InlineR:
                    builder.Append(' ').Append(BitConverter.ToDouble(il, offset)
                        .ToString("R", System.Globalization.CultureInfo.InvariantCulture));

                    return offset + 8;

                case OperandType.InlineString:
                    {
                        var token = BitConverter.ToInt32(il, offset);
                        var handle = MetadataTokens.UserStringHandle(token);

                        // The literal itself, not its heap index — a #US heap offset moves when any earlier
                        // literal changes length, and the string is what the method actually pushes.
                        builder.Append(" \"").Append(names.Reader.GetUserString(handle)).Append('"');

                        return offset + 4;
                    }

                case OperandType.InlineField:
                case OperandType.InlineMethod:
                case OperandType.InlineSig:
                case OperandType.InlineTok:
                case OperandType.InlineType:
                    {
                        var token = BitConverter.ToInt32(il, offset);
                        var handle = MetadataTokens.EntityHandle(token);

                        builder.Append(' ').Append(names.EntityName(handle));

                        return offset + 4;
                    }

                case OperandType.InlineSwitch:
                    {
                        var count = BitConverter.ToInt32(il, offset);

                        offset += 4;
                        builder.Append(" (");

                        // BOUND: exactly `count` targets, and `count` is read from the instruction itself,
                        // which the CLI validator constrains to the remaining body length.
                        for (var index = 0; index < count; index++)
                        {
                            if (index > 0)
                            {
                                builder.Append(", ");
                            }

                            builder.Append(BitConverter.ToInt32(il, offset));
                            offset += 4;
                        }

                        builder.Append(')');

                        return offset;
                    }

                default:
                    throw new NotSupportedException("unhandled IL operand type " + opcode.OperandType);
            }
        }

        private static void AppendExceptionRegions(
            StringBuilder builder,
            MetadataNames names,
            MethodBodyBlock body)
        {
            // BOUND: one iteration per region recorded in the method header.
            foreach (var region in body.ExceptionRegions)
            {
                builder.Append(".try ").Append(region.Kind)
                    .Append(" try=").Append(region.TryOffset).Append('+').Append(region.TryLength)
                    .Append(" handler=").Append(region.HandlerOffset).Append('+').Append(region.HandlerLength);

                if (region.Kind == ExceptionRegionKind.Catch)
                {
                    builder.Append(" catch=").Append(names.EntityName(region.CatchType));
                }

                if (region.Kind == ExceptionRegionKind.Filter)
                {
                    builder.Append(" filter=").Append(region.FilterOffset);
                }

                builder.Append('\n');
            }
        }

        private static Dictionary<short, OpCode> BuildOpcodeTable()
        {
            var table = new Dictionary<short, OpCode>();

            // BOUND: one iteration per public static field of OpCodes — a fixed set of 226 in this runtime.
            foreach (var field in typeof(OpCodes).GetFields(BindingFlags.Public | BindingFlags.Static))
            {
                if (field.GetValue(null) is OpCode opcode)
                {
                    table[opcode.Value] = opcode;
                }
            }

            if (table.Count == 0)
            {
                throw new InvalidOperationException(
                    "the IL opcode table came out empty, which would make every method body decode as "
                    + "undecodable and therefore compare equal to every other one");
            }

            return table;
        }
    }
}
