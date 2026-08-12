// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Text;

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// Stable, token-free names for anything an IL instruction or a signature can point at.
    ///
    /// <para><b>This is the piece that makes the comparison mean something.</b> A metadata token embedded in
    /// IL is a row index into a table, and row indices move when an unrelated member is added — insert one
    /// private field near the top of a type and the raw bytes of methods that were not touched change. A
    /// comparator that reads those bytes literally reports "the IL changed" on a package where nothing did,
    /// which is worse than no comparator at all because it is confidently wrong. Every token is therefore
    /// resolved to the name it refers to, and the name is compared.</para>
    /// </summary>
    internal sealed class MetadataNames
    {
        private readonly MetadataReader _reader;
        private readonly SignatureTypeNameProvider _provider;

        internal MetadataNames(MetadataReader reader)
        {
            _reader = reader ?? throw new ArgumentNullException(nameof(reader));
            _provider = new SignatureTypeNameProvider(reader);
        }

        internal MetadataReader Reader
        {
            get { return _reader; }
        }

        internal SignatureTypeNameProvider Provider
        {
            get { return _provider; }
        }

        /// <summary>
        /// The identity a method keeps across rebuilds: declaring type, name, generic arity, parameter types
        /// and return type. Parameter <i>names</i> are excluded on purpose — they are not part of the compiled
        /// call, and including them here would make a rename look like a different method and produce a
        /// spurious added/removed pair. The public-surface comparison does look at them.
        /// </summary>
        internal string MethodKey(MethodDefinitionHandle handle)
        {
            var definition = _reader.GetMethodDefinition(handle);
            var signature = definition.DecodeSignature(_provider, null);
            var arity = signature.GenericParameterCount;
            var builder = new StringBuilder();

            builder.Append(_provider.TypeName(definition.GetDeclaringType()));
            builder.Append('.');
            builder.Append(_reader.GetString(definition.Name));

            if (arity > 0)
            {
                builder.Append('`').Append(arity);
            }

            builder.Append('(');
            builder.Append(string.Join(", ", signature.ParameterTypes));
            builder.Append(") : ");
            builder.Append(signature.ReturnType);

            return builder.ToString();
        }

        /// <summary>The identity a field keeps across rebuilds: declaring type, name and type.</summary>
        internal string FieldKey(FieldDefinitionHandle handle)
        {
            var definition = _reader.GetFieldDefinition(handle);

            return _provider.TypeName(definition.GetDeclaringType())
                   + "."
                   + _reader.GetString(definition.Name)
                   + " : "
                   + definition.DecodeSignature(_provider, null);
        }

        /// <summary>
        /// A stable name for whatever an IL operand token points at.
        ///
        /// <para>Unknown handle kinds fall back to the kind name plus the row number. That is a deliberate
        /// partial answer rather than a throw: the row number is exactly the unstable value this class exists
        /// to remove, so any such fallback is a potential false positive and is called out by
        /// <see cref="UnresolvedHandleKinds"/> rather than hidden.</para>
        /// </summary>
        internal string EntityName(EntityHandle handle)
        {
            if (handle.IsNil)
            {
                return "<nil>";
            }

            switch (handle.Kind)
            {
                case HandleKind.MethodDefinition:
                    return MethodKey((MethodDefinitionHandle)handle);

                case HandleKind.FieldDefinition:
                    return FieldKey((FieldDefinitionHandle)handle);

                case HandleKind.TypeDefinition:
                    return _provider.TypeName((TypeDefinitionHandle)handle);

                case HandleKind.TypeReference:
                    return _provider.TypeName((TypeReferenceHandle)handle);

                case HandleKind.TypeSpecification:
                    return _reader.GetTypeSpecification((TypeSpecificationHandle)handle)
                        .DecodeSignature(_provider, null);

                case HandleKind.MemberReference:
                    return MemberReferenceName((MemberReferenceHandle)handle);

                case HandleKind.MethodSpecification:
                    return MethodSpecificationName((MethodSpecificationHandle)handle);

                case HandleKind.StandaloneSignature:
                    return StandaloneSignatureName((StandaloneSignatureHandle)handle);

                default:
                    UnresolvedHandleKinds.Add(handle.Kind);

                    return handle.Kind + "#" + MetadataTokens.GetRowNumber(handle)
                        .ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        /// <summary>Handle kinds this class had to fall back to a row number for, if any.</summary>
        internal HashSet<HandleKind> UnresolvedHandleKinds { get; } = new HashSet<HandleKind>();

        private string MemberReferenceName(MemberReferenceHandle handle)
        {
            var reference = _reader.GetMemberReference(handle);
            var parent = EntityName(reference.Parent);
            var name = _reader.GetString(reference.Name);

            if (reference.GetKind() == MemberReferenceKind.Method)
            {
                var signature = reference.DecodeMethodSignature(_provider, null);
                var arity = signature.GenericParameterCount > 0
                    ? "`" + signature.GenericParameterCount.ToString(
                        System.Globalization.CultureInfo.InvariantCulture)
                    : string.Empty;

                return parent + "." + name + arity + "("
                       + string.Join(", ", signature.ParameterTypes) + ") : " + signature.ReturnType;
            }

            return parent + "." + name + " : " + reference.DecodeFieldSignature(_provider, null);
        }

        private string MethodSpecificationName(MethodSpecificationHandle handle)
        {
            var specification = _reader.GetMethodSpecification(handle);
            var arguments = specification.DecodeSignature(_provider, null);

            return EntityName(specification.Method) + "<" + string.Join(", ", arguments) + ">";
        }

        private string StandaloneSignatureName(StandaloneSignatureHandle handle)
        {
            var signature = _reader.GetStandaloneSignature(handle);

            // A standalone signature is either a local-variable list or a calli site. The blob's first byte
            // says which, and DecodeMethodSignature throws on a local list, so the kind is checked first.
            if (signature.GetKind() == StandaloneSignatureKind.LocalVariables)
            {
                return "locals(" + string.Join(", ", signature.DecodeLocalSignature(_provider, null)) + ")";
            }

            var method = signature.DecodeMethodSignature(_provider, null);

            return "calli(" + string.Join(", ", method.ParameterTypes) + ") : " + method.ReturnType;
        }
    }
}
