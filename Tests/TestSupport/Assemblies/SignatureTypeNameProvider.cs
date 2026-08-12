// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using System.Reflection.Metadata;

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// Turns an ECMA-335 signature blob into a stable type-name string, without loading the assembly.
    ///
    /// <para><b>Why strings and not <see cref="Type"/>.</b> The whole point of this comparator is to read
    /// assemblies it cannot execute — a different target framework, a different architecture, a package that
    /// was never restored for this runtime. <c>Assembly.LoadFrom</c> would resolve every reference and fail on
    /// the first one that is not present; a signature decoder that produces text resolves nothing.</para>
    ///
    /// <para><b>Generic parameters are printed positionally</b> (<c>!0</c> for a type parameter, <c>!!0</c> for
    /// a method one), so no generic context has to be threaded through — which is why the context type is
    /// <see cref="object"/> and every call passes <see langword="null"/>. Positional is also the right answer
    /// for a comparison: renaming <c>T</c> to <c>TItem</c> changes no compiled behaviour, and a comparator
    /// that reported it as a difference would be reporting the source, not the artefact. It IS reported at the
    /// public-surface level, where a type-parameter name is part of the documented API.</para>
    ///
    /// <para><b>Assembly-qualified, but without the version.</b> A type reference is printed as
    /// <c>[System.Runtime]System.Console</c>. The name is included because two same-named types from different
    /// assemblies are genuinely different; the version is deliberately excluded because a servicing bump moves
    /// every reference version and would make every method in the assembly report an IL difference — the exact
    /// false positive this class exists to avoid. Reference versions are compared separately and reported as
    /// their own category (see <see cref="AssemblyFacts.ReferencedAssemblies"/>).</para>
    /// </summary>
    internal sealed class SignatureTypeNameProvider
        : ISignatureTypeProvider<string, object>, ICustomAttributeTypeProvider<string>
    {
        private readonly MetadataReader _reader;

        internal SignatureTypeNameProvider(MetadataReader reader)
        {
            _reader = reader ?? throw new ArgumentNullException(nameof(reader));
        }

        /// <summary>The full name of a type defined in this assembly, with <c>+</c> between nesting levels.</summary>
        internal string TypeName(TypeDefinitionHandle handle)
        {
            var definition = _reader.GetTypeDefinition(handle);
            var name = _reader.GetString(definition.Name);

            // BOUND: the walk follows the declaring-type chain, which ECMA-335 requires to be acyclic and
            // which is at most as deep as the source nesting (single digits in practice).
            if (definition.IsNested)
            {
                return TypeName(definition.GetDeclaringType()) + "+" + name;
            }

            var space = _reader.GetString(definition.Namespace);

            return space.Length == 0 ? name : space + "." + name;
        }

        /// <summary>The full name of a type referenced from another assembly, prefixed with that assembly.</summary>
        internal string TypeName(TypeReferenceHandle handle)
        {
            var reference = _reader.GetTypeReference(handle);
            var name = _reader.GetString(reference.Name);
            var space = _reader.GetString(reference.Namespace);
            var qualified = space.Length == 0 ? name : space + "." + name;
            var scope = reference.ResolutionScope;

            if (scope.Kind == HandleKind.TypeReference)
            {
                // A nested type: the scope is the declaring type reference, which already carries the
                // assembly prefix, so it is not repeated here.
                // BOUND: same acyclic nesting chain as above.
                return TypeName((TypeReferenceHandle)scope) + "+" + name;
            }

            if (scope.Kind == HandleKind.AssemblyReference)
            {
                var assembly = _reader.GetAssemblyReference((AssemblyReferenceHandle)scope);

                return "[" + _reader.GetString(assembly.Name) + "]" + qualified;
            }

            return qualified;
        }

        public string GetPrimitiveType(PrimitiveTypeCode typeCode)
        {
            return typeCode switch
            {
                PrimitiveTypeCode.Void => "System.Void",
                PrimitiveTypeCode.Boolean => "System.Boolean",
                PrimitiveTypeCode.Char => "System.Char",
                PrimitiveTypeCode.SByte => "System.SByte",
                PrimitiveTypeCode.Byte => "System.Byte",
                PrimitiveTypeCode.Int16 => "System.Int16",
                PrimitiveTypeCode.UInt16 => "System.UInt16",
                PrimitiveTypeCode.Int32 => "System.Int32",
                PrimitiveTypeCode.UInt32 => "System.UInt32",
                PrimitiveTypeCode.Int64 => "System.Int64",
                PrimitiveTypeCode.UInt64 => "System.UInt64",
                PrimitiveTypeCode.Single => "System.Single",
                PrimitiveTypeCode.Double => "System.Double",
                PrimitiveTypeCode.String => "System.String",
                PrimitiveTypeCode.TypedReference => "System.TypedReference",
                PrimitiveTypeCode.IntPtr => "System.IntPtr",
                PrimitiveTypeCode.UIntPtr => "System.UIntPtr",
                PrimitiveTypeCode.Object => "System.Object",
                _ => typeCode.ToString(),
            };
        }

        public string GetTypeFromDefinition(MetadataReader reader, TypeDefinitionHandle handle, byte rawTypeKind)
        {
            return TypeName(handle);
        }

        public string GetTypeFromReference(MetadataReader reader, TypeReferenceHandle handle, byte rawTypeKind)
        {
            return TypeName(handle);
        }

        public string GetTypeFromSpecification(
            MetadataReader reader,
            object genericContext,
            TypeSpecificationHandle handle,
            byte rawTypeKind)
        {
            return reader.GetTypeSpecification(handle).DecodeSignature(this, genericContext);
        }

        public string GetSZArrayType(string elementType)
        {
            return elementType + "[]";
        }

        public string GetArrayType(string elementType, ArrayShape shape)
        {
            return elementType + "[" + new string(',', Math.Max(0, shape.Rank - 1)) + "]";
        }

        public string GetByReferenceType(string elementType)
        {
            return elementType + "&";
        }

        public string GetPointerType(string elementType)
        {
            return elementType + "*";
        }

        public string GetGenericInstantiation(string genericType, ImmutableArray<string> typeArguments)
        {
            return genericType + "<" + string.Join(", ", typeArguments) + ">";
        }

        public string GetGenericMethodParameter(object genericContext, int index)
        {
            return "!!" + index.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }

        public string GetGenericTypeParameter(object genericContext, int index)
        {
            return "!" + index.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }

        public string GetModifiedType(string modifier, string unmodifiedType, bool isRequired)
        {
            return unmodifiedType + (isRequired ? " modreq(" : " modopt(") + modifier + ")";
        }

        public string GetPinnedType(string elementType)
        {
            return elementType + " pinned";
        }

        public string GetFunctionPointerType(MethodSignature<string> signature)
        {
            return "method " + signature.ReturnType + " *(" + string.Join(", ", signature.ParameterTypes) + ")";
        }

        public string GetSystemType()
        {
            return "System.Type";
        }

        public bool IsSystemType(string type)
        {
            return type == "System.Type";
        }

        public string GetTypeFromSerializedName(string name)
        {
            return name;
        }

        public PrimitiveTypeCode GetUnderlyingEnumType(string type)
        {
            // Reached only for an enum-typed custom-attribute argument. Nothing this comparator decodes today
            // (AssemblyInformationalVersion and friends are strings), and guessing Int32 would silently
            // mis-decode a byte- or long-backed enum, so it refuses rather than inventing a width.
            throw new NotSupportedException(
                "enum-typed custom attribute arguments are not decoded by the assembly comparator (type: "
                + type + ")");
        }
    }
}
