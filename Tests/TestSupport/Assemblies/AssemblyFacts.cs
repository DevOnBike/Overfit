// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections.Immutable;
using System.Globalization;
using System.Reflection;
using System.Reflection.Metadata;
using System.Reflection.Metadata.Ecma335;
using System.Reflection.PortableExecutable;
using System.Security.Cryptography;
using System.Text;

namespace DevOnBike.Overfit.Tests.TestSupport.Assemblies
{
    /// <summary>
    /// Everything the comparator needs from one assembly, read straight out of the PE file.
    ///
    /// <para><b>Nothing is loaded.</b> No <c>Assembly.LoadFrom</c>, no resolution of references, no execution.
    /// That is what lets this read an assembly built for a different target framework or a different
    /// architecture than the process doing the reading — which is the normal case when the question is "can I
    /// take this package", because the package under examination is very often not the one currently
    /// restored.</para>
    ///
    /// <para><b>The object stays open.</b> <see cref="NormalizedIl"/> re-decodes a body on demand for
    /// reporting, so only a 32-byte hash per method is retained up front. On a real package that is the
    /// difference between a few hundred kilobytes and tens of megabytes of normalised text nobody will read.
    /// Dispose it when done.</para>
    /// </summary>
    internal sealed class AssemblyFacts : IDisposable
    {
        private readonly PEReader _peReader;
        private readonly MetadataNames _names;
        private readonly Dictionary<string, MethodDefinitionHandle> _methodHandles =
            new Dictionary<string, MethodDefinitionHandle>(StringComparer.Ordinal);

        private AssemblyFacts(string name, PEReader peReader)
        {
            Name = name;
            _peReader = peReader;

            if (!peReader.HasMetadata)
            {
                throw new InvalidOperationException(
                    "'" + name + "' has no managed metadata. This comparator reads ECMA-335 and says nothing "
                    + "whatever about a native binary — see the class remarks.");
            }

            _names = new MetadataNames(peReader.GetMetadataReader());

            ReadModuleFacts();
            ReadMethodFacts();
            ReadPublicApi();
        }

        /// <summary>Reads an assembly from disk.</summary>
        internal static AssemblyFacts FromFile(string path)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(path);

            if (!File.Exists(path))
            {
                throw new FileNotFoundException("assembly not found: " + path, path);
            }

            return FromImage(File.ReadAllBytes(path), Path.GetFileName(path));
        }

        /// <summary>Reads an assembly already in memory — how the tests drive it, with no fixture on disk.</summary>
        internal static AssemblyFacts FromImage(byte[] image, string name)
        {
            ArgumentNullException.ThrowIfNull(image);

            // The image is read as a FILE, not as a loaded image, so every data directory the inert-difference
            // checks look at (certificate table above all) is at its on-disk offset. Getting this backwards
            // makes the certificate table read as absent on every input, and the whole System.Numerics.Tensors
            // result was "the delta is the certificate table".
            var peReader = new PEReader(ImmutableArray.Create(image));

            try
            {
                return new AssemblyFacts(name, peReader);
            }
            catch
            {
                peReader.Dispose();

                throw;
            }
        }

        /// <summary>A label for reporting — the file name, or whatever the caller passed.</summary>
        internal string Name
        {
            get;
        }

        /// <summary>The assembly's own simple name. Changing it breaks every compiled consumer.</summary>
        internal string AssemblyName
        {
            get; private set;
        }

        /// <summary>The public key, hex encoded, or empty when the assembly is not strong-named.</summary>
        internal string PublicKey { get; private set; } = string.Empty;

        /// <summary>The module version id: a fresh GUID on every build, identical source or not.</summary>
        internal Guid Mvid
        {
            get; private set;
        }

        internal Version AssemblyVersion
        {
            get; private set;
        }

        /// <summary>The <c>AssemblyInformationalVersion</c> attribute, which source-linked builds stamp the commit into.</summary>
        internal string InformationalVersion
        {
            get; private set;
        }

        internal int PeTimeDateStamp
        {
            get; private set;
        }

        internal uint PeCheckSum
        {
            get; private set;
        }

        /// <summary>Size of PE data directory 4 — the Authenticode certificate table. Zero when unsigned.</summary>
        internal int CertificateTableSize
        {
            get; private set;
        }

        internal int StrongNameSignatureSize
        {
            get; private set;
        }

        /// <summary>Debug directory entries rendered as text: PDB checksum, CodeView GUID and path.</summary>
        internal string DebugDirectory
        {
            get; private set;
        }

        /// <summary><see langword="false"/> when the image carries precompiled native code alongside its IL.</summary>
        internal bool IsIlOnly
        {
            get; private set;
        }

        /// <summary>
        /// <see langword="true"/> when the image carries a ReadyToRun / NGen native header.
        ///
        /// <para>This flag is why the comparator can be trusted to say what it does not know: identical IL in
        /// an R2R image does <b>not</b> imply identical execution, because the precompiled native code is a
        /// separate payload this comparison never reads.</para>
        /// </summary>
        internal bool HasPrecompiledNativeCode
        {
            get; private set;
        }

        /// <summary>Methods with no managed body: abstract, extern, P/Invoke or runtime-implemented.</summary>
        internal int MethodsWithoutManagedBody
        {
            get; private set;
        }

        /// <summary>Total bytes of method-body IL across the assembly — the "IL region" figure.</summary>
        internal int TotalIlByteCount
        {
            get; private set;
        }

        /// <summary>Sizes of the metadata heaps, for the "is this bit-identical" question the IL compare drops.</summary>
        internal string MetadataHeapSizes
        {
            get; private set;
        }

        /// <summary>Referenced assemblies as <c>Name/Version</c>, sorted.</summary>
        internal IReadOnlyList<string> ReferencedAssemblies { get; private set; } = Array.Empty<string>();

        /// <summary>
        /// Full names of types this assembly forwards elsewhere via <c>TypeForwardedToAttribute</c>.
        ///
        /// <para>Moving a type to another assembly is <b>allowed</b> when the old assembly forwards it. Without
        /// this set the classifier would report every forwarded type as a removal, which is the single
        /// most-reported false positive against <c>ApiCompat</c> itself.</para>
        /// </summary>
        internal IReadOnlySet<string> ForwardedTypes
        {
            get; private set;
        } =
            new HashSet<string>(StringComparer.Ordinal);

        /// <summary>SHA-256 of the normalised IL of every method, keyed by <see cref="MetadataNames.MethodKey"/>.</summary>
        internal IReadOnlyDictionary<string, string> MethodIl
        {
            get; private set;
        } =
            new Dictionary<string, string>();

        /// <summary>Every externally visible member.</summary>
        internal IReadOnlyList<ApiMember> PublicApi { get; private set; } = Array.Empty<ApiMember>();

        /// <summary>Handle kinds the token resolver could not name, if any — see <see cref="MetadataNames"/>.</summary>
        internal IReadOnlyCollection<string> UnresolvedTokenKinds
        {
            get
            {
                var kinds = new List<string>();

                // BOUND: one iteration per distinct handle kind encountered, which is bounded by the enum.
                foreach (var kind in _names.UnresolvedHandleKinds)
                {
                    kinds.Add(kind.ToString());
                }

                kinds.Sort(StringComparer.Ordinal);

                return kinds;
            }
        }

        /// <summary>The normalised IL text of one method, re-decoded on demand for a failure message.</summary>
        internal string NormalizedIl(string methodKey)
        {
            if (!_methodHandles.TryGetValue(methodKey, out var handle))
            {
                return null;
            }

            return IlNormalizer.Normalize(_peReader, _names, handle);
        }

        public void Dispose()
        {
            _peReader.Dispose();
        }

        private void ReadModuleFacts()
        {
            var reader = _names.Reader;
            var headers = _peReader.PEHeaders;

            Mvid = reader.GetGuid(reader.GetModuleDefinition().Mvid);
            PeTimeDateStamp = headers.CoffHeader.TimeDateStamp;
            PeCheckSum = headers.PEHeader?.CheckSum ?? 0u;
            CertificateTableSize = headers.PEHeader?.CertificateTableDirectory.Size ?? 0;
            StrongNameSignatureSize = headers.CorHeader?.StrongNameSignatureDirectory.Size ?? 0;
            IsIlOnly = headers.CorHeader is not null
                       && (headers.CorHeader.Flags & CorFlags.ILOnly) == CorFlags.ILOnly;
            HasPrecompiledNativeCode = headers.CorHeader is not null
                                       && headers.CorHeader.ManagedNativeHeaderDirectory.Size != 0;

            MetadataHeapSizes = "#Strings=" + reader.GetHeapSize(HeapIndex.String)
                                + " #US=" + reader.GetHeapSize(HeapIndex.UserString)
                                + " #Blob=" + reader.GetHeapSize(HeapIndex.Blob)
                                + " #GUID=" + reader.GetHeapSize(HeapIndex.Guid)
                                + " total=" + reader.MetadataLength;

            var definition = reader.GetAssemblyDefinition();

            AssemblyName = reader.GetString(definition.Name);
            AssemblyVersion = definition.Version;
            PublicKey = definition.PublicKey.IsNil
                ? string.Empty
                : Convert.ToHexString(reader.GetBlobBytes(definition.PublicKey));
            InformationalVersion = ReadStringAttribute(
                reader, definition.GetCustomAttributes(), "AssemblyInformationalVersionAttribute");

            var references = new List<string>();

            // BOUND: one iteration per row of the AssemblyRef table.
            foreach (var handle in reader.AssemblyReferences)
            {
                var reference = reader.GetAssemblyReference(handle);

                references.Add(reader.GetString(reference.Name) + "/" + reference.Version);
            }

            references.Sort(StringComparer.Ordinal);
            ReferencedAssemblies = references;

            var forwarded = new HashSet<string>(StringComparer.Ordinal);

            // BOUND: one iteration per row of the ExportedType table.
            foreach (var handle in reader.ExportedTypes)
            {
                var exported = reader.GetExportedType(handle);

                if (!exported.IsForwarder)
                {
                    continue;
                }

                var space = reader.GetString(exported.Namespace);
                var simple = reader.GetString(exported.Name);

                forwarded.Add(space.Length == 0 ? simple : space + "." + simple);
            }

            ForwardedTypes = forwarded;

            var debug = new StringBuilder();

            // BOUND: one iteration per debug directory entry, a short fixed list written by the compiler.
            foreach (var entry in _peReader.ReadDebugDirectory())
            {
                debug.Append(entry.Type);

                if (entry.Type == DebugDirectoryEntryType.PdbChecksum)
                {
                    var checksum = _peReader.ReadPdbChecksumDebugDirectoryData(entry);

                    debug.Append('(').Append(checksum.AlgorithmName).Append(':')
                        .Append(Convert.ToHexString(checksum.Checksum.AsSpan())).Append(')');
                }

                if (entry.Type == DebugDirectoryEntryType.CodeView)
                {
                    var codeView = _peReader.ReadCodeViewDebugDirectoryData(entry);

                    debug.Append('(').Append(codeView.Guid).Append(':').Append(codeView.Path).Append(')');
                }

                debug.Append(' ');
            }

            DebugDirectory = debug.ToString().Trim();
        }

        private void ReadMethodFacts()
        {
            var reader = _names.Reader;
            var bodies = new Dictionary<string, string>(StringComparer.Ordinal);

            // BOUND: one iteration per row of the MethodDef table.
            foreach (var handle in reader.MethodDefinitions)
            {
                var definition = reader.GetMethodDefinition(handle);
                var key = _names.MethodKey(handle);

                // Two rows sharing a key would silently overwrite one another and make a real difference
                // disappear, so the collision is made visible in the key rather than resolved by luck.
                // BOUND: at most as many passes as rows already carrying this key.
                var disambiguator = 1;

                while (bodies.ContainsKey(key))
                {
                    key = _names.MethodKey(handle) + " #"
                          + disambiguator.ToString(CultureInfo.InvariantCulture);
                    disambiguator++;
                }

                if (definition.RelativeVirtualAddress == 0)
                {
                    MethodsWithoutManagedBody++;
                    bodies[key] = "<no managed body>";
                    _methodHandles[key] = handle;

                    continue;
                }

                TotalIlByteCount += _peReader.GetMethodBody(definition.RelativeVirtualAddress)
                    .GetILBytes().Length;

                var text = IlNormalizer.Normalize(_peReader, _names, handle);

                bodies[key] = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text)));
                _methodHandles[key] = handle;
            }

            MethodIl = bodies;
        }

        private void ReadPublicApi()
        {
            var reader = _names.Reader;
            var members = new List<ApiMember>();

            // BOUND: one iteration per row of the TypeDef table.
            foreach (var handle in reader.TypeDefinitions)
            {
                var definition = reader.GetTypeDefinition(handle);

                if (!IsVisible(reader, definition))
                {
                    continue;
                }

                var typeName = _names.Provider.TypeName(handle);
                var isInterface = (definition.Attributes & TypeAttributes.Interface) != 0;

                members.Add(DescribeType(reader, definition, typeName, isInterface));

                var accessors = new HashSet<int>();

                // BOUND: one iteration per property of this type.
                foreach (var propertyHandle in definition.GetProperties())
                {
                    var property = reader.GetPropertyDefinition(propertyHandle);
                    var pair = property.GetAccessors();

                    AddAccessor(accessors, pair.Getter);
                    AddAccessor(accessors, pair.Setter);

                    var member = DescribeProperty(reader, property, typeName, pair, isInterface);

                    if (member is not null)
                    {
                        members.Add(member);
                    }
                }

                // BOUND: one iteration per event of this type.
                foreach (var eventHandle in definition.GetEvents())
                {
                    var declaration = reader.GetEventDefinition(eventHandle);
                    var pair = declaration.GetAccessors();

                    AddAccessor(accessors, pair.Adder);
                    AddAccessor(accessors, pair.Remover);
                    AddAccessor(accessors, pair.Raiser);

                    var member = DescribeEvent(reader, declaration, typeName, pair, isInterface);

                    if (member is not null)
                    {
                        members.Add(member);
                    }
                }

                // BOUND: one iteration per method of this type.
                foreach (var methodHandle in definition.GetMethods())
                {
                    if (accessors.Contains(MetadataTokens.GetToken(methodHandle)))
                    {
                        continue;
                    }

                    var member = DescribeMethod(reader, methodHandle, typeName, isInterface);

                    if (member is not null)
                    {
                        members.Add(member);
                    }
                }

                // BOUND: one iteration per field of this type.
                foreach (var fieldHandle in definition.GetFields())
                {
                    var member = DescribeField(reader, fieldHandle, typeName);

                    if (member is not null)
                    {
                        members.Add(member);
                    }
                }
            }

            members.Sort((left, right) => string.CompareOrdinal(left.Descriptor, right.Descriptor));
            PublicApi = members;
        }

        private ApiMember DescribeType(
            MetadataReader reader,
            TypeDefinition definition,
            string typeName,
            bool isInterface)
        {
            var interfaces = new List<string>();

            // BOUND: one iteration per interface implementation row for this type.
            foreach (var handle in definition.GetInterfaceImplementations())
            {
                interfaces.Add(_names.EntityName(reader.GetInterfaceImplementation(handle).Interface));
            }

            interfaces.Sort(StringComparer.Ordinal);

            var baseType = definition.BaseType.IsNil ? "<none>" : _names.EntityName(definition.BaseType);
            var isValueType = baseType.EndsWith("System.ValueType", StringComparison.Ordinal)
                              || baseType.EndsWith("System.Enum", StringComparison.Ordinal);
            var isEnum = baseType.EndsWith("System.Enum", StringComparison.Ordinal);
            var hasAccessibleConstructor = false;
            var underlying = string.Empty;
            var instanceFields = 0;
            var nonPublicInstanceFields = 0;

            // BOUND: one iteration per method of this type.
            foreach (var handle in definition.GetMethods())
            {
                var method = reader.GetMethodDefinition(handle);

                if (reader.GetString(method.Name) != ".ctor")
                {
                    continue;
                }

                var access = method.Attributes & MethodAttributes.MemberAccessMask;

                if (access == MethodAttributes.Public || access == MethodAttributes.Family
                    || access == MethodAttributes.FamORAssem)
                {
                    hasAccessibleConstructor = true;
                }
            }

            // Every field, not only the visible ones: the struct rule turns on whether ANY non-public instance
            // field already exists, and the enum's underlying type lives in a private-looking `value__` field.
            // BOUND: one iteration per field of this type.
            foreach (var handle in definition.GetFields())
            {
                var field = reader.GetFieldDefinition(handle);

                if ((field.Attributes & FieldAttributes.Static) != 0)
                {
                    continue;
                }

                instanceFields++;

                if ((field.Attributes & FieldAttributes.FieldAccessMask) != FieldAttributes.Public)
                {
                    nonPublicInstanceFields++;
                }

                if (isEnum && reader.GetString(field.Name) == "value__")
                {
                    underlying = field.DecodeSignature(_names.Provider, null);
                }
            }

            var typeParameters = new List<string>();

            // BOUND: one iteration per generic parameter of this type.
            foreach (var handle in definition.GetGenericParameters())
            {
                typeParameters.Add(reader.GetString(reader.GetGenericParameter(handle).Name));
            }

            return new ApiMember
            {
                DeclaringType = typeName,
                Kind = ApiMemberKind.Type,
                Name = reader.GetString(definition.Name),
                GenericArity = typeParameters.Count,
                Signature = "base " + baseType,
                ParameterNames = string.Join(", ", typeParameters),
                Accessibility = (definition.Attributes & TypeAttributes.VisibilityMask).ToString(),
                IsAbstract = (definition.Attributes & TypeAttributes.Abstract) != 0,
                IsSealed = (definition.Attributes & TypeAttributes.Sealed) != 0,
                IsInterface = isInterface,
                IsValueType = isValueType,
                IsEnum = isEnum,
                BaseType = baseType,
                Interfaces = string.Join(", ", interfaces),
                HasAccessibleConstructor = hasAccessibleConstructor,
                EnumUnderlyingType = underlying,
                InstanceFieldCount = instanceFields,
                NonPublicInstanceFieldCount = nonPublicInstanceFields,
                Constraints = DescribeConstraints(reader, definition.GetGenericParameters()),
            };
        }

        private ApiMember DescribeMethod(
            MetadataReader reader,
            MethodDefinitionHandle handle,
            string typeName,
            bool isInterface)
        {
            var definition = reader.GetMethodDefinition(handle);
            var access = definition.Attributes & MethodAttributes.MemberAccessMask;

            if (!IsVisibleAccess(access))
            {
                return null;
            }

            var signature = definition.DecodeSignature(_names.Provider, null);
            var names = new List<string>();
            var defaults = new List<string>();
            var modifiers = new List<string>();

            // BOUND: one iteration per Param row of this method.
            foreach (var parameterHandle in definition.GetParameters())
            {
                var parameter = reader.GetParameter(parameterHandle);

                if (parameter.SequenceNumber == 0)
                {
                    continue;
                }

                var name = reader.GetString(parameter.Name);
                var index = parameter.SequenceNumber - 1;
                var byReference = index < signature.ParameterTypes.Length
                                  && signature.ParameterTypes[index].EndsWith("&", StringComparison.Ordinal);

                names.Add(name);

                // Positional, WITHOUT the parameter name. Embedding the name here made a pure rename report
                // as both a rename (level 4) and a modifier change (level 5) — one edit, two findings, the
                // more severe of them false.
                modifiers.Add(ParameterModifier(reader, parameter, byReference));

                var constant = ReadConstant(reader, parameter.GetDefaultValue());

                if (constant is not null)
                {
                    defaults.Add(name + " = " + constant);
                }
            }

            var isAbstract = (definition.Attributes & MethodAttributes.Abstract) != 0;

            return new ApiMember
            {
                DeclaringType = typeName,
                Kind = ApiMemberKind.Method,
                Name = reader.GetString(definition.Name),
                GenericArity = signature.GenericParameterCount,
                ParameterCount = signature.ParameterTypes.Length,
                Signature = "(" + string.Join(", ", signature.ParameterTypes) + ") : " + signature.ReturnType,
                ParameterNames = string.Join(", ", names),
                ParameterModifiers = string.Join(", ", modifiers),
                Defaults = string.Join(", ", defaults),
                Accessibility = access.ToString(),
                IsStatic = (definition.Attributes & MethodAttributes.Static) != 0,
                IsAbstract = isAbstract,
                IsVirtual = (definition.Attributes & MethodAttributes.Virtual) != 0,
                IsFinal = (definition.Attributes & MethodAttributes.Final) != 0,

                // An interface member with a body. This one flag decides whether adding a member to an
                // interface is a binary break or an allowed addition.
                HasDefaultImplementation = isInterface && !isAbstract
                                           && (definition.Attributes & MethodAttributes.Static) == 0,
                Constraints = DescribeConstraints(reader, definition.GetGenericParameters()),
            };
        }

        private ApiMember DescribeField(
            MetadataReader reader,
            FieldDefinitionHandle handle,
            string typeName)
        {
            var definition = reader.GetFieldDefinition(handle);
            var access = definition.Attributes & FieldAttributes.FieldAccessMask;

            if (access != FieldAttributes.Public && access != FieldAttributes.Family
                && access != FieldAttributes.FamORAssem)
            {
                return null;
            }

            // The enum's backing field is an implementation detail of the enum, reported through the type's
            // EnumUnderlyingType instead. Listing it as a member would make every enum carry a phantom field.
            if (reader.GetString(definition.Name) == "value__")
            {
                return null;
            }

            return new ApiMember
            {
                DeclaringType = typeName,
                Kind = ApiMemberKind.Field,
                Name = reader.GetString(definition.Name),
                Signature = definition.DecodeSignature(_names.Provider, null),
                Accessibility = access.ToString(),
                IsStatic = (definition.Attributes & FieldAttributes.Static) != 0,
                IsInitOnly = (definition.Attributes & FieldAttributes.InitOnly) != 0,
                IsLiteral = (definition.Attributes & FieldAttributes.Literal) != 0,
                ConstantValue = ReadConstant(reader, definition.GetDefaultValue()),
            };
        }

        private ApiMember DescribeProperty(
            MetadataReader reader,
            PropertyDefinition definition,
            string typeName,
            PropertyAccessors accessors,
            bool isInterface)
        {
            var shape = AccessorShape(reader, accessors.Getter, accessors.Setter, default);

            if (shape is null)
            {
                return null;
            }

            var signature = definition.DecodeSignature(_names.Provider, null);

            return new ApiMember
            {
                DeclaringType = typeName,
                Kind = ApiMemberKind.Property,
                Name = reader.GetString(definition.Name),
                ParameterCount = signature.ParameterTypes.Length,
                Signature = "(" + string.Join(", ", signature.ParameterTypes) + ") : " + signature.ReturnType,
                ParameterNames = shape.Accessors,
                Accessibility = shape.Accessibility,
                IsStatic = shape.IsStatic,
                IsAbstract = shape.IsAbstract,
                IsVirtual = shape.IsVirtual,
                IsFinal = shape.IsFinal,
                HasDefaultImplementation = isInterface && !shape.IsAbstract && !shape.IsStatic,
            };
        }

        private ApiMember DescribeEvent(
            MetadataReader reader,
            EventDefinition definition,
            string typeName,
            EventAccessors accessors,
            bool isInterface)
        {
            var shape = AccessorShape(reader, accessors.Adder, accessors.Remover, accessors.Raiser);

            if (shape is null)
            {
                return null;
            }

            return new ApiMember
            {
                DeclaringType = typeName,
                Kind = ApiMemberKind.Event,
                Name = reader.GetString(definition.Name),
                Signature = _names.EntityName(definition.Type),
                ParameterNames = shape.Accessors,
                Accessibility = shape.Accessibility,
                IsStatic = shape.IsStatic,
                IsAbstract = shape.IsAbstract,
                IsVirtual = shape.IsVirtual,
                IsFinal = shape.IsFinal,
                HasDefaultImplementation = isInterface && !shape.IsAbstract && !shape.IsStatic,
            };
        }

        /// <summary>
        /// The visibility and virtuality of a property or event, taken from its accessors, or
        /// <see langword="null"/> when no accessor is visible.
        ///
        /// <para>Visibility of a property IS the visibility of its accessors: a property whose only accessor is
        /// private is not part of the surface, and reporting it would make a purely internal change look like an
        /// API change.</para>
        /// </summary>
        private static AccessorSummary AccessorShape(
            MetadataReader reader,
            MethodDefinitionHandle first,
            MethodDefinitionHandle second,
            MethodDefinitionHandle third)
        {
            var summary = new AccessorSummary();
            var parts = new List<string>();

            // BOUND: exactly three accessor slots.
            foreach (var handle in new[] { first, second, third })
            {
                if (handle.IsNil)
                {
                    continue;
                }

                var definition = reader.GetMethodDefinition(handle);
                var access = definition.Attributes & MethodAttributes.MemberAccessMask;

                if (!IsVisibleAccess(access))
                {
                    continue;
                }

                parts.Add(reader.GetString(definition.Name));

                // The most permissive visible accessor sets the member's accessibility, matching how a
                // consumer experiences it: a public getter with a protected setter is a public property.
                if (summary.Accessibility.Length == 0 || access == MethodAttributes.Public)
                {
                    summary.Accessibility = access.ToString();
                }

                summary.IsStatic |= (definition.Attributes & MethodAttributes.Static) != 0;
                summary.IsAbstract |= (definition.Attributes & MethodAttributes.Abstract) != 0;
                summary.IsVirtual |= (definition.Attributes & MethodAttributes.Virtual) != 0;
                summary.IsFinal |= (definition.Attributes & MethodAttributes.Final) != 0;
            }

            if (parts.Count == 0)
            {
                return null;
            }

            summary.Accessors = string.Join(" ", parts);

            return summary;
        }

        private string DescribeConstraints(
            MetadataReader reader,
            GenericParameterHandleCollection parameters)
        {
            var all = new List<string>();

            // BOUND: one iteration per generic parameter.
            foreach (var handle in parameters)
            {
                var parameter = reader.GetGenericParameter(handle);
                var parts = new List<string>();
                var attributes = parameter.Attributes;

                if ((attributes & GenericParameterAttributes.ReferenceTypeConstraint) != 0)
                {
                    parts.Add("class");
                }

                if ((attributes & GenericParameterAttributes.NotNullableValueTypeConstraint) != 0)
                {
                    parts.Add("struct");
                }

                if ((attributes & GenericParameterAttributes.DefaultConstructorConstraint) != 0)
                {
                    parts.Add("new()");
                }

                // BOUND: one iteration per constraint row on this parameter.
                foreach (var constraintHandle in parameter.GetConstraints())
                {
                    parts.Add(_names.EntityName(reader.GetGenericParameterConstraint(constraintHandle).Type));
                }

                if (parts.Count == 0)
                {
                    continue;
                }

                parts.Sort(StringComparer.Ordinal);
                all.Add(reader.GetString(parameter.Name) + ": " + string.Join(" + ", parts));
            }

            return string.Join("; ", all);
        }

        private static string ParameterModifier(MetadataReader reader, Parameter parameter, bool byReference)
        {
            var attributes = parameter.Attributes;
            var modifier = "value";

            if (byReference)
            {
                modifier = (attributes & ParameterAttributes.Out) != 0 ? "out"
                    : (attributes & ParameterAttributes.In) != 0 ? "in" : "ref";
            }

            // `optional` is deliberately NOT recorded here: the presence and value of a default is already
            // carried by Defaults, and recording it twice would make adding a default fire the
            // modifier-changed rule at level 5 as well as the correct level-3 rule.
            // BOUND: one iteration per custom attribute on this parameter.
            foreach (var handle in parameter.GetCustomAttributes())
            {
                if (AttributeTypeMatches(reader, reader.GetCustomAttribute(handle), "ParamArrayAttribute"))
                {
                    modifier += "+params";
                }
            }

            return modifier;
        }

        private static bool IsVisibleAccess(MethodAttributes access)
        {
            return access == MethodAttributes.Public
                   || access == MethodAttributes.Family
                   || access == MethodAttributes.FamORAssem;
        }

        private static void AddAccessor(HashSet<int> accessors, MethodDefinitionHandle handle)
        {
            if (!handle.IsNil)
            {
                accessors.Add(MetadataTokens.GetToken(handle));
            }
        }

        private static bool IsVisible(MetadataReader reader, TypeDefinition definition)
        {
            var visibility = definition.Attributes & TypeAttributes.VisibilityMask;

            if (visibility == TypeAttributes.Public)
            {
                return true;
            }

            if (visibility != TypeAttributes.NestedPublic
                && visibility != TypeAttributes.NestedFamily
                && visibility != TypeAttributes.NestedFamORAssem)
            {
                return false;
            }

            // A nested public type inside an internal one is not reachable from outside. BOUND: the walk
            // follows the declaring-type chain, which ECMA-335 requires to be acyclic.
            return IsVisible(reader, reader.GetTypeDefinition(definition.GetDeclaringType()));
        }

        private static string ReadStringAttribute(
            MetadataReader reader,
            CustomAttributeHandleCollection attributes,
            string simpleName)
        {
            // BOUND: one iteration per custom attribute on the target.
            foreach (var handle in attributes)
            {
                var attribute = reader.GetCustomAttribute(handle);

                if (!AttributeTypeMatches(reader, attribute, simpleName))
                {
                    continue;
                }

                var value = attribute.DecodeValue(new SignatureTypeNameProvider(reader));

                if (value.FixedArguments.Length > 0 && value.FixedArguments[0].Value is string text)
                {
                    return text;
                }
            }

            return null;
        }

        private static bool AttributeTypeMatches(
            MetadataReader reader,
            CustomAttribute attribute,
            string simpleName)
        {
            if (attribute.Constructor.Kind != HandleKind.MemberReference)
            {
                return false;
            }

            var parent = reader.GetMemberReference((MemberReferenceHandle)attribute.Constructor).Parent;

            if (parent.Kind != HandleKind.TypeReference)
            {
                return false;
            }

            return reader.GetString(reader.GetTypeReference((TypeReferenceHandle)parent).Name) == simpleName;
        }

        private static string ReadConstant(MetadataReader reader, ConstantHandle handle)
        {
            if (handle.IsNil)
            {
                return null;
            }

            var constant = reader.GetConstant(handle);
            var blob = reader.GetBlobReader(constant.Value);

            return constant.TypeCode switch
            {
                ConstantTypeCode.Boolean => blob.ReadBoolean().ToString(),
                ConstantTypeCode.Char => ((int)blob.ReadChar()).ToString(CultureInfo.InvariantCulture),
                ConstantTypeCode.SByte => blob.ReadSByte().ToString(CultureInfo.InvariantCulture),
                ConstantTypeCode.Byte => blob.ReadByte().ToString(CultureInfo.InvariantCulture),
                ConstantTypeCode.Int16 => blob.ReadInt16().ToString(CultureInfo.InvariantCulture),
                ConstantTypeCode.UInt16 => blob.ReadUInt16().ToString(CultureInfo.InvariantCulture),
                ConstantTypeCode.Int32 => blob.ReadInt32().ToString(CultureInfo.InvariantCulture),
                ConstantTypeCode.UInt32 => blob.ReadUInt32().ToString(CultureInfo.InvariantCulture),
                ConstantTypeCode.Int64 => blob.ReadInt64().ToString(CultureInfo.InvariantCulture),
                ConstantTypeCode.UInt64 => blob.ReadUInt64().ToString(CultureInfo.InvariantCulture),
                ConstantTypeCode.Single => blob.ReadSingle().ToString("R", CultureInfo.InvariantCulture),
                ConstantTypeCode.Double => blob.ReadDouble().ToString("R", CultureInfo.InvariantCulture),
                ConstantTypeCode.String => "\"" + blob.ReadUTF16(blob.RemainingBytes) + "\"",
                ConstantTypeCode.NullReference => "null",
                _ => constant.TypeCode.ToString(),
            };
        }

        /// <summary>The accessor-derived shape of a property or event.</summary>
        private sealed class AccessorSummary
        {
            internal string Accessibility { get; set; } = string.Empty;

            internal string Accessors { get; set; } = string.Empty;

            internal bool IsStatic
            {
                get; set;
            }

            internal bool IsAbstract
            {
                get; set;
            }

            internal bool IsVirtual
            {
                get; set;
            }

            internal bool IsFinal
            {
                get; set;
            }
        }
    }
}
