// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;

namespace DevOnBike.Overfit.Server.OpenAi
{
    /// <summary>
    /// The self-describing API contract served at <c>GET /openapi.yaml</c> and the interactive API reference
    /// (Scalar) at <c>GET /docs</c>. The YAML is embedded (one source of truth = <c>docs/openapi.yaml</c>) so
    /// it ships inside the single self-contained binary; the viewer is a self-served HTML page that loads its
    /// assets from a CDN rather than bloating the binary — no server-side generator, so the whole docs path
    /// stays reflection-free and Native-AOT-clean. Host-agnostic so any host can serve both.
    /// </summary>
    public static class OpenApiDocument
    {
        private static string? _yaml;

        /// <summary>The embedded <c>openapi.yaml</c> contract, read once and cached.</summary>
        // OVERFIT040 on Yaml: `StreamReader.ReadToEnd` has a `ReadToEndAsync` sibling and there is nothing for
        // it to overlap with. BOUND BY WHAT THE STREAM IS: `Assembly.GetManifestResourceStream` returns an
        // UnmanagedMemoryStream over bytes already mapped in with the assembly image — the read is a memory
        // copy, with no file handle, no socket and no device to wait on, so there is no IO here to yield
        // during. The result is cached in `_yaml` on the first call, so even that copy happens once per
        // process; making this a task would add a state machine to a memcpy.
#pragma warning disable OVERFIT040
        public static string Yaml()
#pragma warning restore OVERFIT040
        {
            if (_yaml != null)
            {
                return _yaml;
            }

            using var stream = typeof(OpenApiDocument).Assembly.GetManifestResourceStream("openapi.yaml");
            if (stream == null)
            {
                _yaml = "openapi: 3.0.3\ninfo:\n  title: Overfit\n  version: '1.0.0'\npaths: {}\n";
                return _yaml;
            }

            using var reader = new StreamReader(stream, Encoding.UTF8);
            _yaml = reader.ReadToEnd();
            return _yaml;
        }

        /// <summary>
        /// Scalar API-reference viewer, pointed at <c>/openapi.yaml</c> via its standalone CDN bundle — the
        /// modern replacement for Swagger UI that the .NET templates moved to. Purely client-side: no
        /// server-side OpenAPI generator (which drags MVC.Abstractions and trips IL3053 under the AOT guard).
        /// </summary>
        public const string ApiReferenceHtml = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8" />
              <meta name="viewport" content="width=device-width, initial-scale=1" />
              <title>Overfit API — Reference</title>
            </head>
            <body>
              <script id="api-reference" data-url="/openapi.yaml"></script>
              <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
            </body>
            </html>
            """;
    }
}
