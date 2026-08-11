// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using System.Text.Json;
using DevOnBike.Overfit.Server.OpenAi;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;

namespace DevOnBike.Overfit.Server.AspNet.Endpoints
{
    /// <summary>
    /// Shared plumbing for the route-group endpoints: enabling synchronous body IO (the shared exchanges
    /// write synchronously, and Kestrel disallows it by default) and writing an OpenAI-shaped error before an
    /// exchange runs. Serialization stays on the source-gen <see cref="OpenAiJsonContext"/>, so the whole
    /// endpoint surface remains reflection-free for Native AOT.
    /// </summary>
    internal static class EndpointHelpers
    {
        public static void EnableSynchronousIO(HttpContext ctx)
        {
            var bodyControl = ctx.Features.Get<IHttpBodyControlFeature>();
            if (bodyControl != null)
            {
                bodyControl.AllowSynchronousIO = true;
            }
        }

        /// <summary>
        /// Writes an OpenAI-shaped error body.
        ///
        /// <para><b>Asynchronous, and it had to become so rather than merely stop blocking.</b> It used to
        /// end in <c>WriteAsync(bytes).AsTask().GetAwaiter().GetResult()</c> — blocking a request thread on
        /// a task (OVERFIT039). The obvious repair, a synchronous <c>Body.Write</c>, was wrong and a test
        /// caught it: <c>AllowSynchronousIO</c> is set by the redaction GATEWAY's middleware, not by the
        /// OpenAI server, and this helper is shared by both. On the server it threw "Synchronous operations
        /// are disallowed" from the 400 path — the error path, which is exactly where a second failure is
        /// least welcome. Awaiting it is correct for both hosts and blocks neither.</para>
        /// </summary>
        /// <param name="cancellationToken">
        /// Normally <c>ctx.RequestAborted</c>. Required by OVERFIT030 and useful rather than ceremonial: a
        /// caller that has already disconnected should not hold a request thread while its error body is
        /// written to a socket nobody is reading.
        /// </param>
        public static async Task WriteErrorAsync(HttpResponse response, int status, string message, CancellationToken cancellationToken)
        {
            var body = new OpenAiErrorResponse { Error = new OpenAiError { Message = message } };
            var json = JsonSerializer.Serialize(body, OpenAiJsonContext.Default.OpenAiErrorResponse);
            var bytes = Encoding.UTF8.GetBytes(json);
            
            response.StatusCode = status;
            response.ContentType = "application/json";
            response.ContentLength = bytes.Length;

            await response.Body.WriteAsync(bytes, cancellationToken);
        }
    }
}
