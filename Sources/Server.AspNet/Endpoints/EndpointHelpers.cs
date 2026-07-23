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
            if (bodyControl is not null)
            {
                bodyControl.AllowSynchronousIO = true;
            }
        }

        public static void WriteError(HttpResponse response, int status, string message)
        {
            var body = new OpenAiErrorResponse { Error = new OpenAiError { Message = message } };
            var json = JsonSerializer.Serialize(body, OpenAiJsonContext.Default.OpenAiErrorResponse);
            var bytes = Encoding.UTF8.GetBytes(json);
            response.StatusCode = status;
            response.ContentType = "application/json";
            response.ContentLength = bytes.Length;
            response.Body.WriteAsync(bytes).AsTask().GetAwaiter().GetResult();
        }
    }
}
