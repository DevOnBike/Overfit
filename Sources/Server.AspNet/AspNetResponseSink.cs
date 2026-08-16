// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Text;
using DevOnBike.Overfit.Server.OpenAi;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;

namespace DevOnBike.Overfit.Server.AspNet
{
    /// <summary>
    /// Adapts an ASP.NET <see cref="HttpResponse"/> to <see cref="IOpenAiResponseSink"/> so the Kestrel host
    /// drives the shared <see cref="ChatCompletionExchange"/> unchanged. Writes are synchronous — the
    /// exchange streams tokens from a synchronous generate callback — so synchronous body IO is enabled on
    /// entering the stream; Kestrel disallows it by default.
    /// </summary>
    // OVERFIT040 for this class's two writing methods, at the type because they share one constraint.
    //
    // THE CONSTRAINT IS THE CALLBACK ABOVE THEM: these implement `IOpenAiResponseSink`, whose members return
    // void, and the caller is `ChatCompletionExchange`, which writes each token from inside the decode
    // callback the model invokes per token. There is no `await` point available inside a synchronous callback,
    // so an asynchronous sink would have to be blocked on at the call site — which is OVERFIT039, the worse
    // shape of the same problem. `EndpointHelpers.EnableSynchronousIO` exists exactly to let this work, and
    // the interface is public API in the Server assembly rather than this file's to change.
    //
    // WHAT IS GIVEN UP: the request thread that entered the endpoint is held for the whole generation. That is
    // the server's design — one thread per in-flight completion, for its duration — and it is why the session
    // pool bounds concurrency instead of the thread pool doing it.
#pragma warning disable OVERFIT040
    internal sealed class AspNetResponseSink : IOpenAiResponseSink
    {
        private readonly HttpResponse _response;

        public AspNetResponseSink(HttpResponse response) => _response = response;

        public void WriteBody(int statusCode, string contentType, string body)
            => WriteBinary(statusCode, contentType, Encoding.UTF8.GetBytes(body));

        public void WriteBinary(int statusCode, string contentType, byte[] body)
        {
            _response.StatusCode = statusCode;
            _response.ContentType = contentType;
            _response.ContentLength = body.Length;
            _response.Body.Write(body, 0, body.Length);
        }

        public void BeginEventStream()
        {
            // Synchronous body IO (the exchange writes each token from the decode callback) is enabled by the
            // endpoint before the exchange runs; here we only set the SSE headers.
            _response.StatusCode = StatusCodes.Status200OK;
            _response.ContentType = "text/event-stream";
            _response.Headers.CacheControl = "no-cache";
        }

        public void WriteEvent(string data)
        {
            var bytes = Encoding.UTF8.GetBytes($"data: {data}\n\n");
            _response.Body.Write(bytes, 0, bytes.Length);
            _response.Body.Flush();
        }
    }
#pragma warning restore OVERFIT040
}
