// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Net;
using System.Text;

namespace DevOnBike.Overfit.Server.OpenAi
{
    /// <summary>
    /// Adapts a raw <see cref="HttpListenerResponse"/> to <see cref="IOpenAiResponseSink"/> so the
    /// dependency-free <c>HttpListener</c> server drives the shared <see cref="ChatCompletionExchange"/>
    /// exactly as the ASP.NET host does. This is the only place the CLI server touches the wire for chat.
    /// </summary>
    internal sealed class HttpListenerResponseSink : IOpenAiResponseSink
    {
        private readonly HttpListenerResponse _response;

        public HttpListenerResponseSink(HttpListenerResponse response) => _response = response;

        public void WriteBody(int statusCode, string contentType, string body)
            => WriteBinary(statusCode, contentType, Encoding.UTF8.GetBytes(body));

        public void WriteBinary(int statusCode, string contentType, byte[] body)
        {
            _response.StatusCode = statusCode;
            _response.ContentType = contentType;
            _response.ContentLength64 = body.Length;
            _response.OutputStream.Write(body, 0, body.Length);
        }

        public void BeginEventStream()
        {
            _response.StatusCode = (int)HttpStatusCode.OK;
            _response.ContentType = "text/event-stream";
            _response.Headers["Cache-Control"] = "no-cache";
            _response.SendChunked = true;
        }

        public void WriteEvent(string data)
        {
            var bytes = Encoding.UTF8.GetBytes($"data: {data}\n\n");
            _response.OutputStream.Write(bytes, 0, bytes.Length);
            _response.OutputStream.Flush();
        }
    }
}
