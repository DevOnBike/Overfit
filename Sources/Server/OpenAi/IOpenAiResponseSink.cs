// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Server.OpenAi
{
    /// <summary>
    /// The transport-neutral surface the shared chat handler (<see cref="ChatCompletionExchange"/>) writes
    /// through, so the OpenAI wire protocol — request validation, the streaming SSE shape, finish-reason
    /// logic, the response objects — lives once and both hosts (the Native-AOT <c>HttpListener</c> CLI server
    /// and the ASP.NET Minimal-API host) supply only a thin adapter over their own response object.
    ///
    /// <para>Three primitives cover every write the protocol needs: a complete-body response for errors and
    /// non-streaming results, the switch into event-stream mode, and one already-serialized SSE frame. The
    /// adapter owns nothing but byte output; all serialization stays in the shared handler on the source-gen
    /// <c>OpenAiJsonContext</c>, which keeps the whole path allocation-lean and trim/AOT-safe.</para>
    /// </summary>
    public interface IOpenAiResponseSink
    {
        /// <summary>
        /// Sends a complete-body response (a validation error, or the non-streaming chat result) and finishes
        /// the exchange. Called at most once, and never after <see cref="BeginEventStream"/>.
        /// </summary>
        void WriteBody(int statusCode, string contentType, string body);

        /// <summary>
        /// Sends a complete binary-body response (synthesized audio) and finishes the exchange. Same one-shot
        /// contract as <see cref="WriteBody"/>.
        /// </summary>
        void WriteBinary(int statusCode, string contentType, byte[] body);

        /// <summary>
        /// Switches the response into Server-Sent-Events mode: status 200, <c>text/event-stream</c>,
        /// <c>Cache-Control: no-cache</c>, chunked transfer. Any transport-specific streaming setup (e.g.
        /// enabling synchronous body writes) belongs here.
        /// </summary>
        void BeginEventStream();

        /// <summary>
        /// Writes one SSE frame: the sink adds the <c>data: …\n\n</c> framing and flushes so the client sees
        /// the token immediately. <paramref name="data"/> is the already-serialized chunk JSON, or the literal
        /// <c>[DONE]</c> sentinel.
        /// </summary>
        void WriteEvent(string data);
    }
}
