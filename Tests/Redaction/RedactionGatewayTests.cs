// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Redaction;
using DevOnBike.Overfit.Server;
using DevOnBike.Overfit.Server.OpenAi;

namespace DevOnBike.Overfit.Tests.Redaction
{
    /// <summary>
    /// The redaction gateway's pure transform: an outbound request is scrubbed of PII/secrets before it would be
    /// forwarded, and the originals are restored on the response coming back. (The HTTP forward itself is a thin
    /// wrapper exercised manually / in integration.)
    /// </summary>
    public sealed class RedactionGatewayTests
    {
        private static readonly Redactor _redactor = Redactor.CreateDefault();

        // Redact-everything policy: exercises the redaction transform itself (the BLOCK policy is tested separately).
        private static readonly RedactionPolicy _redactAll =
            new(new Dictionary<string, RedactionAction>(), RedactionAction.Redact);

        [Fact]
        public void RedactRequest_ScrubsOutboundMessages_AndReportsCounts()
        {
            var req = new ChatCompletionRequest
            {
                Model = "gpt-4",
                Messages =
                [
                    new OpenAiMessage { Role = "system", Content = "You are a helpful assistant." },
                    new OpenAiMessage { Role = "user", Content = "Email jane@acme.com and use key sk-ABCDEFGHIJKLMNOPQRSTUVWX." }
                ]
            };

            var (matches, counts, _, _) = RedactionGateway.RedactRequest(req, _redactor, _redactAll);

            // The outbound payload no longer contains the secrets.
            Assert.DoesNotContain("jane@acme.com", req.Messages[1].Content);
            Assert.DoesNotContain("sk-ABCDEFGHIJKLMNOPQRSTUVWX", req.Messages[1].Content);
            Assert.Contains("[REDACTED_EMAIL_0]", req.Messages[1].Content);

            // System message had nothing to redact — untouched.
            Assert.Equal("You are a helpful assistant.", req.Messages[0].Content);

            Assert.Equal(2, matches.Count);
            Assert.Equal(1, counts["EMAIL"]);
            Assert.Equal(1, counts["API_KEY"]);
        }

        [Fact]
        public void RestoreResponse_RehydratesPlaceholders_TheModelEchoedBack()
        {
            var req = new ChatCompletionRequest
            {
                Messages = [new OpenAiMessage { Role = "user", Content = "Confirm the address admin@corp.io please." }]
            };
            var (matches, _, _, _) = RedactionGateway.RedactRequest(req, _redactor, _redactAll);
            var placeholder = matches[0].Placeholder;

            // Upstream answered referring to the placeholder it was given.
            var response = new ChatCompletionResponse
            {
                Choices = [new ChatChoice { Index = 0, Message = new OpenAiMessage { Role = "assistant", Content = $"Yes, {placeholder} is confirmed." } }]
            };

            RedactionGateway.RestoreResponse(response, matches);

            // The caller sees the real value again — coherent answer, secret never reached the upstream.
            Assert.Equal("Yes, admin@corp.io is confirmed.", response.Choices[0].Message!.Content);
        }

        [Fact]
        public void RedactRequest_CleanPayload_NoRedactions()
        {
            var req = new ChatCompletionRequest
            {
                Messages = [new OpenAiMessage { Role = "user", Content = "What is the capital of France?" }]
            };

            var (matches, counts, blocked, _) = RedactionGateway.RedactRequest(req, _redactor, _redactAll);

            Assert.Empty(matches);
            Assert.Empty(counts);
            Assert.False(blocked);
            Assert.Equal("What is the capital of France?", req.Messages[0].Content);
        }

        [Fact]
        public void DefaultPolicy_BlocksSecrets_RedactsPii()
        {
            var policy = RedactionPolicy.Default();

            // A secret (API key) → BLOCK: the whole request is refused, nothing forwarded.
            var withSecret = new ChatCompletionRequest
            {
                Messages = [new OpenAiMessage { Role = "user", Content = "deploy with key sk-ABCDEFGHIJKLMNOPQRSTUVWX now" }]
            };
            var (_, _, blocked, blockedCategories) = RedactionGateway.RedactRequest(withSecret, _redactor, policy);
            Assert.True(blocked);
            Assert.Contains("API_KEY", blockedCategories);

            // PII only (email) → REDACT + forward, not blocked.
            var withPii = new ChatCompletionRequest
            {
                Messages = [new OpenAiMessage { Role = "user", Content = "summarize the thread from bob@corp.io" }]
            };
            var (matches, counts, blocked2, _) = RedactionGateway.RedactRequest(withPii, _redactor, policy);
            Assert.False(blocked2);
            Assert.Equal(1, counts["EMAIL"]);
            Assert.DoesNotContain("bob@corp.io", withPii.Messages[0].Content);
        }
    }
}
