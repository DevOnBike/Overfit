// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Server;

namespace DevOnBike.Overfit.Tests.Redaction
{
    /// <summary>
    /// Gateway client authentication: when keys are configured, callers must present a valid bearer token; with no
    /// keys the gateway is open (back-compat). Validates the bearer parsing and accept/reject decisions.
    /// </summary>
    public sealed class GatewayClientAuthTests
    {
        [Fact]
        public void NoKeys_AuthDisabled_AllowsEveryone()
        {
            var auth = new GatewayClientAuth(null);
            Assert.False(auth.Enabled);
            Assert.True(auth.IsAuthorized(null));
            Assert.True(auth.IsAuthorized("Bearer whatever"));
        }

        [Fact]
        public void WithKeys_AcceptsValidBearer_RejectsInvalidAndMissing()
        {
            var auth = new GatewayClientAuth(new[] { "sk-client-a", "sk-client-b" });
            Assert.True(auth.Enabled);

            Assert.True(auth.IsAuthorized("Bearer sk-client-a"));
            Assert.True(auth.IsAuthorized("Bearer sk-client-b"));

            Assert.False(auth.IsAuthorized("Bearer sk-wrong"));
            Assert.False(auth.IsAuthorized(null));
            Assert.False(auth.IsAuthorized(""));
            Assert.False(auth.IsAuthorized("Bearer "));
        }

        [Fact]
        public void BearerPrefix_IsOptionalAndCaseInsensitive()
        {
            var auth = new GatewayClientAuth(new[] { "sk-client-a" });

            // Raw token without the scheme.
            Assert.True(auth.IsAuthorized("sk-client-a"));
            // Case-insensitive scheme.
            Assert.True(auth.IsAuthorized("bearer sk-client-a"));
        }

        [Fact]
        public void BlankConfiguredKeys_AreIgnored()
        {
            // A config with only whitespace keys leaves auth disabled rather than locking everyone out by accident.
            var auth = new GatewayClientAuth(new[] { "  ", "" });
            Assert.False(auth.Enabled);
            Assert.True(auth.IsAuthorized(null));
        }
    }
}
