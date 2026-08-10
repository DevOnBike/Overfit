// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Anomalies.Hosting;

namespace DevOnBike.Overfit.Tests.Anomalies
{
    /// <summary>
    /// `POST /ack` suppresses a finding for a caller-chosen duration, so the question of who may call it is
    /// the whole of `SEC-1`.
    ///
    /// <para><b>These tests exist because the endpoint itself cannot be tested.</b> It lives in
    /// `Sources/Cli`, which the test project does not reference and which exposes no internals — so until
    /// 2026-08-10 the code that decides whether a stranger may silence an incident had no coverage at all and
    /// no way to get any. The decision was extracted here for that reason; the HTTP plumbing stayed
    /// behind.</para>
    /// </summary>
    public sealed class GuardAckAuthorizationTests
    {
        private const string Token = "s3cr3t-token-value";

        /// <summary>
        /// The whole point. An unset secret must refuse, not wave through — the `NetworkPolicy` written to
        /// protect this port was measured inert on a Docker-Desktop-class cluster, so failing open would
        /// leave a customer whose CNI ignores policy with no protection whatsoever.
        /// </summary>
        [Theory]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("   ")]
        public void WithNoTokenConfigured_EverythingIsRefused(string? configured)
        {
            Assert.False(GuardAckAuthorization.IsConfigured(configured));

            // Including a caller who presents something that looks plausible, and one who presents nothing.
            Assert.False(GuardAckAuthorization.IsAuthorised($"Bearer {Token}", configured));
            Assert.False(GuardAckAuthorization.IsAuthorised(null, configured));
            Assert.False(GuardAckAuthorization.IsAuthorised("Bearer ", configured));
        }

        [Fact]
        public void TheConfiguredTokenIsAccepted()
        {
            Assert.True(GuardAckAuthorization.IsConfigured(Token));
            Assert.True(GuardAckAuthorization.IsAuthorised($"Bearer {Token}", Token));
        }

        /// <summary>
        /// The scheme is matched case-insensitively because clients differ, but the SECRET is not — a
        /// case-insensitive secret comparison silently divides the key space.
        /// </summary>
        [Fact]
        public void TheSchemeIsCaseInsensitiveAndTheSecretIsNot()
        {
            Assert.True(GuardAckAuthorization.IsAuthorised($"bearer {Token}", Token));
            Assert.True(GuardAckAuthorization.IsAuthorised($"BEARER {Token}", Token));

            Assert.False(GuardAckAuthorization.IsAuthorised($"Bearer {Token.ToUpperInvariant()}", Token));
        }

        [Theory]
        [InlineData("")]
        [InlineData("Bearer")]
        [InlineData("Bearer wrong")]
        [InlineData("Basic s3cr3t-token-value")]
        [InlineData("s3cr3t-token-value")]
        [InlineData("Bearer s3cr3t-token-valu")]
        [InlineData("Bearer s3cr3t-token-value-extra")]
        public void EverythingElseIsRefused(string header)
        {
            Assert.False(GuardAckAuthorization.IsAuthorised(header, Token));
        }

        /// <summary>
        /// A prefix of the real token must not be accepted, and neither must the real token with anything
        /// appended. Stated as its own case because a naive `StartsWith` comparison passes both and looks
        /// correct in every other test above.
        /// </summary>
        [Fact]
        public void APrefixOrAnExtensionOfTheTokenIsNotTheToken()
        {
            for (var length = 1; length < Token.Length; length++)
            {
                Assert.False(GuardAckAuthorization.IsAuthorised($"Bearer {Token[..length]}", Token));
            }

            Assert.False(GuardAckAuthorization.IsAuthorised($"Bearer {Token}x", Token));
        }

        /// <summary>
        /// Surrounding whitespace is tolerated on both sides, because a token pasted from a Secret or typed
        /// into a shell arrives with it and refusing would send an operator hunting for a mistake that is not
        /// theirs. Interior whitespace is part of the secret and is not touched.
        /// </summary>
        [Fact]
        public void SurroundingWhitespaceIsToleratedOnBothSides()
        {
            Assert.True(GuardAckAuthorization.IsAuthorised($"Bearer  {Token}  ", $"  {Token}\n"));
            Assert.False(GuardAckAuthorization.IsAuthorised($"Bearer {Token} extra", Token));
        }
    }
}
