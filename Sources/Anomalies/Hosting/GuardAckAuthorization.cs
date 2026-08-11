// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Security.Cryptography;
using System.Text;

namespace DevOnBike.Overfit.Anomalies.Hosting
{
    /// <summary>
    /// Whether a caller may acknowledge — and therefore silence — a finding.
    ///
    /// <para><b>A pure function, in this assembly, on purpose.</b> The endpoint that serves <c>POST /ack</c>
    /// lives in <c>Sources/Cli</c>, which the test project does not reference and which exposes no internals,
    /// so nothing about it could be tested. Leaving a security decision in that position is how it stays
    /// unexamined. The HTTP plumbing stays there; the decision lives here, where a test can reach it.</para>
    ///
    /// <para><b>Fail closed on an unset secret.</b> `/ack` suppresses a finding for a caller-chosen duration,
    /// so an unauthenticated one is worse than no endpoint at all: the guard keeps running, keeps looking
    /// healthy, and reports nothing. The `NetworkPolicy` written to protect that port was measured
    /// <b>inert</b> on a Docker-Desktop-class cluster — no policy-capable CNI, and a probe pod still reached
    /// the port with the policy applied — so a control that assumes the customer's CNI enforces policy is not
    /// a control.</para>
    ///
    /// <para><b>Constant-time comparison.</b> A short-circuiting compare over a secret leaks its prefix to
    /// anyone who can time the response, and this endpoint is reachable by whoever can reach the scrape
    /// port.</para>
    /// </summary>
    public static class GuardAckAuthorization
    {
        private const string Scheme = "Bearer ";

        /// <summary>
        /// Whether <paramref name="authorizationHeader"/> presents <paramref name="expectedToken"/>.
        /// </summary>
        /// <param name="authorizationHeader">The request's <c>Authorization</c> header, or null.</param>
        /// <param name="expectedToken">The configured secret. Null, empty or whitespace refuses everything.</param>
        public static bool IsAuthorised(string? authorizationHeader, string? expectedToken)
        {
            if (string.IsNullOrWhiteSpace(expectedToken))
            {
                return false;
            }

            var header = authorizationHeader ?? string.Empty;

            if (!header.StartsWith(Scheme, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            var presented = Encoding.UTF8.GetBytes(header.Substring(Scheme.Length).Trim());
            var expected = Encoding.UTF8.GetBytes(expectedToken.Trim());

            // FixedTimeEquals is length-safe: it returns false for differing lengths without branching on
            // content, so a wrong-length guess reveals nothing beyond the length it already knew it sent.
            return CryptographicOperations.FixedTimeEquals(presented, expected);
        }

        /// <summary>
        /// Why a call was refused, so the endpoint can answer 401 or 503 rather than conflating them.
        ///
        /// <para>The two are different problems and only one belongs to the caller: an operator holding a
        /// valid token needs to know the guard was never given one, and "unauthorised" sends them hunting for
        /// a mistake of their own that does not exist.</para>
        /// </summary>
        public static bool IsConfigured(string? expectedToken)
        {
            return !string.IsNullOrWhiteSpace(expectedToken);
        }
    }
}
