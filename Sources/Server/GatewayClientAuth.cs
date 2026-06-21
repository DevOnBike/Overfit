// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Security.Cryptography;
using System.Text;

namespace DevOnBike.Overfit.Server
{
    /// <summary>
    /// Authenticates callers to the gateway with a set of gateway-issued client keys — distinct from the upstream key
    /// the gateway holds. Clients present <c>Authorization: Bearer &lt;gateway-key&gt;</c>; the gateway validates it,
    /// then swaps in the real upstream key (which never reaches the client). Keys are compared in constant time
    /// against their SHA-256 hashes, so neither the key nor its length leaks through timing.
    ///
    /// <para>When no keys are configured, authentication is <see cref="Enabled"/> = false and every caller is allowed
    /// (with a startup warning) — so an existing deployment keeps working until keys are issued.</para>
    /// </summary>
    public sealed class GatewayClientAuth
    {
        private readonly byte[][] _hashes;

        public GatewayClientAuth(IReadOnlyCollection<string>? clientKeys)
        {
            if (clientKeys is null || clientKeys.Count == 0)
            {
                _hashes = [];
                return;
            }

            var list = new List<byte[]>(clientKeys.Count);
            foreach (var key in clientKeys)
            {
                if (!string.IsNullOrWhiteSpace(key))
                {
                    list.Add(SHA256.HashData(Encoding.UTF8.GetBytes(key.Trim())));
                }
            }

            _hashes = list.ToArray();
        }

        /// <summary>True when at least one client key is configured — i.e. callers must authenticate.</summary>
        public bool Enabled => _hashes.Length > 0;

        /// <summary>
        /// Returns true when the request is authorized: either auth is disabled, or the bearer token in
        /// <paramref name="authorizationHeader"/> matches a configured key. The comparison runs against every key in
        /// constant time so a near-miss is indistinguishable from a far-miss.
        /// </summary>
        public bool IsAuthorized(string? authorizationHeader)
        {
            if (!Enabled)
            {
                return true;
            }

            if (string.IsNullOrEmpty(authorizationHeader))
            {
                return false;
            }

            var token = authorizationHeader.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase)
                ? authorizationHeader.Substring("Bearer ".Length).Trim()
                : authorizationHeader.Trim();

            if (token.Length == 0)
            {
                return false;
            }

            var presented = SHA256.HashData(Encoding.UTF8.GetBytes(token));

            var authorized = false;
            foreach (var hash in _hashes)
            {
                // Compare against ALL configured keys (no early exit) to keep total time independent of which matched.
                authorized |= CryptographicOperations.FixedTimeEquals(presented, hash);
            }

            return authorized;
        }
    }
}
