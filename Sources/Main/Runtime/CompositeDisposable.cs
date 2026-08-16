// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Runtime
{
    /// <summary>
    /// Disposes a small fixed set of owned resources together, in order. Used to hand a single
    /// <see cref="IDisposable"/> to an owner that must keep several backing resources alive for the same
    /// lifetime (e.g. a model's memory-mapped file plus its offline-repacked-weights sidecar map).
    /// </summary>
    public sealed class CompositeDisposable : IDisposable
    {
        private readonly IDisposable?[] _resources;

        public CompositeDisposable(params IDisposable?[] resources)
        {
            _resources = resources ?? throw new ArgumentNullException(nameof(resources));
        }

        /// <summary>Returns a single disposable owning the non-null items, or the sole item / null when there is
        /// at most one — so callers avoid wrapping when they don't need to.</summary>
        public static IDisposable? Of(params IDisposable?[] resources)
        {
            var live = 0;
            IDisposable? last = null;
            foreach (var r in resources)
            {
                if (r != null)
                {
                    live++;
                    last = r;
                }
            }

            return live switch
            {
                0 => null,
                1 => last,
                _ => new CompositeDisposable(resources),
            };
        }

        public void Dispose()
        {
            foreach (var r in _resources)
            {
                r?.Dispose();
            }
        }
    }
}
