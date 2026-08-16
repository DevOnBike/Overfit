// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Collections;
using System.Collections.ObjectModel;
using BenchmarkDotNet.Attributes;

namespace Benchmarks
{
    /// <summary>
    /// <c>for</c> against <c>foreach</c> with the loop body stripped down to a single field read, so the loop
    /// shape is the thing being measured rather than whatever it was wrapped around.
    ///
    /// <para><b>The general claim "foreach is free" is true for exactly one case and false for the rest, and
    /// the difference is the static type of the thing being iterated — not the keyword.</b> Over a
    /// <c>T[]</c> or a <c>Span&lt;T&gt;</c> the compiler lowers <c>foreach</c> to an indexed walk with no
    /// enumerator at all. Over a <c>List&lt;T&gt;</c> it uses a struct enumerator, which stays on the stack.
    /// Over an interface — <c>IReadOnlyList&lt;T&gt;</c>, <c>IEnumerable&lt;T&gt;</c>, <c>ICollection&lt;T&gt;</c>
    /// — <c>GetEnumerator</c> is an interface call returning a reference (or a boxed struct), so every loop
    /// allocates and every step is a virtual call that cannot inline. That is the row worth remembering:
    /// changing a field's declared type from <c>string[]</c> to <c>IReadOnlyList&lt;string&gt;</c> for
    /// tidiness silently converts an allocation-free loop into an allocating one.</para>
    ///
    /// <para>The body sums <c>string.Length</c>: one dependent load per element, nothing the JIT can
    /// vectorise away, and nothing expensive enough to bury the loop overhead — which is the mistake the
    /// realistic arm in <see cref="SignalCatalogLoopBenchmark"/> exists to demonstrate.</para>
    ///
    /// <para><see cref="SimpleJobAttribute"/> rather than the shared config: at twenty elements this is a
    /// nanosecond-scale routine and <c>InvocationCount=1</c> would measure the timer.</para>
    /// </summary>
    [SimpleJob]
    [MemoryDiagnoser]
    public class LoopShapeBenchmark
    {
        /// <summary>How many collections the monomorphic/polymorphic arms walk. Must divide <see cref="Elements"/>.</summary>
        private const int Shards = 4;

        private string[] _array = [];
        private List<string> _list = [];
        private IReadOnlyList<string> _readOnlyList = [];
        private IEnumerable<string> _enumerable = [];
        private IReadOnlyList<string>[] _monomorphic = [];
        private IReadOnlyList<string>[] _polymorphic = [];

        /// <summary>20 is the size of the real marker tables; 1000 separates per-element cost from call overhead.</summary>
        [Params(20, 1000)]
        public int Elements
        {
            get; set;
        }

        [GlobalSetup]
        public void Setup()
        {
            _array = new string[Elements];

            for (var i = 0; i < Elements; i++)
            {
                _array[i] = $"metric_name_number_{i}";
            }

            _list = new List<string>(_array);

            // Same array, only the declared type differs — which is the entire point of the interface arms.
            _readOnlyList = _array;
            _enumerable = _array;

            // Both sets hold the same number of elements in the same number of collections. The only
            // difference is how many distinct runtime types the shared call site observes.
            var shard = Elements / Shards;

            _monomorphic = new IReadOnlyList<string>[Shards];
            _polymorphic = new IReadOnlyList<string>[Shards];

            for (var s = 0; s < Shards; s++)
            {
                var items = new string[shard];
                _array.AsSpan(s * shard, shard).CopyTo(items);

                _monomorphic[s] = items;

                _polymorphic[s] = s switch
                {
                    0 => items,
                    1 => new List<string>(items),
                    2 => new MarkerList(items),
                    _ => new ReadOnlyCollection<string>(items)
                };
            }
        }

        [Benchmark(Baseline = true)]
        public int For_Array()
        {
            var total = 0;

            for (var i = 0; i < _array.Length; i++)
            {
                total += _array[i].Length;
            }

            return total;
        }

        [Benchmark]
        public int Foreach_Array()
        {
            var total = 0;

            foreach (var value in _array)
            {
                total += value.Length;
            }

            return total;
        }

        /// <summary>Over a span the enumerator is elided exactly as it is over the array.</summary>
        [Benchmark]
        public int Foreach_Span()
        {
            var total = 0;

            foreach (var value in _array.AsSpan())
            {
                total += value.Length;
            }

            return total;
        }

        [Benchmark]
        public int For_List()
        {
            var total = 0;

            for (var i = 0; i < _list.Count; i++)
            {
                total += _list[i].Length;
            }

            return total;
        }

        /// <summary>Struct enumerator, reached through the concrete type: no allocation, but not free either.</summary>
        [Benchmark]
        public int Foreach_List()
        {
            var total = 0;

            foreach (var value in _list)
            {
                total += value.Length;
            }

            return total;
        }

        /// <summary>The same array behind an interface. This is the arm the memory column is here for.</summary>
        [Benchmark]
        public int Foreach_IReadOnlyList()
        {
            var total = 0;

            foreach (var value in _readOnlyList)
            {
                total += value.Length;
            }

            return total;
        }

        /// <summary>Indexed access through the interface — no enumerator, but a virtual indexer per step.</summary>
        [Benchmark]
        public int For_IReadOnlyList()
        {
            var total = 0;

            for (var i = 0; i < _readOnlyList.Count; i++)
            {
                total += _readOnlyList[i].Length;
            }

            return total;
        }

        [Benchmark]
        public int Foreach_IEnumerable()
        {
            var total = 0;

            foreach (var value in _enumerable)
            {
                total += value.Length;
            }

            return total;
        }

        /// <summary>
        /// Four collections behind <c>IReadOnlyList&lt;string&gt;</c>, all of them <c>string[]</c> at runtime.
        /// The <c>GetEnumerator</c> call site sees one type, so profile-guided optimisation can devirtualise
        /// it, inline it, and let escape analysis keep the enumerator off the heap.
        /// </summary>
        [Benchmark]
        public int Foreach_Interface_Monomorphic()
        {
            var total = 0;

            for (var s = 0; s < _monomorphic.Length; s++)
            {
                foreach (var value in _monomorphic[s])
                {
                    total += value.Length;
                }
            }

            return total;
        }

        /// <summary>
        /// The same element count through the same interface, but four distinct runtime types — array,
        /// <c>List&lt;T&gt;</c>, a hand-written collection with a class enumerator, and
        /// <c>ReadOnlyCollection&lt;T&gt;</c>.
        ///
        /// <para><b>This arm exists because the monomorphic measurement flatters the interface.</b> A
        /// benchmark that only ever puts one implementation behind an abstraction is not measuring the
        /// abstraction — it is measuring the JIT's ability to see through it, which is precisely what real
        /// polymorphic code takes away. Read the <c>Allocated</c> column first: it is binary and unambiguous,
        /// whereas the timing here carries two effects at once (lost devirtualisation, and the genuinely
        /// different cost of four different enumerators) and cannot separate them.</para>
        /// </summary>
        [Benchmark]
        public int Foreach_Interface_Polymorphic()
        {
            var total = 0;

            for (var s = 0; s < _polymorphic.Length; s++)
            {
                foreach (var value in _polymorphic[s])
                {
                    total += value.Length;
                }
            }

            return total;
        }

        /// <summary>
        /// Indexed access over the polymorphic set: no enumerator to allocate, but the virtual indexer is now
        /// a call site the JIT cannot resolve either.
        /// </summary>
        [Benchmark]
        public int For_Interface_Polymorphic()
        {
            var total = 0;

            for (var s = 0; s < _polymorphic.Length; s++)
            {
                var collection = _polymorphic[s];

                for (var i = 0; i < collection.Count; i++)
                {
                    total += collection[i].Length;
                }
            }

            return total;
        }

        /// <summary>
        /// A collection whose enumerator is a class rather than a struct, so that if the JIT fails to inline
        /// <see cref="GetEnumerator"/> there is nothing left to keep the object off the heap. Deliberately
        /// not a wrapper over <c>List&lt;T&gt;</c>: the point is a fourth distinct type at the call site.
        /// </summary>
        private sealed class MarkerList : IReadOnlyList<string>
        {
            private readonly string[] _items;

            internal MarkerList(string[] items)
            {
                _items = items;
            }

            public int Count => _items.Length;

            public string this[int index] => _items[index];

            public IEnumerator<string> GetEnumerator() => new Cursor(_items);

            IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

            private sealed class Cursor : IEnumerator<string>
            {
                private readonly string[] _items;
                private int _index = -1;

                internal Cursor(string[] items)
                {
                    _items = items;
                }

                public string Current => _items[_index];

                object IEnumerator.Current => Current;

                public bool MoveNext()
                {
                    _index++;

                    return _index < _items.Length;
                }

                public void Reset()
                {
                    _index = -1;
                }

                public void Dispose()
                {
                }
            }
        }
    }
}
