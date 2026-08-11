// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Threading;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace DevOnBike.Overfit.Analyzers
{
    /// <summary>
    /// "Is this expression a task" — the type gate shared by <see cref="SyncOverAsyncAnalyzer"/> (OVERFIT039)
    /// and <see cref="DiscardedTaskAnalyzer"/> (OVERFIT046).
    ///
    /// <para><b>It lives in one file because it is the whole rule in both of them, and a second copy would
    /// drift.</b> OVERFIT039's own doc-comment makes the point: name-matching instead of type-matching would
    /// flag every <c>SemaphoreSlim.Wait()</c> in the decode loop, and a rule with false positives on correct
    /// code gets suppressed wholesale rather than obeyed. The same sentence applies to OVERFIT046, where the
    /// noise would instead be the seventeen <c>_ = someParameter</c> unused-parameter suppressions in
    /// <c>Sources</c> — a discard is only interesting when what it throws away is a task.</para>
    ///
    /// <para>Walking the base chain rather than comparing one name is deliberate: a type deriving from
    /// <c>Task</c> is still a task, and <c>OriginalDefinition</c> collapses <c>Task&lt;T&gt;</c> to its
    /// unbound form so one comparison covers every element type.</para>
    /// </summary>
    internal static class AwaitableType
    {
        /// <summary>Whether <paramref name="expression"/> has type Task, Task&lt;T&gt;, ValueTask or ValueTask&lt;T&gt;.</summary>
        public static bool IsAwaitable(
            SemanticModel semanticModel, ExpressionSyntax expression, CancellationToken cancellationToken)
        {
            var type = semanticModel.GetTypeInfo(expression, cancellationToken).Type;

            if (type == null)
            {
                return false;
            }

            for (var current = type; current != null; current = current.BaseType)
            {
                var name = current.OriginalDefinition.ToDisplayString();

                if (name is "System.Threading.Tasks.Task"
                    or "System.Threading.Tasks.Task<TResult>"
                    or "System.Threading.Tasks.ValueTask"
                    or "System.Threading.Tasks.ValueTask<TResult>")
                {
                    return true;
                }
            }

            return false;
        }
    }
}
