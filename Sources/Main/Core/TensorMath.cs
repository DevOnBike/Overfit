using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DevOnBike.Overfit.Core
{
    public static class TensorMath
    {
        // ====================================================================
        // 1. PODSTAWOWA ALGEBRA I MATERIE (ZOPTYMALIZOWANE)
        // ====================================================================

        public static AutogradNode Add(AutogradNode left, AutogradNode right)
        {
            var resultData = new FastTensor<float>(left.Data.Shape);
            TensorPrimitives.Add(left.Data.AsSpan(), right.Data.AsSpan(), resultData.AsSpan());

            var outputNode = new AutogradNode(resultData, left.RequiresGrad || right.RequiresGrad);
            if (outputNode.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.Add, outputNode, left, right);

            return outputNode;
        }

        public static void AddBackward(AutogradNode a, AutogradNode b, AutogradNode output)
        {
            if (a.RequiresGrad) TensorPrimitives.Add(a.Grad.AsSpan(), output.Grad.AsSpan(), a.Grad.AsSpan());
            if (b.RequiresGrad) TensorPrimitives.Add(b.Grad.AsSpan(), output.Grad.AsSpan(), b.Grad.AsSpan());
        }

        public static AutogradNode AddBias(AutogradNode input, AutogradNode bias)
        {
            var N = input.Data.GetDim(0); var C = input.Data.GetDim(1);
            var resultData = new FastTensor<float>(input.Data.Shape);
            var inS = input.Data.AsSpan(); var bS = bias.Data.AsSpan(); var resS = resultData.AsSpan();

            for (var i = 0; i < N; i++)
                TensorPrimitives.Add(inS.Slice(i * C, C), bS, resS.Slice(i * C, C));

            var outputNode = new AutogradNode(resultData, input.RequiresGrad || bias.RequiresGrad);
            if (outputNode.RequiresGrad) ComputationGraph.Active.Record(OpCode.AddBias, outputNode, input, bias);
            return outputNode;
        }

        public static void AddBiasBackward(AutogradNode input, AutogradNode bias, AutogradNode output)
        {
            int N = input.Data.GetDim(0), C = input.Data.GetDim(1);
            if (input.RequiresGrad) TensorPrimitives.Add(input.Grad.AsSpan(), output.Grad.AsSpan(), input.Grad.AsSpan());
            if (bias.RequiresGrad)
            {
                var bGS = bias.Grad.AsSpan();
                var oGS = output.Grad.AsSpan();
                for (var i = 0; i < N; i++)
                    TensorPrimitives.Add(bGS, oGS.Slice(i * C, C), bGS);
            }
        }

        public static AutogradNode MatMul(AutogradNode left, AutogradNode right)
        {
            var resultData = MatMulRaw(left.Data, right.Data);
            var outputNode = new AutogradNode(resultData, left.RequiresGrad || right.RequiresGrad);
            if (outputNode.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.MatMul, outputNode, left, right);
            return outputNode;
        }

        public static FastTensor<float> MatMulRaw(FastTensor<float> A, FastTensor<float> B)
        {
            int aRows = A.GetDim(0), aCols = A.GetDim(1), bCols = B.GetDim(1);
            var C = new FastTensor<float>(aRows, bCols);

            Parallel.For(0, aRows, i =>
            {
                var aS = A.AsSpan(); var bS = B.AsSpan(); var cS = C.AsSpan();
                var rowC = cS.Slice(i * bCols, bCols);
                for (var k = 0; k < aCols; k++)
                {
                    var valA = aS[i * aCols + k];
                    if (valA == 0) continue;
                    TensorPrimitives.MultiplyAdd(bS.Slice(k * bCols, bCols), valA, rowC, rowC);
                }
            });
            return C;
        }

        private static void MatMulRawSequential(ReadOnlySpan<float> aS, ReadOnlySpan<float> bS, int aR, int aC, int bC, Span<float> cS)
        {
            cS.Clear();

            for (var i = 0; i < aR; i++)
            {
                var rowC = cS.Slice(i * bC, bC);
                var rowA = aS.Slice(i * aC, aC);

                for (var k = 0; k < aC; k++)
                {
                    var valA = rowA[k];

                    if (valA != 0f)
                    {
                        TensorPrimitives.MultiplyAdd(bS.Slice(k * bC, bC), valA, rowC, rowC);
                    }
                }
            }
        }

        public static void MatMulBackward(AutogradNode a, AutogradNode b, AutogradNode output)
        {
            if (a.RequiresGrad)
            {
                // GradA = GradOut * B^T
                using var gradA = MatMul_A_BT(output.Grad, b.Data);
                TensorPrimitives.Add(a.Grad.AsSpan(), gradA.AsSpan(), a.Grad.AsSpan());
            }

            if (b.RequiresGrad)
            {
                // GradB = A^T * GradOut
                using var gradB = MatMul_AT_B(a.Data, output.Grad);
                TensorPrimitives.Add(b.Grad.AsSpan(), gradB.AsSpan(), b.Grad.AsSpan());
            }
        }

        // Zoptymalizowane C = A * B^T (Używa SIMD Dot Product)
        private static FastTensor<float> MatMul_A_BT(FastTensor<float> A, FastTensor<float> B)
        {
            int N = A.GetDim(0), K = A.GetDim(1), M = B.GetDim(0);
            var C = new FastTensor<float>(false, N, M);

            Parallel.For(0, N, i =>
            {
                var aRow = A.AsSpan().Slice(i * K, K);
                var cRow = C.AsSpan().Slice(i * M, M);

                // HOISTING: Wyciągamy Span z B przed wewnętrzną pętlę
                var bS = B.AsSpan();

                for (var j = 0; j < M; j++)
                {
                    // Używamy zbuforowanego bS
                    cRow[j] = TensorPrimitives.Dot(aRow, bS.Slice(j * K, K));
                }
            });
            return C;
        }

        // Zoptymalizowane C = A^T * B (Używa SIMD MultiplyAdd)
        private static FastTensor<float> MatMul_AT_B(FastTensor<float> A, FastTensor<float> B)
        {
            int K = A.GetDim(0), N = A.GetDim(1), M = B.GetDim(1);
            // Ważne: true, bo sumujemy wartości do zera
            var C = new FastTensor<float>(true, N, M);

            Parallel.For(0, N, i =>
            {
                var cRow = C.AsSpan().Slice(i * M, M);

                // HOISTING: Inicjalizacja Span na stosie tylko raz per wątek/wiersz!
                var aS = A.AsSpan();
                var bS = B.AsSpan();

                for (var k = 0; k < K; k++)
                {
                    var aVal = aS[k * N + i];

                    if (aVal == 0) continue;

                    // Używamy zbuforowanego bS
                    TensorPrimitives.MultiplyAdd(bS.Slice(k * M, M), aVal, cRow, cRow);
                }
            });

            return C;
        }

        // ====================================================================
        // 2. CNN - CONV, POOL, GAP (NCHW)
        // ====================================================================

        public static AutogradNode Conv2D(AutogradNode input, AutogradNode weights, int inC, int outC, int h, int w, int k)
        {
            int outH = h - k + 1, outW = w - k + 1, batchSize = input.Data.GetDim(0), kSqInC = k * k * inC;
            int colSizePerImg = kSqInC * outH * outW, inSize = inC * h * w, outSize = outC * outH * outW;

            using var workspace = new FastTensor<float>(false, batchSize, colSizePerImg);
            var resultData = new FastTensor<float>(batchSize, outC, outH, outW);
            using var weights2D = weights.Data.Reshape(outC, kSqInC);

            Parallel.For(0, batchSize, n =>
            {
                var colSpan = workspace.AsSpan().Slice(n * colSizePerImg, colSizePerImg);
                Im2Col(input.Data.AsSpan().Slice(n * inSize, inSize), inC, h, w, k, 1, 0, colSpan);
                MatMulRawSequential(weights2D.AsSpan(), colSpan, outC, kSqInC, outH * outW, resultData.AsSpan().Slice(n * outSize, outSize));
            });

            var outputNode = new AutogradNode(resultData, input.RequiresGrad || weights.RequiresGrad);
            if (outputNode.RequiresGrad) ComputationGraph.Active.Record(OpCode.Conv2D, outputNode, input, weights, inC, outC, h, w, k);
            return outputNode;
        }

        public static void Conv2DBackward(AutogradNode input, AutogradNode weights, AutogradNode output, int inC, int outC, int h, int w, int k)
        {
            if (!input.RequiresGrad && !weights.RequiresGrad) return;

            int outH = h - k + 1, outW = w - k + 1, batchSize = input.Data.GetDim(0), kSqInC = k * k * inC;
            int colSizePerImg = kSqInC * outH * outW, inSize = inC * h * w, outSize = outC * outH * outW;
            var K = outH * outW;

            FastTensor<float> weights2DTContig = null;
            if (input.RequiresGrad)
            {
                using var weights2D = weights.Data.Reshape(outC, kSqInC);
                using var weights2DT = weights2D.Transpose(0, 1);
                weights2DTContig = weights2DT.ToContiguous();
            }

            var weightLock = new object();

            // LEVEL 5: Brak zagnieżdżonej równoległości, zero alokacji w najgłębszych pętlach!
            Parallel.For(0, batchSize,
                // 1. Thread-Local Storage dla wątku
                () => weights.RequiresGrad ? new FastTensor<float>(true, outC, kSqInC) : null,

                // 2. Ciało pętli
                (n, loopState, localDw) =>
                {
                    using var colData = new FastTensor<float>(false, kSqInC, K);
                    Im2Col(input.Data.AsSpan().Slice(n * inSize, inSize), inC, h, w, k, 1, 0, colData.AsSpan());

                    var outGradSlice = output.Grad.AsSpan().Slice(n * outSize, outSize);
                    using var outGradMat = new FastTensor<float>(false, outC, K);
                    outGradSlice.CopyTo(outGradMat.AsSpan());

                    // 3. Gradienty Wag (dW) - MEGA SZYBKI DOT PRODUCT ZAMIAST MATMUL
                    if (weights.RequiresGrad)
                    {
                        var dwSpan = localDw.AsSpan();
                        var outGradSpan = outGradMat.AsSpan();
                        var colSpan = colData.AsSpan();

                        for (var r = 0; r < outC; r++)
                        {
                            var outGradRow = outGradSpan.Slice(r * K, K);
                            for (var c = 0; c < kSqInC; c++)
                            {
                                // Bezpośredni SIMD Dot Product wymazuje potrzebę drogiego Transpose/ToContiguous!
                                dwSpan[r * kSqInC + c] += TensorPrimitives.Dot(outGradRow, colSpan.Slice(c * K, K));
                            }
                        }
                    }

                    // 4. Gradienty Wejścia (dX)
                    if (input.RequiresGrad)
                    {
                        using var dCol = new FastTensor<float>(false, kSqInC, K);
                        // Używamy MatMulRawSequential, by nie tworzyć równoległości wewnątrz równoległości!
                        MatMulRawSequential(weights2DTContig.AsSpan(), outGradMat.AsSpan(), kSqInC, outC, K, dCol.AsSpan());
                        Col2Im(dCol.AsSpan(), inC, h, w, k, 1, 0, input.Grad.AsSpan().Slice(n * inSize, inSize));
                    }

                    return localDw;
                },

                // 3. Zrzut danych z wątku do wspólnej puli (tylko raz na zakończenie życia wątku!)
                (localDw) =>
                {
                    if (localDw != null)
                    {
                        lock (weightLock)
                        {
                            TensorPrimitives.Add(weights.Grad.AsSpan(), localDw.AsSpan(), weights.Grad.AsSpan());
                        }
                        localDw.Dispose();
                    }
                }
            );

            weights2DTContig?.Dispose();
        }

        public static AutogradNode MaxPool2D(AutogradNode input, int channels, int inputH, int inputW, int poolSize)
        {
            int outputH = inputH / poolSize, outputW = inputW / poolSize, batchSize = input.Data.GetDim(0);
            var resultData = new FastTensor<float>(batchSize, channels, outputH, outputW);
            var maxIndices = new AutogradNode(new FastTensor<float>(batchSize, channels, outputH, outputW), false);

            Parallel.For(0, batchSize, n =>
            {
                // 1. Pobieramy "nagie" wskaźniki-referencje do początku pamięci (ZERO BOUNDS CHECKING)
                ref var inRef = ref MemoryMarshal.GetReference(input.Data.AsSpan());
                ref var outRef = ref MemoryMarshal.GetReference(resultData.AsSpan());
                ref var idxRef = ref MemoryMarshal.GetReference(maxIndices.Data.AsSpan());

                var bInOffset = n * channels * inputH * inputW;
                var bOutOffset = n * channels * outputH * outputW;

                for (var c = 0; c < channels; c++)
                {
                    var cInOffset = bInOffset + c * inputH * inputW;
                    var cOutOffset = bOutOffset + c * outputH * outputW;

                    for (var oh = 0; oh < outputH; oh++)
                    {
                        var ohInOffset = cInOffset + oh * poolSize * inputW;
                        var ohOutOffset = cOutOffset + oh * outputW;

                        for (var ow = 0; ow < outputW; ow++)
                        {
                            var maxVal = float.MinValue;
                            var maxIdx = -1;
                            var owInOffset = ohInOffset + ow * poolSize;

                            for (var ph = 0; ph < poolSize; ph++)
                            {
                                var phInOffset = owInOffset + ph * inputW;

                                for (var pw = 0; pw < poolSize; pw++)
                                {
                                    var absIdx = phInOffset + pw;

                                    // BŁYSKAWICZNY ODCZYT: Kompiluje się do instrukcji mov w asemblerze!
                                    var val = Unsafe.Add(ref inRef, absIdx);

                                    if (val > maxVal) { maxVal = val; maxIdx = absIdx; }
                                }
                            }

                            var outAbsIdx = ohOutOffset + ow;

                            // BŁYSKAWICZNY ZAPIS
                            Unsafe.Add(ref outRef, outAbsIdx) = maxVal;
                            Unsafe.Add(ref idxRef, outAbsIdx) = maxIdx;
                        }
                    }
                }
            });

            var output = new AutogradNode(resultData, input.RequiresGrad);
            if (output.RequiresGrad) ComputationGraph.Active.Record(OpCode.MaxPool2D, output, input, maxIndices);
            return output;
        }

        public static void MaxPool2DBackward(AutogradNode input, AutogradNode maxIndices, AutogradNode output)
        {
            // Propagacja wsteczna też bez bounds checkingu!
            ref var iGRef = ref MemoryMarshal.GetReference(input.Grad.AsSpan());
            ref var oGRef = ref MemoryMarshal.GetReference(output.Grad.AsSpan());
            ref var idxRef = ref MemoryMarshal.GetReference(maxIndices.Data.AsSpan());

            var len = maxIndices.Data.Size;
            for (var i = 0; i < len; i++)
            {
                var maxIdx = (int)Unsafe.Add(ref idxRef, i);
                Unsafe.Add(ref iGRef, maxIdx) += Unsafe.Add(ref oGRef, i);
            }
        }

        public static AutogradNode GlobalAveragePool2D(AutogradNode input, int channels, int h, int w)
        {
            var batchSize = input.Data.GetDim(0);
            var resData = new FastTensor<float>(batchSize, channels);
            float spatialSize = h * w;

            Parallel.For(0, batchSize, n =>
            {
                for (var c = 0; c < channels; c++)
                    resData[n, c] = TensorPrimitives.Sum(input.Data.AsSpan().Slice(n * channels * h * w + c * h * w, h * w)) / spatialSize;
            });

            var output = new AutogradNode(resData, input.RequiresGrad);
            if (output.RequiresGrad) ComputationGraph.Active.Record(OpCode.GlobalAveragePool2D, output, input, null, h, w, channels);
            return output;
        }

        public static void GlobalAvgPool2DBackward(AutogradNode input, AutogradNode output, int h, int w, int channels)
        {
            var batchSize = input.Data.GetDim(0); float spatialSize = h * w;
            Parallel.For(0, batchSize, n =>
            {
                for (var c = 0; c < channels; c++)
                    input.Grad.AsSpan().Slice(n * channels * h * w + c * h * w, h * w).Fill(output.Grad[n, c] / spatialSize);
            });
        }

        // ====================================================================
        // 3. AKTYWACJE I REGULARYZACJA
        // ====================================================================

        public static AutogradNode ReLU(AutogradNode input)
        {
            // Używamy clearMemory: false, ponieważ funkcja SIMD i tak nadpisze CAŁY bufor.
            // Oszczędza to czas systemu operacyjnego na czyszczeniu pamięci.
            var res = new FastTensor<float>(false, input.Data.Shape);

            TensorPrimitives.Max(input.Data.AsSpan(), 0f, res.AsSpan());

            var output = new AutogradNode(res, input.RequiresGrad);
            if (output.RequiresGrad)
                ComputationGraph.Active.Record(OpCode.ReLU, output, input);

            return output;
        }

        public static void ReluBackward(AutogradNode input, AutogradNode output)
        {
            if (!input.RequiresGrad)
            {
                return;
            }

            // ReadOnlySpan — kompilator wie że nie piszemy przez te spany → może lepiej optymalizować
            var inS = (ReadOnlySpan<float>)input.Data.AsSpan();
            var goS = (ReadOnlySpan<float>)output.Grad.AsSpan();
            var giS = input.Grad.AsSpan();
            var i = 0;

            // SIMD path — Vector<float> zamiast TensorPrimitives.Sign, bo Sign zwraca Span<int>
            // a nie Span<float> — nie da się go użyć bezpośrednio w MultiplyAdd.
            //
            // Na 9950X3D (.NET 10, x64-v4): Vector<float>.Count == 16 (AVX-512)
            // → 16 float per iteracja, maska w rejestrze, zero alokacji, zero skoków.
            if (Vector.IsHardwareAccelerated)
            {
                var vZero = Vector<float>.Zero;
                var vecSize = Vector<float>.Count;

                for (; i <= inS.Length - vecSize; i += vecSize)
                {
                    var vIn = new Vector<float>(inS.Slice(i));
                    var vGo = new Vector<float>(goS.Slice(i));
                    var vGi = new Vector<float>(giS.Slice(i));

                    // GreaterThan → maska bitowa 0xFFFFFFFF / 0x00000000 w rejestrze
                    var vMask = Vector.GreaterThan(vIn, vZero);

                    // ConditionalSelect: przepuść vGo tam gdzie in > 0, wygaś do 0 gdzie in <= 0
                    var vFiltered = Vector.ConditionalSelect(vMask, vGo, vZero);

                    // giS += filtered_go — akumulacja w rejestrze, jeden zapis do pamięci
                    (vGi + vFiltered).CopyTo(giS.Slice(i));
                }
            }

            // Scalar tail — obsługuje resztę gdy len % vecSize != 0 (lub gdy brak AVX)
            for (; i < inS.Length; i++)
            {
                if (inS[i] > 0f)
                {
                    giS[i] += goS[i];
                }
            }
        }

        public static AutogradNode Dropout(AutogradNode input, float probability, bool isTraining)
        {
            var resData = new FastTensor<float>(input.Data.Shape);
            var mask = new AutogradNode(new FastTensor<float>(input.Data.Shape), false);

            if (isTraining)
            {
                var scale = 1f / (1f - probability);
                for (var i = 0; i < input.Data.Size; i++)
                {
                    if (Random.Shared.NextSingle() > probability) { resData.AsSpan()[i] = input.Data.AsSpan()[i] * scale; mask.Data.AsSpan()[i] = scale; }
                }
            }
            else
            {
                input.Data.AsSpan().CopyTo(resData.AsSpan());
            }

            var output = new AutogradNode(resData, input.RequiresGrad);

            if (output.RequiresGrad && isTraining)
            {
                ComputationGraph.Active.Record(OpCode.Dropout, output, input, mask);
            }
            else
            {
                mask.Dispose();
            }

            return output;
        }

        public static void DropoutBackward(AutogradNode input, AutogradNode mask, AutogradNode output)
        {
            var giS = input.Grad.AsSpan();
            var goS = output.Grad.AsSpan();
            var mS = mask.Data.AsSpan();

            // OSTATECZNA MAGIA SIMD: Jeden wektorowy strzał przez całą pamięć!
            TensorPrimitives.MultiplyAdd(goS, mS, giS, giS);
        }

        // ====================================================================
        // 4. FUNKCJE STRATY (LOSS)
        // ====================================================================

        public static AutogradNode SoftmaxCrossEntropy(AutogradNode logits, AutogradNode target)
        {
            int rows = logits.Data.GetDim(0), cols = logits.Data.GetDim(1);
            var totalLoss = 0f;
            using var pRowBuf = new FastTensor<float>(cols);
            for (var r = 0; r < rows; r++)
            {
                TensorPrimitives.SoftMax(logits.Data.AsSpan().Slice(r * cols, cols), pRowBuf.AsSpan());
                for (var c = 0; c < cols; c++) if (target.Data[r, c] > 0.5f) totalLoss -= MathF.Log(pRowBuf.AsSpan()[c] + 1e-15f);
            }
            var res = new FastTensor<float>(1, 1) { [0, 0] = totalLoss / rows };
            var output = new AutogradNode(res, logits.RequiresGrad);
            if (logits.RequiresGrad) ComputationGraph.Active.Record(OpCode.SoftmaxCrossEntropy, output, logits, target);
            return output;
        }

        public static void SoftmaxCrossEntropyBackward(AutogradNode logits, AutogradNode target, AutogradNode output)
        {
            int rows = logits.Data.GetDim(0), cols = logits.Data.GetDim(1);
            var scale = output.Grad[0, 0] / rows;
            using var pRowBuf = new FastTensor<float>(cols);
            for (var r = 0; r < rows; r++)
            {
                TensorPrimitives.SoftMax(logits.Data.AsSpan().Slice(r * cols, cols), pRowBuf.AsSpan());
                for (var c = 0; c < cols; c++) logits.Grad[r, c] += (pRowBuf.AsSpan()[c] - target.Data[r, c]) * scale;
            }
        }

        public static AutogradNode MSELoss(AutogradNode prediction, AutogradNode target)
        {
            using var diff = new FastTensor<float>(prediction.Data.Shape);

            TensorPrimitives.Subtract(prediction.Data.AsSpan(), target.Data.AsSpan(), diff.AsSpan());
            var mse = TensorPrimitives.Dot(diff.AsSpan(), diff.AsSpan()) / prediction.Data.Size;
            var res = new FastTensor<float>(1, 1) { [0, 0] = mse };
            var output = new AutogradNode(res, prediction.RequiresGrad);
            if (output.RequiresGrad) ComputationGraph.Active.Record(OpCode.MSELoss, output, prediction, target);

            return output;
        }

        public static void MSELossBackward(AutogradNode p, AutogradNode t, AutogradNode o)
        {
            // Pobieramy skalar z pierwszej pozycji (najszybszy dostęp)
            var factor = o.Grad.AsSpan()[0] * (2f / p.Data.Size);

            var pGrad = p.Grad.AsSpan();
            var pData = p.Data.AsSpan();
            var tData = t.Data.AsSpan();

            // Magia SIMD bez alokacji: (pData - tData) * factor => pData * factor - tData * factor
            TensorPrimitives.MultiplyAdd(pData, factor, pGrad, pGrad);
            TensorPrimitives.MultiplyAdd(tData, -factor, pGrad, pGrad);
        }

        // ====================================================================
        // 5. BATCH NORM I NARZĘDZIA
        // ====================================================================

        public static AutogradNode BatchNorm1D(AutogradNode input, AutogradNode gamma, AutogradNode beta, FastTensor<float> runningMean, FastTensor<float> runningVar, float momentum, float eps, bool isTraining)
        {
            int N = input.Data.GetDim(0), C = input.Data.GetDim(1);
            var outputData = new FastTensor<float>(input.Data.Shape);
            var mean = new AutogradNode(new FastTensor<float>(C), false);
            var invStd = new AutogradNode(new FastTensor<float>(C), false);

            if (isTraining)
            {
                for (var i = 0; i < N; i++) TensorPrimitives.Add(mean.Data.AsSpan(), input.Data.AsSpan().Slice(i * C, C), mean.Data.AsSpan());
                foreach (ref var m in mean.Data.AsSpan()) m /= N;
                for (var c = 0; c < C; c++)
                {
                    float varSum = 0; for (var i = 0; i < N; i++) { var d = input.Data[i, c] - mean.Data[c]; varSum += d * d; }
                    var bVar = varSum / N; invStd.Data[c] = 1f / MathF.Sqrt(bVar + eps);
                    runningMean[c] = (1 - momentum) * runningMean[c] + momentum * mean.Data[c];
                    runningVar[c] = (1 - momentum) * runningVar[c] + momentum * bVar;
                }
            }
            else
            {
                for (var c = 0; c < C; c++) { mean.Data[c] = runningMean[c]; invStd.Data[c] = 1f / MathF.Sqrt(runningVar[c] + eps); }
            }

            for (var i = 0; i < N; i++)
            {
                for (var c = 0; c < C; c++) outputData[i, c] = gamma.Data[c] * (input.Data[i, c] - mean.Data[c]) * invStd.Data[c] + beta.Data[c];
            }

            var output = new AutogradNode(outputData, input.RequiresGrad);
            if (output.RequiresGrad && isTraining) ComputationGraph.Active.Record(OpCode.BatchNorm1D, output, input, null, 0, 0, 0, 0, 0, new[] { gamma, beta, mean, invStd });
            return output;
        }

        public static void BatchNorm1DBackward(
            AutogradNode input,
            AutogradNode output,
            AutogradNode gamma,
            AutogradNode beta,
            AutogradNode mean,
            AutogradNode invStd)
        {
            if (!input.RequiresGrad && !gamma.RequiresGrad && !beta.RequiresGrad) return;

            int N = input.Data.GetDim(0), C = input.Data.GetDim(1);

            var inS = input.Data.AsSpan();
            var outGradS = output.Grad.AsSpan();
            var meanS = (ReadOnlySpan<float>)mean.Data.AsSpan();
            var invStdS = (ReadOnlySpan<float>)invStd.Data.AsSpan();
            var gammaS = (ReadOnlySpan<float>)gamma.Data.AsSpan();

            // ── Bufory robocze ───────────────────────────────────────────────────
            const int StackAllocThreshold = 256;

            FastBuffer<float> xHatBuf = null;
            FastBuffer<float> coeffBuf = null;
            FastBuffer<float> termBuf = null;

            try
            {
                var xHatRow = C <= StackAllocThreshold
                    ? stackalloc float[C]
                    : (xHatBuf = new FastBuffer<float>(C)).AsSpan();

                var coeff = C <= StackAllocThreshold
                    ? stackalloc float[C]
                    : (coeffBuf = new FastBuffer<float>(C)).AsSpan();

                var term = C <= StackAllocThreshold
                    ? stackalloc float[C]
                    : (termBuf = new FastBuffer<float>(C)).AsSpan();

                var sumDy = C <= StackAllocThreshold ? stackalloc float[C] : new float[C];
                var sumDyXHat = C <= StackAllocThreshold ? stackalloc float[C] : new float[C];

                // ── Krok 0: Pre-kalkulacja coeff = gamma * invStd / N ────────────
                TensorPrimitives.Multiply(gammaS, invStdS, coeff);
                TensorPrimitives.Multiply(coeff, 1f / N, coeff);

                // ── Krok 1A: Redukcja sumDy / sumDyXHat przez wiersze ────────────
                for (var i = 0; i < N; i++)
                {
                    var gradRow = (ReadOnlySpan<float>)outGradS.Slice(i * C, C);
                    var inRow = (ReadOnlySpan<float>)inS.Slice(i * C, C);

                    TensorPrimitives.Subtract(inRow, meanS, xHatRow);
                    TensorPrimitives.Multiply(xHatRow, invStdS, xHatRow);

                    TensorPrimitives.Add(sumDy, gradRow, sumDy);

                    TensorPrimitives.MultiplyAdd(gradRow, xHatRow, sumDyXHat, sumDyXHat);

                    if (beta.RequiresGrad)
                        TensorPrimitives.Add(beta.Grad.AsSpan(), gradRow, beta.Grad.AsSpan());

                    if (gamma.RequiresGrad)
                        TensorPrimitives.MultiplyAdd(gradRow, xHatRow, gamma.Grad.AsSpan(), gamma.Grad.AsSpan());
                }

                // ── Krok 1B: Gradient wejścia ────────────────────────────────────
                if (input.RequiresGrad)
                {
                    var inGradS = input.Grad.AsSpan();

                    for (var i = 0; i < N; i++)
                    {
                        var gradRow = (ReadOnlySpan<float>)outGradS.Slice(i * C, C);
                        var inRow = (ReadOnlySpan<float>)inS.Slice(i * C, C);
                        var inGradRow = inGradS.Slice(i * C, C);

                        TensorPrimitives.Subtract(inRow, meanS, xHatRow);
                        TensorPrimitives.Multiply(xHatRow, invStdS, xHatRow);

                        TensorPrimitives.Multiply(gradRow, (float)N, term);
                        TensorPrimitives.Subtract(term, sumDy, term);

                        TensorPrimitives.Multiply(xHatRow, sumDyXHat, xHatRow);
                        TensorPrimitives.Subtract(term, xHatRow, term);

                        TensorPrimitives.MultiplyAdd(coeff, term, inGradRow, inGradRow);
                    }
                }
            }
            finally
            {
                xHatBuf?.Dispose();
                coeffBuf?.Dispose();
                termBuf?.Dispose();
            }
        }

        public static AutogradNode Reshape(AutogradNode input, params int[] newShape)
        {
            var output = new AutogradNode(input.Data.Reshape(newShape), input.RequiresGrad);
            if (output.RequiresGrad) ComputationGraph.Active.Record(OpCode.Reshape, output, input);
            return output;
        }

        public static void ReshapeBackward(AutogradNode input, AutogradNode output)
        {
            TensorPrimitives.Add(input.Grad.AsSpan(), output.Grad.AsSpan(), input.Grad.AsSpan());
        }

        public static void Im2Col(ReadOnlySpan<float> input, int channels, int height, int width, int kSize, int stride, int padding, Span<float> output)
        {
            var outH = (height + 2 * padding - kSize) / stride + 1;
            var outW = (width + 2 * padding - kSize) / stride + 1;
            var channelSize = height * width;

            for (var c = 0; c < channels; c++)
            {
                for (var kh = 0; kh < kSize; kh++)
                {
                    for (var kw = 0; kw < kSize; kw++)
                    {
                        var rowOffset = (c * kSize * kSize + kh * kSize + kw) * outH * outW;
                        for (var y = 0; y < outH; y++)
                        {
                            var i = y * stride - padding + kh;
                            var outIdxY = rowOffset + y * outW;

                            if (i >= 0 && i < height)
                            {
                                var inputRowOffset = c * channelSize + i * width;

                                // FAST PATH: Zero instrukcji warunkowych w najgłębszej pętli
                                if (stride == 1)
                                {
                                    var startX = Math.Max(0, padding - kw);
                                    var endX = Math.Min(outW, width + padding - kw);

                                    if (startX > 0) output.Slice(outIdxY, startX).Clear();

                                    if (endX > startX)
                                    {
                                        var startJ = startX - padding + kw;
                                        var len = endX - startX;
                                        // Bulk Memory Copy - kompilator JIT robi z tego jedną wektorową instrukcję!
                                        input.Slice(inputRowOffset + startJ, len).CopyTo(output.Slice(outIdxY + startX, len));
                                    }

                                    if (endX < outW) output.Slice(outIdxY + endX, outW - endX).Clear();
                                }
                                else
                                {
                                    for (var x = 0; x < outW; x++)
                                    {
                                        var j = x * stride - padding + kw;
                                        output[outIdxY + x] = (j >= 0 && j < width) ? input[inputRowOffset + j] : 0f;
                                    }
                                }
                            }
                            else
                            {
                                output.Slice(outIdxY, outW).Clear();
                            }
                        }
                    }
                }
            }
        }

        public static void Col2Im(ReadOnlySpan<float> colData, int channels, int height, int width, int kSize, int stride, int padding, Span<float> gradInput)
        {
            var outH = (height + 2 * padding - kSize) / stride + 1;
            var outW = (width + 2 * padding - kSize) / stride + 1;
            var channelSize = height * width;

            for (var c = 0; c < channels; c++)
            {
                for (var kh = 0; kh < kSize; kh++)
                {
                    for (var kw = 0; kw < kSize; kw++)
                    {
                        var rowOffset = (c * kSize * kSize + kh * kSize + kw) * outH * outW;
                        for (var y = 0; y < outH; y++)
                        {
                            var i = y * stride - padding + kh;
                            var outIdxY = rowOffset + y * outW;

                            if (i >= 0 && i < height)
                            {
                                var inputRowOffset = c * channelSize + i * width;

                                // FAST PATH: Akumulacja SIMD bez warunków
                                if (stride == 1)
                                {
                                    var startX = Math.Max(0, padding - kw);
                                    var endX = Math.Min(outW, width + padding - kw);

                                    if (endX > startX)
                                    {
                                        var startJ = startX - padding + kw;
                                        var len = endX - startX;

                                        var inSlice = gradInput.Slice(inputRowOffset + startJ, len);
                                        var outSlice = colData.Slice(outIdxY + startX, len);

                                        // Wektoryzowane dodawanie potężnych bloków pamięci!
                                        TensorPrimitives.Add(inSlice, outSlice, inSlice);
                                    }
                                }
                                else
                                {
                                    for (var x = 0; x < outW; x++)
                                    {
                                        var j = x * stride - padding + kw;
                                        if (j >= 0 && j < width)
                                        {
                                            gradInput[inputRowOffset + j] += colData[outIdxY + x];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        public static AutogradNode Linear(AutogradNode input, AutogradNode weights, AutogradNode bias)
        {
            var product = MatMul(input, weights);
            return AddBias(product, bias);
        }
    }
}