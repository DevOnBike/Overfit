using System.Numerics.Tensors;

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
            var N = input.Data.Shape[0]; var C = input.Data.Shape[1];
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
            int N = input.Data.Shape[0], C = input.Data.Shape[1];
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
            int aRows = A.Shape[0], aCols = A.Shape[1], bCols = B.Shape[1];
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
                for (var k = 0; k < aC; k++)
                {
                    var valA = aS[i * aC + k];
                    if (valA == 0) continue;
                    TensorPrimitives.MultiplyAdd(bS.Slice(k * bC, bC), valA, rowC, rowC);
                }
            }
        }

        public static void MatMulBackward(AutogradNode a, AutogradNode b, AutogradNode output)
        {
            if (a.RequiresGrad)
            {
                using var bT = b.Data.Transpose(0, 1);
                using var bTContig = bT.ToContiguous();
                using var gradA = MatMulRaw(output.Grad, bTContig);
                TensorPrimitives.Add(a.Grad.AsSpan(), gradA.AsSpan(), a.Grad.AsSpan());
            }
            if (b.RequiresGrad)
            {
                using var aT = a.Data.Transpose(0, 1);
                using var aTContig = aT.ToContiguous();
                using var gradB = MatMulRaw(aTContig, output.Grad);
                TensorPrimitives.Add(b.Grad.AsSpan(), gradB.AsSpan(), b.Grad.AsSpan());
            }
        }

        // ====================================================================
        // 2. CNN - CONV, POOL, GAP (NCHW)
        // ====================================================================

        public static AutogradNode Conv2D(AutogradNode input, AutogradNode weights, int inC, int outC, int h, int w, int k)
        {
            int outH = h - k + 1, outW = w - k + 1, batchSize = input.Data.Shape[0], kSqInC = k * k * inC;
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

            int outH = h - k + 1, outW = w - k + 1, batchSize = input.Data.Shape[0], kSqInC = k * k * inC;
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
            int outputH = inputH / poolSize, outputW = inputW / poolSize, batchSize = input.Data.Shape[0];
            var resultData = new FastTensor<float>(batchSize, channels, outputH, outputW);
            var maxIndices = new AutogradNode(new FastTensor<float>(batchSize, channels, outputH, outputW), false);

            Parallel.For(0, batchSize, n => {
                for (var c = 0; c < channels; c++)
                {
                    for (var oh = 0; oh < outputH; oh++)
                    {
                        for (var ow = 0; ow < outputW; ow++)
                        {
                            var maxVal = float.MinValue; var maxIdx = -1;
                            for (var ph = 0; ph < poolSize; ph++)
                            {
                                for (var pw = 0; pw < poolSize; pw++)
                                {
                                    int ih = oh * poolSize + ph, iw = ow * poolSize + pw;
                                    var val = input.Data[n, c, ih, iw];
                                    if (val > maxVal) { maxVal = val; maxIdx = n * (channels * inputH * inputW) + c * (inputH * inputW) + ih * inputW + iw; }
                                }
                            }
                            resultData[n, c, oh, ow] = maxVal;
                            maxIndices.Data[n, c, oh, ow] = maxIdx;
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
            var iGS = input.Grad.AsSpan();
            var oGS = output.Grad.AsSpan();
            var idxS = maxIndices.Data.AsSpan();
            for (var i = 0; i < idxS.Length; i++) iGS[(int)idxS[i]] += oGS[i];
        }

        public static AutogradNode GlobalAveragePool2D(AutogradNode input, int channels, int h, int w)
        {
            var batchSize = input.Data.Shape[0];
            var resData = new FastTensor<float>(batchSize, channels);
            float spatialSize = h * w;

            Parallel.For(0, batchSize, n => {
                for (var c = 0; c < channels; c++)
                    resData[n, c] = TensorPrimitives.Sum(input.Data.AsSpan().Slice(n * channels * h * w + c * h * w, h * w)) / spatialSize;
            });

            var output = new AutogradNode(resData, input.RequiresGrad);
            if (output.RequiresGrad) ComputationGraph.Active.Record(OpCode.GlobalAveragePool2D, output, input, null, h, w, channels);
            return output;
        }

        public static void GlobalAvgPool2DBackward(AutogradNode input, AutogradNode output, int h, int w, int channels)
        {
            var batchSize = input.Data.Shape[0]; float spatialSize = h * w;
            Parallel.For(0, batchSize, n => {
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
            var inS = input.Data.AsSpan(); var goS = output.Grad.AsSpan(); var giS = input.Grad.AsSpan();
            for (var i = 0; i < inS.Length; i++) if (inS[i] > 0) giS[i] += goS[i];
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
            else input.Data.AsSpan().CopyTo(resData.AsSpan());

            var output = new AutogradNode(resData, input.RequiresGrad);
            if (output.RequiresGrad && isTraining) ComputationGraph.Active.Record(OpCode.Dropout, output, input, mask);
            return output;
        }

        public static void DropoutBackward(AutogradNode input, AutogradNode mask, AutogradNode output)
        {
            var giS = input.Grad.AsSpan(); var goS = output.Grad.AsSpan(); var mS = mask.Data.AsSpan();
            for (var i = 0; i < giS.Length; i++) giS[i] += goS[i] * mS[i];
        }

        // ====================================================================
        // 4. FUNKCJE STRATY (LOSS)
        // ====================================================================

        public static AutogradNode SoftmaxCrossEntropy(AutogradNode logits, AutogradNode target)
        {
            int rows = logits.Data.Shape[0], cols = logits.Data.Shape[1];
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
            int rows = logits.Data.Shape[0], cols = logits.Data.Shape[1];
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
            var factor = o.Grad[0, 0] * (2f / p.Data.Size);
            for (var i = 0; i < p.Data.Size; i++) p.Grad.AsSpan()[i] += (p.Data.AsSpan()[i] - t.Data.AsSpan()[i]) * factor;
        }

        // ====================================================================
        // 5. BATCH NORM I NARZĘDZIA
        // ====================================================================

        public static AutogradNode BatchNorm1D(AutogradNode input, AutogradNode gamma, AutogradNode beta, FastTensor<float> runningMean, FastTensor<float> runningVar, float momentum, float eps, bool isTraining)
        {
            int N = input.Data.Shape[0], C = input.Data.Shape[1];
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

        public static void BatchNorm1DBackward(AutogradNode input, AutogradNode output, AutogradNode gamma, AutogradNode beta, AutogradNode mean, AutogradNode invStd)
        {
            if (!input.RequiresGrad && !gamma.RequiresGrad && !beta.RequiresGrad) return;

            int N = input.Data.Shape[0], C = input.Data.Shape[1];
            var inS = input.Data.AsSpan();
            var outGradS = output.Grad.AsSpan();
            var inGradS = input.RequiresGrad ? input.Grad.AsSpan() : default;
            var gammaS = gamma.Data.AsSpan();
            var meanS = mean.Data.AsSpan();
            var invStdS = invStd.Data.AsSpan();

            // 1. Gradienty dla Gamma i Beta
            if (gamma.RequiresGrad || beta.RequiresGrad)
            {
                var dGammaS = gamma.Grad.AsSpan();
                var dBetaS = beta.Grad.AsSpan();
                for (var i = 0; i < N; i++)
                {
                    for (var c = 0; c < C; c++)
                    {
                        var go = outGradS[i * C + c];
                        if (beta.RequiresGrad) dBetaS[c] += go;
                        if (gamma.RequiresGrad) dGammaS[c] += go * (inS[i * C + c] - meanS[c]) * invStdS[c];
                    }
                }
            }

            // 2. Gradient wejścia (skomplikowana matematyka wyrównania statystycznego)
            if (input.RequiresGrad)
            {
                Span<float> sumDy = stackalloc float[C];
                Span<float> sumDyXHat = stackalloc float[C];

                for (var i = 0; i < N; i++)
                {
                    for (var c = 0; c < C; c++)
                    {
                        var go = outGradS[i * C + c];
                        var xHat = (inS[i * C + c] - meanS[c]) * invStdS[c];
                        sumDy[c] += go;
                        sumDyXHat[c] += go * xHat;
                    }
                }

                for (var i = 0; i < N; i++)
                {
                    for (var c = 0; c < C; c++)
                    {
                        var go = outGradS[i * C + c];
                        var xHat = (inS[i * C + c] - meanS[c]) * invStdS[c];
                        var dx = (gammaS[c] * invStdS[c] / N) * (N * go - sumDy[c] - xHat * sumDyXHat[c]);
                        inGradS[i * C + c] += dx;
                    }
                }
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
                            var outIdxY = rowOffset + y * outW; // Hoisting 1

                            if (i >= 0 && i < height)
                            {
                                var inputRowOffset = c * channelSize + i * width; // Hoisting 2

                                for (var x = 0; x < outW; x++)
                                {
                                    var j = x * stride - padding + kw;
                                    // W samym sercu pętli mamy już tylko tanie dodawanie indeksów
                                    output[outIdxY + x] = (j >= 0 && j < width) ? input[inputRowOffset + j] : 0f;
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
                            var outIdxY = rowOffset + y * outW; // Hoisting 1

                            if (i >= 0 && i < height)
                            {
                                var inputRowOffset = c * channelSize + i * width; // Hoisting 2

                                for (var x = 0; x < outW; x++)
                                {
                                    var j = x * stride - padding + kw;
                                    if (j >= 0 && j < width)
                                    {
                                        // Ponownie - minimalizujemy operacje w środku pętli
                                        gradInput[inputRowOffset + j] += colData[outIdxY + x];
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
            // Linear to operacja: Y = X * W + B
            // Wykorzystujemy istniejące i zoptymalizowane metody MatMul oraz AddBias
            var product = MatMul(input, weights);

            return AddBias(product, bias);
        }
    }
}