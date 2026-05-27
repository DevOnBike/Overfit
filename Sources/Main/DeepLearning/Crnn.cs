// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using DevOnBike.Overfit.Autograd;
using DevOnBike.Overfit.DeepLearning.Abstractions;
using DevOnBike.Overfit.Ops;
using DevOnBike.Overfit.Tensors;
using DevOnBike.Overfit.Tensors.Core;

namespace DevOnBike.Overfit.DeepLearning
{
    /// <summary>
    /// A ready-to-use <b>CRNN</b> (Convolutional Recurrent Neural Network) for sequence recognition —
    /// the standard OCR / handwriting / spectrogram-labelling architecture, here as a single
    /// <see cref="IModule"/> facade so a developer wires it with three numbers:
    /// <c>new Crnn(imageHeight, imageWidth, classCount)</c>.
    ///
    /// <para>Pipeline: a 2-D <see cref="ConvLayer"/> over the (single-channel) image →
    /// <c>map-to-sequence</c> (reshape + <see cref="ComputationGraph.TransposeLastTwo"/> turns the conv
    /// feature map into a width-indexed sequence whose per-step feature is <c>channels × convHeight</c>) →
    /// an <see cref="LstmLayer"/> reading that sequence left-to-right → a per-step
    /// <see cref="LinearLayer"/> producing <see cref="ClassCount"/> logits per timestep. Train the
    /// <c>[<see cref="TimeSteps"/> × ClassCount]</c> logits with <see cref="CtcLoss"/> (no image↔label
    /// alignment needed); read text back with <see cref="Recognize(ReadOnlySpan{float})"/>.</para>
    ///
    /// <para>Scope: single image per <see cref="Forward"/> call (batch = 1 — CTC is per-sequence; batch by
    /// looping or by replicas). The image width is fixed at construction — pad variable-width inputs to it.
    /// The convolution is VALID (output width = <c>imageWidth − kernelSize + 1</c> = <see cref="TimeSteps"/>),
    /// so keep <c>imageWidth</c> comfortably larger than <c>2·maxLabelLength + 1</c> for CTC.</para>
    /// </summary>
    public sealed class Crnn : IModule
    {
        private readonly ConvLayer[] _convs;
        private readonly LstmLayer _lstm;
        private readonly LinearLayer _classifier;

        private readonly int _imageHeight;
        private readonly int _imageWidth;
        private readonly int _classCount;
        private readonly int _blankIndex;
        private readonly int _convChannels;
        private readonly int _lstmHidden;
        private readonly int _outHeight;
        private readonly int _outWidth;
        private readonly int _seqFeatures;
        private readonly int _inferenceArenaElements;

        private ComputationGraph? _inferenceGraph;
        private bool _isTraining = true;

        /// <param name="imageHeight">Input image height in pixels (single channel).</param>
        /// <param name="imageWidth">Input image width in pixels — fixed; pad inputs to this width.</param>
        /// <param name="classCount">Number of output classes <b>including</b> the CTC blank.</param>
        /// <param name="convChannels">Conv feature maps per layer (default 16).</param>
        /// <param name="kernelSize">Square conv kernel — must be <b>odd</b> (default 3) so SAME padding
        /// preserves spatial size, and ≤ height/width.</param>
        /// <param name="lstmHidden">LSTM hidden size (default 64).</param>
        /// <param name="convLayers">Number of stacked SAME conv+ReLU layers in the front-end (default 2).</param>
        /// <param name="blankIndex">CTC blank class; defaults to <c>classCount − 1</c>.</param>
        /// <param name="inferenceArenaElements">
        /// Tape arena (floats) for the lazily-created internal graph used by
        /// <see cref="Recognize(ReadOnlySpan{float})"/> / <see cref="ForwardInference"/> (default 4M).
        /// </param>
        public Crnn(
            int imageHeight,
            int imageWidth,
            int classCount,
            int convChannels = 16,
            int kernelSize = 3,
            int lstmHidden = 64,
            int convLayers = 2,
            int blankIndex = -1,
            int inferenceArenaElements = 4_000_000)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(imageHeight);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(imageWidth);
            ArgumentOutOfRangeException.ThrowIfLessThan(classCount, 2);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(convChannels);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kernelSize);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(lstmHidden);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(convLayers);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inferenceArenaElements);

            if (kernelSize % 2 == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(kernelSize), "kernelSize must be odd for SAME padding.");
            }
            if (kernelSize > imageHeight || kernelSize > imageWidth)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(kernelSize), $"kernelSize {kernelSize} must be ≤ imageHeight ({imageHeight}) and imageWidth ({imageWidth}).");
            }

            _blankIndex = blankIndex < 0 ? classCount - 1 : blankIndex;
            if ((uint)_blankIndex >= (uint)classCount)
            {
                throw new ArgumentOutOfRangeException(nameof(blankIndex), $"blankIndex must be in [0,{classCount}).");
            }

            _imageHeight = imageHeight;
            _imageWidth = imageWidth;
            _classCount = classCount;
            _convChannels = convChannels;
            _lstmHidden = lstmHidden;

            // SAME conv stack (padding = kSize/2, stride 1, odd kernel) → spatial dims preserved, so the
            // time axis (= width) is kept at full length for CTC and conv layers can stack freely.
            var padding = kernelSize / 2;
            _outHeight = imageHeight;
            _outWidth = imageWidth;                       // = TimeSteps (CTC T) — full width, no shrink
            _seqFeatures = convChannels * _outHeight;
            _inferenceArenaElements = inferenceArenaElements;

            _convs = new ConvLayer[convLayers];
            for (var i = 0; i < convLayers; i++)
            {
                var inChannels = i == 0 ? 1 : convChannels;
                _convs[i] = new ConvLayer(
                    inChannels, convChannels, h: imageHeight, w: imageWidth, kSize: kernelSize,
                    padding: padding, stride: 1, useBias: true);
            }

            _lstm = new LstmLayer(_seqFeatures, lstmHidden, returnSequences: true);
            _classifier = new LinearLayer(lstmHidden, classCount);
        }

        /// <summary>Number of output timesteps = CTC sequence length T (= <c>imageWidth</c>, SAME conv).</summary>
        public int TimeSteps => _outWidth;

        /// <summary>Output classes including the blank.</summary>
        public int ClassCount => _classCount;

        /// <summary>The CTC blank class index.</summary>
        public int BlankIndex => _blankIndex;

        /// <summary>Fixed input image height.</summary>
        public int ImageHeight => _imageHeight;

        /// <summary>Fixed input image width.</summary>
        public int ImageWidth => _imageWidth;

        public bool IsTraining => _isTraining;

        /// <summary>
        /// Forward pass for one image. <paramref name="input"/> must be a single
        /// <c>[1, 1, imageHeight, imageWidth]</c> tensor (build one with <see cref="CreateInput"/>).
        /// Returns the per-timestep logits as <c>[<see cref="TimeSteps"/>, <see cref="ClassCount"/>]</c>.
        /// </summary>
        public AutogradNode Forward(ComputationGraph graph, AutogradNode input)
        {
            ArgumentNullException.ThrowIfNull(graph);
            ArgumentNullException.ThrowIfNull(input);

            var feat = input;                                          // [1, 1, H, W]
            foreach (var conv in _convs)
            {
                feat = graph.Relu(conv.Forward(graph, feat));          // SAME → [1, C, H, W]
            }
            var merged = graph.Reshape(feat, 1, _seqFeatures, _outWidth);   // [1, C*outH, outW]
            var seqIn = graph.TransposeLastTwo(merged);               // [1, outW, C*outH]
            var seq = _lstm.Forward(graph, seqIn);                    // [1, outW, lstmHidden]
            var seq2 = graph.Reshape(seq, _outWidth, _lstmHidden);    // [outW, lstmHidden]
            return _classifier.Forward(graph, seq2);                  // [outW, classCount]
        }

        /// <summary>
        /// Wraps <see cref="CtcLoss"/> with this model's T / classCount / blank: computes the loss for
        /// <paramref name="target"/> and writes the gradient seed into <paramref name="logits"/>'s grad
        /// buffer (call <c>graph.BackwardFromGrad(logits)</c> afterwards). <paramref name="logits"/> must
        /// be a <see cref="Forward"/> result.
        /// </summary>
        public float ComputeCtcLoss(AutogradNode logits, ReadOnlySpan<int> target)
        {
            ArgumentNullException.ThrowIfNull(logits);
            return CtcLoss.Forward(
                logits.DataView.AsReadOnlySpan(), _outWidth, _classCount, target, _blankIndex, logits.GradView.AsSpan());
        }

        /// <summary>
        /// Builds the <c>[1, 1, imageHeight, imageWidth]</c> input node from a row-major image
        /// (length <c>imageHeight·imageWidth</c>). The caller owns it (dispose after the forward — and,
        /// when training, after <c>BackwardFromGrad</c>, since the conv backward reads the input).
        /// </summary>
        public AutogradNode CreateInput(ReadOnlySpan<float> image)
        {
            var expected = _imageHeight * _imageWidth;
            if (image.Length != expected)
            {
                throw new ArgumentException($"image length {image.Length} != imageHeight*imageWidth ({expected}).", nameof(image));
            }

            var storage = new TensorStorage<float>(expected, clearMemory: false);
            image.CopyTo(storage.AsSpan());
            return new AutogradNode(storage, new TensorShape(1, 1, _imageHeight, _imageWidth), requiresGrad: false);
        }

        /// <summary>
        /// Recognises the text in <paramref name="image"/> using a caller-owned graph (reset internally):
        /// forward + greedy CTC decode. Returns the predicted label sequence (blank-collapsed).
        /// </summary>
        /// <param name="beamWidth">
        /// 0 or 1 ⇒ greedy best-path decode (fast). &gt; 1 ⇒ CTC prefix beam search of that width
        /// (best-labeling; a small accuracy gain on ambiguous input).
        /// </param>
        public int[] Recognize(ComputationGraph graph, ReadOnlySpan<float> image, int beamWidth = 0)
        {
            ArgumentNullException.ThrowIfNull(graph);
            graph.Reset();
            InvalidateParameterCaches();
            using var input = CreateInput(image);
            var logits = Forward(graph, input);
            var scores = logits.DataView.AsReadOnlySpan();
            return beamWidth > 1
                ? CtcDecoder.BeamSearchDecode(scores, _outWidth, _classCount, _blankIndex, beamWidth)
                : CtcDecoder.GreedyDecode(scores, _outWidth, _classCount, _blankIndex);
        }

        /// <summary>
        /// Recognises with <b>language-model-rescored</b> beam search (best labeling, steered by
        /// <paramref name="languageModel"/>). Uses a caller-owned graph (reset internally).
        /// </summary>
        public int[] Recognize(
            ComputationGraph graph, ReadOnlySpan<float> image, int beamWidth,
            ICtcLanguageModel languageModel, double languageModelWeight = 1.0, double insertionBonus = 0.0)
        {
            ArgumentNullException.ThrowIfNull(graph);
            ArgumentNullException.ThrowIfNull(languageModel);
            graph.Reset();
            InvalidateParameterCaches();
            using var input = CreateInput(image);
            var logits = Forward(graph, input);
            return CtcDecoder.BeamSearchDecode(
                logits.DataView.AsReadOnlySpan(), _outWidth, _classCount, _blankIndex,
                Math.Max(2, beamWidth), languageModel, languageModelWeight, insertionBonus);
        }

        /// <summary>
        /// Convenience: recognise using the model's own lazily-created inference graph (no graph
        /// management for the caller). Not thread-safe; for concurrent recognition use the
        /// <see cref="Recognize(ComputationGraph, ReadOnlySpan{float})"/> overload with per-thread graphs.
        /// </summary>
        public int[] Recognize(ReadOnlySpan<float> image, int beamWidth = 0)
        {
            _inferenceGraph ??= new ComputationGraph(_inferenceArenaElements);
            return Recognize(_inferenceGraph, image, beamWidth);
        }

        /// <summary>
        /// <see cref="IModule"/> inference shim — runs <see cref="Forward"/> on the internal graph and
        /// copies the <c>[TimeSteps × ClassCount]</c> logits into <paramref name="output"/>.
        /// </summary>
        public void ForwardInference(ReadOnlySpan<float> input, Span<float> output)
        {
            var needed = _outWidth * _classCount;
            if (output.Length < needed)
            {
                throw new ArgumentException($"output length {output.Length} < TimeSteps*ClassCount ({needed}).", nameof(output));
            }

            _inferenceGraph ??= new ComputationGraph(_inferenceArenaElements);
            _inferenceGraph.Reset();
            InvalidateParameterCaches();
            using var node = CreateInput(input);
            var logits = Forward(_inferenceGraph, node);
            logits.DataView.AsReadOnlySpan().Slice(0, needed).CopyTo(output);
        }

        public IEnumerable<AutogradNode> Parameters()
        {
            foreach (var conv in _convs)
            {
                foreach (var p in conv.Parameters()) { yield return p; }
            }
            foreach (var p in _lstm.Parameters()) { yield return p; }
            foreach (var p in _classifier.Parameters()) { yield return p; }
        }

        public void Train()
        {
            _isTraining = true;
            foreach (var conv in _convs) { conv.Train(); }
            _lstm.Train();
            _classifier.Train();
        }

        public void Eval()
        {
            _isTraining = false;
            foreach (var conv in _convs) { conv.Eval(); }
            _lstm.Eval();
            _classifier.Eval();
        }

        public void InvalidateParameterCaches()
        {
            foreach (var conv in _convs) { conv.InvalidateParameterCaches(); }
            _lstm.InvalidateParameterCaches();
            _classifier.InvalidateParameterCaches();
        }

        public void Save(BinaryWriter bw)
        {
            ArgumentNullException.ThrowIfNull(bw);
            foreach (var conv in _convs) { conv.Save(bw); }
            _lstm.Save(bw);
            _classifier.Save(bw);
        }

        public void Load(BinaryReader br)
        {
            ArgumentNullException.ThrowIfNull(br);
            foreach (var conv in _convs) { conv.Load(br); }
            _lstm.Load(br);
            _classifier.Load(br);
        }

        public void Dispose()
        {
            _inferenceGraph?.Dispose();
            foreach (var conv in _convs) { conv.Dispose(); }
            _lstm.Dispose();
            _classifier.Dispose();
        }
    }
}
