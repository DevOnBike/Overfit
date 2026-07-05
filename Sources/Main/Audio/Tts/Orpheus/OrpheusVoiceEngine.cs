// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

using System.Runtime.InteropServices;
using DevOnBike.Overfit.Audio.Tts.Snac;
using DevOnBike.Overfit.Diagnostics;
using DevOnBike.Overfit.LanguageModels;
using DevOnBike.Overfit.LanguageModels.Contracts;

namespace DevOnBike.Overfit.Audio.Tts.Orpheus
{
    /// <summary>
    /// End-to-end Orpheus text-to-speech: an LLM (Llama-3.2-3B arch, loaded as GGUF) predicts SNAC audio tokens,
    /// which <see cref="OrpheusSnacBridge"/> de-interleaves and <see cref="Snac.Snac"/> vocodes to a 24 kHz
    /// waveform — all in pure managed .NET on the CPU. Load once, then <see cref="Synthesize"/> per utterance.
    /// </summary>
    public sealed class OrpheusVoiceEngine : IDisposable
    {
        // Orpheus's end-of-speech marker (in addition to the tokenizer's end-of-text id).
        private const int EndOfSpeechTokenId = 128258;
        private const int DefaultMaxTokens = 1200;

        // ~0.12 s of silence inserted between sentences when synthesizing multi-sentence text.
        private const float InterSentenceSilenceSeconds = 0.12f;

        private readonly OverfitClient _llm;
        private readonly Snac.Snac _snac;
        private readonly bool _ownsLlm;
        private readonly TtsTextNormalizer _normalizer = new();

        private OrpheusVoiceEngine(OverfitClient llm, Snac.Snac snac, bool ownsLlm)
        {
            _llm = llm;
            _snac = snac;
            _ownsLlm = ownsLlm;
        }

        public int SampleRate => _snac.SampleRate;

        /// <summary>
        /// Loads the Orpheus LM from <paramref name="orpheusGgufPath"/> (a Llama-arch GGUF file) and the SNAC
        /// decoder from <paramref name="snacDir"/>. The context is sized for a full audio-token stream.
        /// </summary>
        public static OrpheusVoiceEngine Load(string orpheusGgufPath, string snacDir)
        {
            var llm = OverfitClient.LoadGguf(orpheusGgufPath, maxContextLength: 4096);
            var snac = Snac.Snac.Load(snacDir);
            return new OrpheusVoiceEngine(llm, snac, ownsLlm: true);
        }

        /// <summary>Wraps an already-loaded LM + decoder (the caller keeps ownership of <paramref name="llm"/>).</summary>
        public static OrpheusVoiceEngine FromComponents(OverfitClient llm, Snac.Snac snac)
            => new(llm, snac, ownsLlm: false);

        /// <summary>
        /// Synthesizes <paramref name="text"/> in the preset <paramref name="voice"/> to mono 24 kHz PCM in
        /// <c>[-1, 1]</c>. Sampling matches the Orpheus reference (temperature 0.6, top-p 0.9, repetition penalty
        /// 1.1); <paramref name="seed"/> makes a run reproducible. With <paramref name="normalize"/> (default), text
        /// is run through <see cref="TtsTextNormalizer"/> (numbers/abbreviations/lexicon) and long input is split
        /// into sentences, each synthesized and concatenated with a short silence.
        /// </summary>
        public float[] Synthesize(
            string text, string voice = OrpheusPrompt.DefaultVoice, int maxTokens = DefaultMaxTokens, int seed = 0, bool normalize = true)
        {
            var input = normalize ? _normalizer.Normalize(text) : text;
            var sentences = normalize ? SentenceSplitter.Split(input) : [input];
            if (sentences.Count <= 1)
            {
                return SynthesizeChunk(sentences.Count == 1 ? sentences[0] : input, voice, maxTokens, seed);
            }

            var parts = new List<float[]>(sentences.Count);
            foreach (var sentence in sentences)
            {
                parts.Add(SynthesizeChunk(sentence, voice, maxTokens, seed));
            }
            return Concatenate(parts, (int)(SampleRate * InterSentenceSilenceSeconds));
        }

        private float[] SynthesizeChunk(string text, string voice, int maxTokens, int seed)
        {
            var codes = GenerateAudioCodes(text, voice, maxTokens, seed);
            if (codes.Count < OrpheusSnacBridge.FrameStride)
            {
                throw new OverfitRuntimeException(
                    $"Orpheus produced {codes.Count} audio codes for \"{text}\" — not enough for a frame. The model "
                    + "may not be an Orpheus TTS checkpoint, or generation stopped immediately.");
            }

            var levels = OrpheusSnacBridge.Redistribute(CollectionsMarshal.AsSpan(codes));
            var audio = _snac.Decode(levels);
            // Tail-only trim — trimming the lead clips a sentence's first soft onset (garbles word 1).
            return AudioPostProcessing.TrimSilence(audio, trimLeading: false);
        }

        private static float[] Concatenate(List<float[]> parts, int gapSamples)
        {
            var total = 0;
            for (var p = 0; p < parts.Count; p++)
            {
                total += parts[p].Length;
                if (p < parts.Count - 1)
                {
                    total += gapSamples;
                }
            }

            var output = new float[total];
            var offset = 0;
            for (var p = 0; p < parts.Count; p++)
            {
                parts[p].AsSpan().CopyTo(output.AsSpan(offset));
                offset += parts[p].Length;
                if (p < parts.Count - 1)
                {
                    offset += gapSamples; // already zero-filled
                }
            }
            return output;
        }

        private List<int> GenerateAudioCodes(string text, string voice, int maxTokens, int seed)
        {
            // Canonical id sequence (incl. the audio-priming control tokens) — NOT the bare "<|audio|>…<|eot_id|>"
            // string, which omits start_of_speech and corrupts the first word.
            var promptTokens = OrpheusPrompt.BuildPromptTokens(_llm.Tokenizer, text, voice);

            var sampling = new SamplingOptions(
                SamplingStrategy.TopP, temperature: 0.6f, topK: 0, topP: 0.9f, seed: seed,
                repetitionPenalty: 1.1f, repetitionPenaltyContextSize: 64);

            var eos = _llm.Tokenizer.EndOfTextTokenId;
            using var session = _llm.Engine.CreateSession();
            session.Prefill(promptTokens);

            var genSw = ValueStopwatch.StartNew();
            var generatedCount = 0;
            var codes = new List<int>();
            var accepted = 0;
            for (var step = 0; step < maxTokens; step++)
            {
                var token = session.GenerateNextToken(in sampling);
                generatedCount++;
                if (token == eos || token == EndOfSpeechTokenId)
                {
                    break;
                }

                var piece = _llm.Tokenizer.DecodeToString(MemoryMarshal.CreateReadOnlySpan(ref token, 1));
                if (!OrpheusSnacBridge.TryReadCustomTokenNumber(piece, out var customTokenNumber))
                {
                    continue;
                }

                var code = OrpheusSnacBridge.DecodeCustomToken(customTokenNumber, accepted);
                if (code > 0 && code < 4096)
                {
                    codes.Add(code);
                    accepted++;
                }
            }

            var genSeconds = genSw.GetElapsedTime().TotalSeconds;
            Console.WriteLine($"  gen: {generatedCount} tok in {genSeconds:F1}s = {generatedCount / genSeconds:F1} tok/s (preset / inference engine)");
            return codes;
        }

        public void Dispose()
        {
            if (_ownsLlm)
            {
                _llm.Dispose();
            }
        }
    }
}
