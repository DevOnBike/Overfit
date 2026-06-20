// Copyright (c) 2026 DevOnBike.
// This file is part of DevonBike Overfit.
// DevonBike Overfit is licensed under the GNU AGPLv3.
// For commercial licensing options, contact: devonbike@gmail.com

namespace DevOnBike.Overfit.Audio.Mp3
{
    /// <summary>
    /// MPEG-1/2/2.5 Layer III audio decoder. The DSP pipeline (bit reservoir → side info → scalefactors →
    /// Huffman → requantize → reorder → stereo → antialias → IMDCT/overlap-add → polyphase subband synthesis)
    /// is a faithful port of the public-domain pdmp3 reference (Krister Lagerström) for MPEG-1, extended with
    /// the MPEG-2/2.5 LSF side-info + scalefactor scheme. Output is mono 32-bit float in [-1, 1].
    /// Allocation policy: all per-frame working buffers live in pre-allocated instance fields — the only
    /// allocation in <see cref="DecodeMono"/> is the single output buffer (sized once from the frame probe).
    /// </summary>
    internal sealed class Mp3Decoder
    {
        private const float InvSqrt2 = 0.70710678118654752f;

        private static readonly float[] IsRatios = { 0.0f, 0.267949f, 0.577350f, 1.0f, 1.732051f, 3.732051f };
        private static readonly int[] Pretab = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 2 };

        // MPEG-1 scalefactor sizes [scalefac_compress][slen1, slen2].
        private static readonly int[] ScfSizeSlen1 = { 0, 0, 0, 0, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4 };
        private static readonly int[] ScfSizeSlen2 = { 0, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 2, 3 };

        // MPEG-2 LSF scalefactor-band counts: [tindex 0..5][blocktype 0=long,1=short,2=mixed][partition 0..3].
        private static readonly int[] NrSfb =
        {
            6, 5, 5, 5,   9, 9, 9, 9,    6, 9, 9, 9,
            6, 5, 7, 3,   9, 9, 12, 6,   6, 9, 12, 6,
            11, 10, 0, 0, 18, 18, 0, 0,  15, 18, 0, 0,
            7, 7, 7, 0,   12, 12, 12, 0, 6, 15, 12, 0,
            6, 6, 6, 3,   12, 9, 9, 6,   6, 12, 9, 6,
            8, 8, 5, 0,   15, 12, 9, 0,  6, 18, 9, 0,
        };

        // ── per-frame state (pre-allocated; reused every frame) ──
        private readonly float[] _is = new float[2 * 2 * 576];          // [(gr*2+ch)*576 + line]
        private readonly int[] _scalefacL = new int[2 * 2 * 23];        // [(gr*2+ch)*23 + sfb]
        private readonly int[] _scalefacS = new int[2 * 2 * 13 * 3];    // [((gr*2+ch)*13 + sfb)*3 + win]

        // side info (indexed by gr*2+ch unless noted)
        private int _mainDataBegin;
        private readonly int[] _part23 = new int[4];
        private readonly int[] _bigValues = new int[4];
        private readonly int[] _globalGain = new int[4];
        private readonly int[] _scfCompress = new int[4];
        private readonly int[] _winSwitch = new int[4];
        private readonly int[] _blockType = new int[4];
        private readonly int[] _mixedBlock = new int[4];
        private readonly int[] _tableSelect = new int[4 * 3];
        private readonly int[] _subblockGain = new int[4 * 3];
        private readonly int[] _region0 = new int[4];
        private readonly int[] _region1 = new int[4];
        private readonly int[] _preflag = new int[4];
        private readonly int[] _scalefacScale = new int[4];
        private readonly int[] _count1TableSelect = new int[4];
        private readonly int[] _count1 = new int[4];
        private readonly int[] _scfsi = new int[2 * 4]; // [ch*4 + band] (MPEG-1)

        // bit reservoir
        private readonly byte[] _reservoir = new byte[16384];
        private int _resTop;

        // hybrid synthesis overlap store + subband synthesis V buffer
        private readonly float[] _store = new float[2 * 32 * 18]; // [ch][sb][i]
        private bool _hsynthInit = true;
        private readonly float[] _vVec = new float[2 * 1024];     // [ch][1024]
        private bool _synthInit = true;

        // scratch
        private readonly float[] _imdctOut = new float[36];
        private readonly float[] _reorder = new float[576];
        private readonly float[] _uVec = new float[512];
        private readonly float[] _sVec = new float[32];
        private readonly float[] _pcm0 = new float[576];
        private readonly float[] _pcm1 = new float[576];

        // current-frame header context
        private MpegVersion _version;
        private bool _isMpeg1;
        private int _sfIndex; // 0..8 into the 9-rate band tables
        private int _modeExt;
        private bool _joint;
        private int _nch;

        private static int GC(int gr, int ch) => gr * 2 + ch;

        public float[] DecodeMono(byte[] bytes, out int sampleRate)
        {
            var info = Mp3Reader.Probe(bytes);
            sampleRate = info.SampleRate;
            // OVERFIT001: by-contract — the decoded PCM is the return value; per-frame scratch is pre-allocated
            // instance state (validated ~zero per-decode overhead by Mp3ReaderTests.Decode_PerFrame_ZeroAlloc).
#pragma warning disable OVERFIT001
            var output = new float[info.SampleCount > 0 ? info.SampleCount : 0];
#pragma warning restore OVERFIT001
            if (output.Length == 0)
            {
                return output;
            }

            _resTop = 0;
            _hsynthInit = true;
            _synthInit = true;
            _store.AsSpan().Clear();
            _vVec.AsSpan().Clear();

            var pos = Mp3Reader.SkipId3(bytes);
            var writePos = 0;
            while (pos + 4 <= bytes.Length)
            {
                if (!Mp3FrameHeader.TryParse(bytes, pos, out var h))
                {
                    pos++;
                    continue;
                }
                var len = h.FrameLengthBytes;
                if (len < 4 || pos + len > bytes.Length)
                {
                    break;
                }
                writePos = DecodeFrame(bytes, pos, len, h, output, writePos);
                pos += len;
            }
            return output;
        }

        private int DecodeFrame(byte[] bytes, int pos, int len, in Mp3FrameHeader h, Span<float> output, int writePos)
        {
            _version = h.Version;
            _isMpeg1 = h.IsMpeg1;
            _sfIndex = Mp3Tables.SfBandTableIndex(h.Version, h.SampleRateIndex);
            _modeExt = h.ModeExtension;
            _joint = h.Mode == ChannelMode.JointStereo;
            _nch = h.Channels;
            var nGran = _isMpeg1 ? 2 : 1;

            var crc = h.CrcProtected ? 2 : 0;
            var sideStart = pos + 4 + crc;
            var mainStart = sideStart + h.SideInfoBytes;
            var mainSize = len - 4 - crc - h.SideInfoBytes;
            if (mainSize < 0)
            {
                return writePos;
            }

            var sbr = new Mp3BitReader(bytes, sideStart);
            ParseSideInfo(ref sbr, nGran);

            if (_mainDataBegin > _resTop)
            {
                // Not enough carry-over yet (stream start): stash this frame's main data, emit silence.
                AppendReservoir(bytes, mainStart, mainSize);
                return EmitSilence(output, writePos, nGran * 576);
            }
            AssembleReservoir(bytes, mainStart, mainSize, _mainDataBegin);

            var mbr = new Mp3BitReader(_reservoir, 0);
            ReadMainData(ref mbr, nGran);

            for (var gr = 0; gr < nGran; gr++)
            {
                for (var ch = 0; ch < _nch; ch++)
                {
                    Requantize(gr, ch);
                    Reorder(gr, ch);
                }
                Stereo(gr);
                for (var ch = 0; ch < _nch; ch++)
                {
                    Antialias(gr, ch);
                    HybridSynthesis(gr, ch);
                    FrequencyInversion(gr, ch);
                    SubbandSynthesis(gr, ch, ch == 0 ? _pcm0 : _pcm1);
                }
                for (var i = 0; i < 576 && writePos < output.Length; i++)
                {
                    output[writePos++] = _nch == 1 ? _pcm0[i] : 0.5f * (_pcm0[i] + _pcm1[i]);
                }
            }
            return writePos;
        }

        private static int EmitSilence(Span<float> output, int writePos, int count)
        {
            for (var i = 0; i < count && writePos < output.Length; i++)
            {
                output[writePos++] = 0f;
            }
            return writePos;
        }

        // ── bit reservoir assembly ──
        private void AppendReservoir(byte[] bytes, int mainStart, int mainSize)
        {
            if (_resTop + mainSize > _reservoir.Length)
            {
                _resTop = 0; // overflow guard (should not happen on valid streams)
            }
            bytes.AsSpan(mainStart, mainSize).CopyTo(_reservoir.AsSpan(_resTop));
            _resTop += mainSize;
        }

        private void AssembleReservoir(byte[] bytes, int mainStart, int mainSize, int begin)
        {
            // Move the last `begin` bytes of the reservoir to the front, then append this frame's main data.
            _reservoir.AsSpan(_resTop - begin, begin).CopyTo(_reservoir.AsSpan(0, begin));
            if (begin + mainSize > _reservoir.Length)
            {
                mainSize = _reservoir.Length - begin;
            }
            bytes.AsSpan(mainStart, mainSize).CopyTo(_reservoir.AsSpan(begin, mainSize));
            _resTop = begin + mainSize;
        }

        // ── side info ──
        private void ParseSideInfo(ref Mp3BitReader br, int nGran)
        {
            if (_isMpeg1)
            {
                _mainDataBegin = (int)br.ReadBits(9);
                br.ReadBits(_nch == 1 ? 5 : 3); // private bits
                for (var ch = 0; ch < _nch; ch++)
                {
                    for (var band = 0; band < 4; band++)
                    {
                        _scfsi[ch * 4 + band] = (int)br.ReadBits(1);
                    }
                }
            }
            else
            {
                _mainDataBegin = (int)br.ReadBits(8);
                br.ReadBits(_nch == 1 ? 1 : 2); // private bits
            }

            for (var gr = 0; gr < nGran; gr++)
            {
                for (var ch = 0; ch < _nch; ch++)
                {
                    var g = GC(gr, ch);
                    _part23[g] = (int)br.ReadBits(12);
                    _bigValues[g] = (int)br.ReadBits(9);
                    _globalGain[g] = (int)br.ReadBits(8);
                    _scfCompress[g] = (int)br.ReadBits(_isMpeg1 ? 4 : 9);
                    _winSwitch[g] = (int)br.ReadBits(1);
                    if (_winSwitch[g] == 1)
                    {
                        _blockType[g] = (int)br.ReadBits(2);
                        _mixedBlock[g] = (int)br.ReadBits(1);
                        for (var r = 0; r < 2; r++)
                        {
                            _tableSelect[g * 3 + r] = (int)br.ReadBits(5);
                        }
                        _tableSelect[g * 3 + 2] = 0;
                        for (var w = 0; w < 3; w++)
                        {
                            _subblockGain[g * 3 + w] = (int)br.ReadBits(3);
                        }
                        _region0[g] = _blockType[g] == 2 && _mixedBlock[g] == 0 ? 8 : 7;
                        _region1[g] = 20 - _region0[g];
                    }
                    else
                    {
                        for (var r = 0; r < 3; r++)
                        {
                            _tableSelect[g * 3 + r] = (int)br.ReadBits(5);
                        }
                        _region0[g] = (int)br.ReadBits(4);
                        _region1[g] = (int)br.ReadBits(3);
                        _blockType[g] = 0;
                        _mixedBlock[g] = 0;
                        _subblockGain[g * 3] = _subblockGain[g * 3 + 1] = _subblockGain[g * 3 + 2] = 0;
                    }
                    _preflag[g] = _isMpeg1 ? (int)br.ReadBits(1) : 0;
                    _scalefacScale[g] = (int)br.ReadBits(1);
                    _count1TableSelect[g] = (int)br.ReadBits(1);
                }
            }
        }

        // ── main data: scalefactors + Huffman ──
        private void ReadMainData(ref Mp3BitReader br, int nGran)
        {
            for (var gr = 0; gr < nGran; gr++)
            {
                for (var ch = 0; ch < _nch; ch++)
                {
                    var part2Start = br.BitPosition;
                    if (_isMpeg1)
                    {
                        ReadScaleFactorsMpeg1(ref br, gr, ch);
                    }
                    else
                    {
                        ReadScaleFactorsLsf(ref br, gr, ch);
                    }
                    ReadHuffman(ref br, part2Start, gr, ch);
                }
            }
        }

        private void ReadScaleFactorsMpeg1(ref Mp3BitReader br, int gr, int ch)
        {
            var g = GC(gr, ch);
            var slen1 = ScfSizeSlen1[_scfCompress[g]];
            var slen2 = ScfSizeSlen2[_scfCompress[g]];
            var lBase = g * 23;

            if (_winSwitch[g] != 0 && _blockType[g] == 2)
            {
                if (_mixedBlock[g] != 0)
                {
                    for (var sfb = 0; sfb < 8; sfb++)
                    {
                        _scalefacL[lBase + sfb] = (int)br.ReadBits(slen1);
                    }
                    for (var sfb = 3; sfb < 12; sfb++)
                    {
                        var nbits = sfb < 6 ? slen1 : slen2;
                        for (var w = 0; w < 3; w++)
                        {
                            _scalefacS[(g * 13 + sfb) * 3 + w] = (int)br.ReadBits(nbits);
                        }
                    }
                }
                else
                {
                    for (var sfb = 0; sfb < 12; sfb++)
                    {
                        var nbits = sfb < 6 ? slen1 : slen2;
                        for (var w = 0; w < 3; w++)
                        {
                            _scalefacS[(g * 13 + sfb) * 3 + w] = (int)br.ReadBits(nbits);
                        }
                    }
                }
            }
            else
            {
                // long blocks, with scfsi sharing between granule 0 and 1
                ReadLongBand(ref br, gr, ch, 0, 6, slen1, 0);
                ReadLongBand(ref br, gr, ch, 6, 11, slen1, 1);
                ReadLongBand(ref br, gr, ch, 11, 16, slen2, 2);
                ReadLongBand(ref br, gr, ch, 16, 21, slen2, 3);
            }
        }

        private void ReadLongBand(ref Mp3BitReader br, int gr, int ch, int from, int to, int slen, int scfsiBand)
        {
            var g = GC(gr, ch);
            if (_scfsi[ch * 4 + scfsiBand] == 0 || gr == 0)
            {
                for (var sfb = from; sfb < to; sfb++)
                {
                    _scalefacL[g * 23 + sfb] = (int)br.ReadBits(slen);
                }
            }
            else
            {
                var g0 = GC(0, ch);
                for (var sfb = from; sfb < to; sfb++)
                {
                    _scalefacL[g * 23 + sfb] = _scalefacL[g0 * 23 + sfb];
                }
            }
        }

        private void ReadScaleFactorsLsf(ref Mp3BitReader br, int gr, int ch)
        {
            var g = GC(gr, ch);
            var sfc = _scfCompress[g];
            int slen0, slen1, slen2, slen3, tindex;

            if (sfc < 400)
            {
                slen0 = (sfc >> 4) / 5;
                slen1 = (sfc >> 4) % 5;
                slen2 = (sfc & 0xf) >> 2;
                slen3 = sfc & 0x3;
                tindex = 0;
            }
            else if (sfc < 500)
            {
                sfc -= 400;
                slen0 = (sfc >> 2) / 5;
                slen1 = (sfc >> 2) % 5;
                slen2 = sfc & 0x3;
                slen3 = 0;
                tindex = 1;
            }
            else
            {
                sfc -= 500;
                slen0 = sfc / 3;
                slen1 = sfc % 3;
                slen2 = 0;
                slen3 = 0;
                tindex = 2;
                _preflag[g] = 1;
            }

            var blockClass = _blockType[g] == 2 ? (_mixedBlock[g] != 0 ? 2 : 1) : 0;
            var nrBase = (tindex * 3 + blockClass) * 4;
            Span<int> nr = stackalloc int[4] { NrSfb[nrBase], NrSfb[nrBase + 1], NrSfb[nrBase + 2], NrSfb[nrBase + 3] };
            Span<int> slen = stackalloc int[4] { slen0, slen1, slen2, slen3 };

            if (_blockType[g] == 2)
            {
                // short (or mixed) — fill scalefac_s[sfb][win] sfb-major, win-minor.
                var sfb = 0;
                var win = 0;
                for (var p = 0; p < 4; p++)
                {
                    for (var k = 0; k < nr[p]; k++)
                    {
                        var v = slen[p] > 0 ? (int)br.ReadBits(slen[p]) : 0;
                        if (sfb < 13)
                        {
                            _scalefacS[(g * 13 + sfb) * 3 + win] = v;
                        }
                        if (++win == 3)
                        {
                            win = 0;
                            sfb++;
                        }
                    }
                }
            }
            else
            {
                // long — fill scalefac_l[0..] sequentially.
                var sfb = 0;
                for (var p = 0; p < 4; p++)
                {
                    for (var k = 0; k < nr[p]; k++)
                    {
                        var v = slen[p] > 0 ? (int)br.ReadBits(slen[p]) : 0;
                        if (sfb < 23)
                        {
                            _scalefacL[g * 23 + sfb] = v;
                        }
                        sfb++;
                    }
                }
            }
        }

        private void ReadHuffman(ref Mp3BitReader br, int part2Start, int gr, int ch)
        {
            var g = GC(gr, ch);
            var isBase = g * 576;
            for (var i = 0; i < 576; i++)
            {
                _is[isBase + i] = 0f;
            }

            if (_part23[g] == 0)
            {
                _count1[g] = 0;
                return;
            }

            var bitPosEnd = part2Start + _part23[g] - 1;
            int region1Start, region2Start;
            if (_winSwitch[g] == 1 && _blockType[g] == 2)
            {
                region1Start = 36;
                region2Start = 576;
            }
            else
            {
                var bl = Mp3Tables.SfBandLong[_sfIndex];
                region1Start = bl[_region0[g] + 1];
                region2Start = bl[_region0[g] + _region1[g] + 2];
            }

            var pos = 0;
            var bigEnd = _bigValues[g] * 2;
            while (pos < bigEnd)
            {
                int table = pos < region1Start ? _tableSelect[g * 3]
                    : pos < region2Start ? _tableSelect[g * 3 + 1]
                    : _tableSelect[g * 3 + 2];
                Mp3Huffman.DecodeBigValue(ref br, table, out var x, out var y);
                _is[isBase + pos++] = x;
                _is[isBase + pos++] = y;
            }

            var countTable = _count1TableSelect[g] + 32;
            while (pos <= 572 && br.BitPosition <= bitPosEnd)
            {
                Mp3Huffman.DecodeQuad(ref br, countTable, out var v, out var w, out var x, out var y);
                _is[isBase + pos++] = v;
                if (pos >= 576)
                {
                    break;
                }
                _is[isBase + pos++] = w;
                if (pos >= 576)
                {
                    break;
                }
                _is[isBase + pos++] = x;
                if (pos >= 576)
                {
                    break;
                }
                _is[isBase + pos++] = y;
            }

            if (br.BitPosition > bitPosEnd + 1)
            {
                pos -= 4; // overshoot — drop the last quad
            }
            _count1[g] = pos;
            for (; pos < 576; pos++)
            {
                _is[isBase + pos] = 0f;
            }
            br.SetBitPosition(bitPosEnd + 1);
        }

        // ── requantize ──
        private void Requantize(int gr, int ch)
        {
            var g = GC(gr, ch);
            var bl = Mp3Tables.SfBandLong[_sfIndex];
            var bs = Mp3Tables.SfBandShort[_sfIndex];
            var count1 = _count1[g];

            if (_winSwitch[g] == 1 && _blockType[g] == 2)
            {
                if (_mixedBlock[g] != 0)
                {
                    var sfb = 0;
                    var next = bl[1];
                    for (var i = 0; i < 36; i++)
                    {
                        if (i == next)
                        {
                            sfb++;
                            next = bl[sfb + 1];
                        }
                        RequantizeLong(g, i, sfb);
                    }
                    sfb = 3;
                    next = bs[sfb + 1] * 3;
                    var winLen = bs[sfb + 1] - bs[sfb];
                    var ii = 36;
                    while (ii < count1)
                    {
                        if (ii == next)
                        {
                            sfb++;
                            next = bs[sfb + 1] * 3;
                            winLen = bs[sfb + 1] - bs[sfb];
                        }
                        for (var w = 0; w < 3; w++)
                        {
                            for (var j = 0; j < winLen; j++)
                            {
                                RequantizeShort(g, ii, sfb, w);
                                ii++;
                            }
                        }
                    }
                }
                else
                {
                    var sfb = 0;
                    var next = bs[1] * 3;
                    var winLen = bs[1] - bs[0];
                    var ii = 0;
                    while (ii < count1)
                    {
                        if (ii == next)
                        {
                            sfb++;
                            next = bs[sfb + 1] * 3;
                            winLen = bs[sfb + 1] - bs[sfb];
                        }
                        for (var w = 0; w < 3; w++)
                        {
                            for (var j = 0; j < winLen; j++)
                            {
                                RequantizeShort(g, ii, sfb, w);
                                ii++;
                            }
                        }
                    }
                }
            }
            else
            {
                var sfb = 0;
                var next = bl[1];
                for (var i = 0; i < count1; i++)
                {
                    if (i == next)
                    {
                        sfb++;
                        next = bl[sfb + 1];
                    }
                    RequantizeLong(g, i, sfb);
                }
            }
        }

        private void RequantizeLong(int g, int i, int sfb)
        {
            var isBase = g * 576;
            var sfMult = _scalefacScale[g] != 0 ? 1.0f : 0.5f;
            var pfXpt = _preflag[g] * (sfb < 21 ? Pretab[sfb] : 0);
            var tmp1 = MathF.Pow(2.0f, -(sfMult * (_scalefacL[g * 23 + sfb] + pfXpt)));
            var tmp2 = MathF.Pow(2.0f, 0.25f * (_globalGain[g] - 210));
            var v = _is[isBase + i];
            var tmp3 = v < 0f ? -Pow43(-v) : Pow43(v);
            _is[isBase + i] = tmp1 * tmp2 * tmp3;
        }

        private void RequantizeShort(int g, int i, int sfb, int win)
        {
            var isBase = g * 576;
            var sfMult = _scalefacScale[g] != 0 ? 1.0f : 0.5f;
            var tmp1 = MathF.Pow(2.0f, -(sfMult * _scalefacS[(g * 13 + sfb) * 3 + win]));
            var tmp2 = MathF.Pow(2.0f, 0.25f * (_globalGain[g] - 210 - 8 * _subblockGain[g * 3 + win]));
            var v = _is[isBase + i];
            var tmp3 = v < 0f ? -Pow43(-v) : Pow43(v);
            _is[isBase + i] = tmp1 * tmp2 * tmp3;
        }

        private static float Pow43(float x) => MathF.Pow(x, 4.0f / 3.0f);

        // ── reorder short blocks ──
        private void Reorder(int gr, int ch)
        {
            var g = GC(gr, ch);
            if (!(_winSwitch[g] == 1 && _blockType[g] == 2))
            {
                return;
            }
            var isBase = g * 576;
            var bs = Mp3Tables.SfBandShort[_sfIndex];
            var sfb = _mixedBlock[g] != 0 ? 3 : 0;
            var nextSfb = bs[sfb + 1] * 3;
            var winLen = bs[sfb + 1] - bs[sfb];
            var i = sfb == 0 ? 0 : 36;
            while (i < 576)
            {
                if (i == nextSfb)
                {
                    for (var j = 0; j < 3 * winLen; j++)
                    {
                        _is[isBase + 3 * bs[sfb] + j] = _reorder[j];
                    }
                    if (i >= _count1[g])
                    {
                        return;
                    }
                    sfb++;
                    nextSfb = bs[sfb + 1] * 3;
                    winLen = bs[sfb + 1] - bs[sfb];
                }
                for (var w = 0; w < 3; w++)
                {
                    for (var j = 0; j < winLen; j++)
                    {
                        _reorder[j * 3 + w] = _is[isBase + i];
                        i++;
                    }
                }
            }
            for (var j = 0; j < 3 * winLen; j++)
            {
                _is[isBase + 3 * bs[12] + j] = _reorder[j];
            }
        }

        // ── stereo ──
        private void Stereo(int gr)
        {
            if (_nch != 2 || !_joint || _modeExt == 0)
            {
                return;
            }
            var g0 = GC(gr, 0);
            var g1 = GC(gr, 1);
            var b0 = g0 * 576;
            var b1 = g1 * 576;

            if ((_modeExt & 0x2) != 0) // MS stereo
            {
                var maxPos = _count1[g0] > _count1[g1] ? _count1[g0] : _count1[g1];
                for (var i = 0; i < maxPos; i++)
                {
                    var left = (_is[b0 + i] + _is[b1 + i]) * InvSqrt2;
                    var right = (_is[b0 + i] - _is[b1 + i]) * InvSqrt2;
                    _is[b0 + i] = left;
                    _is[b1 + i] = right;
                }
            }

            if ((_modeExt & 0x1) != 0) // intensity stereo
            {
                var bl = Mp3Tables.SfBandLong[_sfIndex];
                var bs = Mp3Tables.SfBandShort[_sfIndex];
                if (_winSwitch[g0] == 1 && _blockType[g0] == 2)
                {
                    if (_mixedBlock[g0] != 0)
                    {
                        for (var sfb = 0; sfb < 8; sfb++)
                        {
                            if (bl[sfb] >= _count1[g1])
                            {
                                IntensityLong(gr, sfb);
                            }
                        }
                        for (var sfb = 3; sfb < 12; sfb++)
                        {
                            if (bs[sfb] * 3 >= _count1[g1])
                            {
                                IntensityShort(gr, sfb);
                            }
                        }
                    }
                    else
                    {
                        for (var sfb = 0; sfb < 12; sfb++)
                        {
                            if (bs[sfb] * 3 >= _count1[g1])
                            {
                                IntensityShort(gr, sfb);
                            }
                        }
                    }
                }
                else
                {
                    for (var sfb = 0; sfb < 21; sfb++)
                    {
                        if (bl[sfb] >= _count1[g1])
                        {
                            IntensityLong(gr, sfb);
                        }
                    }
                }
            }
        }

        private void IntensityLong(int gr, int sfb)
        {
            var g0 = GC(gr, 0);
            var isPos = _scalefacL[g0 * 23 + sfb];
            if (isPos == 7)
            {
                return;
            }
            var bl = Mp3Tables.SfBandLong[_sfIndex];
            float ratioL, ratioR;
            if (isPos == 6)
            {
                ratioL = 1f;
                ratioR = 0f;
            }
            else
            {
                ratioL = IsRatios[isPos] / (1f + IsRatios[isPos]);
                ratioR = 1f / (1f + IsRatios[isPos]);
            }
            var b0 = GC(gr, 0) * 576;
            var b1 = GC(gr, 1) * 576;
            for (var i = bl[sfb]; i < bl[sfb + 1]; i++)
            {
                var sample = _is[b0 + i];
                _is[b0 + i] = ratioL * sample;
                _is[b1 + i] = ratioR * sample;
            }
        }

        private void IntensityShort(int gr, int sfb)
        {
            var g0 = GC(gr, 0);
            var bs = Mp3Tables.SfBandShort[_sfIndex];
            var winLen = bs[sfb + 1] - bs[sfb];
            var b0 = GC(gr, 0) * 576;
            var b1 = GC(gr, 1) * 576;
            for (var win = 0; win < 3; win++)
            {
                var isPos = _scalefacS[(g0 * 13 + sfb) * 3 + win];
                if (isPos == 7)
                {
                    continue;
                }
                float ratioL, ratioR;
                if (isPos == 6)
                {
                    ratioL = 1f;
                    ratioR = 0f;
                }
                else
                {
                    ratioL = IsRatios[isPos] / (1f + IsRatios[isPos]);
                    ratioR = 1f / (1f + IsRatios[isPos]);
                }
                var start = bs[sfb] * 3 + winLen * win;
                for (var i = start; i < start + winLen; i++)
                {
                    var sample = _is[b0 + i];
                    _is[b0 + i] = ratioL * sample;
                    _is[b1 + i] = ratioR * sample;
                }
            }
        }

        // ── antialias ──
        private void Antialias(int gr, int ch)
        {
            var g = GC(gr, ch);
            if (_winSwitch[g] == 1 && _blockType[g] == 2 && _mixedBlock[g] == 0)
            {
                return;
            }
            var sblim = _winSwitch[g] == 1 && _blockType[g] == 2 && _mixedBlock[g] == 1 ? 2 : 32;
            var isBase = g * 576;
            for (var sb = 1; sb < sblim; sb++)
            {
                for (var i = 0; i < 8; i++)
                {
                    var li = 18 * sb - 1 - i;
                    var ui = 18 * sb + i;
                    var lower = _is[isBase + li];
                    var upper = _is[isBase + ui];
                    _is[isBase + li] = lower * Mp3Tables.AliasCs[i] - upper * Mp3Tables.AliasCa[i];
                    _is[isBase + ui] = upper * Mp3Tables.AliasCs[i] + lower * Mp3Tables.AliasCa[i];
                }
            }
        }

        // ── IMDCT + windowing + overlap-add ──
        private void ImdctWin(ReadOnlySpan<float> input, int blockType)
        {
            var win = Mp3Tables.ImdctWin;
            var outp = _imdctOut;
            outp.AsSpan().Clear();

            if (blockType == 2)
            {
                var cos = Mp3Tables.CosN12; // [6 × 12]
                for (var i = 0; i < 3; i++)
                {
                    for (var p = 0; p < 12; p++)
                    {
                        var sum = 0f;
                        for (var m = 0; m < 6; m++)
                        {
                            sum += input[i + 3 * m] * cos[m * 12 + p];
                        }
                        outp[6 * i + p + 6] += sum * win[2 * 36 + p];
                    }
                }
            }
            else
            {
                var cos = Mp3Tables.CosN36; // [18 × 36]
                var wb = blockType * 36;
                for (var p = 0; p < 36; p++)
                {
                    var sum = 0f;
                    for (var m = 0; m < 18; m++)
                    {
                        sum += input[m] * cos[m * 36 + p];
                    }
                    outp[p] = sum * win[wb + p];
                }
            }
        }

        private void HybridSynthesis(int gr, int ch)
        {
            var g = GC(gr, ch);
            var isBase = g * 576;
            var storeBase = ch * 32 * 18;
            if (_hsynthInit)
            {
                _store.AsSpan().Clear();
                _hsynthInit = false;
            }
            for (var sb = 0; sb < 32; sb++)
            {
                var bt = _winSwitch[g] == 1 && _mixedBlock[g] == 1 && sb < 2 ? 0 : _blockType[g];
                ImdctWin(_is.AsSpan(isBase + sb * 18, 18), bt);
                var sStore = storeBase + sb * 18;
                for (var i = 0; i < 18; i++)
                {
                    _is[isBase + sb * 18 + i] = _imdctOut[i] + _store[sStore + i];
                    _store[sStore + i] = _imdctOut[i + 18];
                }
            }
        }

        private void FrequencyInversion(int gr, int ch)
        {
            var isBase = GC(gr, ch) * 576;
            for (var sb = 1; sb < 32; sb += 2)
            {
                for (var i = 1; i < 18; i += 2)
                {
                    _is[isBase + sb * 18 + i] = -_is[isBase + sb * 18 + i];
                }
            }
        }

        // ── polyphase subband synthesis → float PCM[576] ──
        private void SubbandSynthesis(int gr, int ch, Span<float> pcm)
        {
            var g = GC(gr, ch);
            var isBase = g * 576;
            var vBase = ch * 1024;
            var n = Mp3Tables.SynthN;       // [64 × 32]
            var d = Mp3Tables.SynthWindow;  // [512]
            if (_synthInit)
            {
                _vVec.AsSpan().Clear();
                _synthInit = false;
            }

            for (var ss = 0; ss < 18; ss++)
            {
                for (var i = 1023; i > 63; i--)
                {
                    _vVec[vBase + i] = _vVec[vBase + i - 64];
                }
                for (var i = 0; i < 32; i++)
                {
                    _sVec[i] = _is[isBase + i * 18 + ss];
                }
                for (var i = 0; i < 64; i++)
                {
                    var sum = 0f;
                    for (var j = 0; j < 32; j++)
                    {
                        sum += n[i * 32 + j] * _sVec[j];
                    }
                    _vVec[vBase + i] = sum;
                }
                for (var i = 0; i < 8; i++)
                {
                    for (var j = 0; j < 32; j++)
                    {
                        _uVec[(i << 6) + j] = _vVec[vBase + (i << 7) + j];
                        _uVec[(i << 6) + j + 32] = _vVec[vBase + (i << 7) + j + 96];
                    }
                }
                for (var i = 0; i < 512; i++)
                {
                    _uVec[i] *= d[i];
                }
                for (var i = 0; i < 32; i++)
                {
                    var sum = 0f;
                    for (var j = 0; j < 16; j++)
                    {
                        sum += _uVec[(j << 5) + i];
                    }
                    pcm[32 * ss + i] = sum;
                }
            }
        }
    }
}
