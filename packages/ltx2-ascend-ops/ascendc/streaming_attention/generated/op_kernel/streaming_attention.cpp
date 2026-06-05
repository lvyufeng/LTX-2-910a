#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "streaming_attention_tiling.h"

namespace {
constexpr uint32_t kMaxHeadDim = 128;
constexpr uint32_t kMaxBlockN = 64;
constexpr uint32_t kAlignedBlockN = 64;
constexpr uint32_t kDefaultBlockM = 16;
constexpr uint32_t kMaxBlockM = 64;
constexpr uint32_t kFullSoftmaxFastMaxSeq = 8192;
constexpr uint32_t kMaxFullSoftmaxFastUbN = 4096;
constexpr uint32_t kMaxFullSoftmaxLongUbN = 512;
constexpr uint32_t kMaxFullSoftmaxUbN = kMaxFullSoftmaxFastUbN;
constexpr uint32_t kValueTileElements = kMaxBlockM * kMaxHeadDim;
constexpr uint32_t kMatmulLocalWorkspaceBytes = 64U * 1024U;
constexpr float kNegInf = -3.4028234663852886e38f;
constexpr bool kUseOnePassOnline = true;
constexpr bool kDebugWriteQkScores = false;
constexpr bool kDebugWriteTiling = false;

class StreamingAttentionKernel {
public:
    using QkAType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using QkBType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true>;
    using QkBiasType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using QkCType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
    using PvAType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using PvBType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
    using PvBiasType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;
    using PvCType = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

    __aicore__ inline StreamingAttentionKernel() {}

    AscendC::TPipe pipe_;
    AscendC::Matmul<QkAType, QkBType, QkCType, QkBiasType> qkMm_;
    AscendC::Matmul<PvAType, PvBType, PvCType, PvBiasType> pvMm_;

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, __gm__ uint8_t* userWorkspace,
                                const StreamingAttentionTilingData& tiling)
    {
        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(q), tiling.totalElements);
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(k), tiling.totalElements);
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(v), tiling.totalElements);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out), tiling.totalElements);
        seqLen_ = tiling.seqLen;
        headDim_ = tiling.headDim;
        rowCount_ = tiling.rowCount;
        blockM_ = tiling.blockM > kMaxBlockM ? kMaxBlockM : tiling.blockM;
        if (blockM_ == 0U) {
            blockM_ = kDefaultBlockM;
        }
        mode_ = tiling.mode;
        scoreRowStride_ = tiling.blockN;
        __gm__ half* scoreScratch = reinterpret_cast<__gm__ half*>(userWorkspace);
        const uint32_t scratchTiles = AscendC::GetBlockNum();
        const uint32_t scoreTileElements = blockM_ * scoreRowStride_;
        scoreGm_.SetGlobalBuffer(scoreScratch, static_cast<uint64_t>(scratchTiles) * scoreTileElements);
        __gm__ float* pvScratch = reinterpret_cast<__gm__ float*>(
            scoreScratch + static_cast<uint64_t>(scratchTiles) * scoreTileElements);
        pvGm_.SetGlobalBuffer(pvScratch, static_cast<uint64_t>(scratchTiles) * kValueTileElements);
        // AscendC vector Exp on dav-m200 touches a full aligned lane group even when
        // we only request a single scalar lane. An 8-float scratch was large enough
        // for one-shot calls but got clobbered across repeated native invocations,
        // producing sparse row/dim corruptions (the #51 instability). Over-aligning
        // the scratch to 64 floats removes that hazard for the scalar bring-up path.
        // Keep the QK Matmul debug path cube-only: allocating VEC buffers on an
        // AiCore-only probe can hide whether Matmul itself is writing GM.
        pipe_.InitBuffer(matmulWorkspace_, kMatmulLocalWorkspaceBytes);
        qkMm_.SetLocalWorkspace(matmulWorkspace_.Get<uint8_t>());
        pvMm_.SetLocalWorkspace(matmulWorkspace_.Get<uint8_t>());
        if (!kDebugWriteQkScores) {
            // Full-matmul mode keeps the whole score row resident in UB so softmax
            // is a single load/exp/store instead of per-64-key GM round-trips. The
            // blocked online modes only ever touch one kAlignedBlockN chunk, so size
            // their score buffers minimally. seqLen_ is a multiple of 16 (wrapper
            // contract) and <= kFullSoftmaxMaxSeq whenever the host selects mode 1.
            uint32_t scoreBufElems = kAlignedBlockN;
            if (mode_ == 1U) {
                scoreBufElems = seqLen_ > kMaxFullSoftmaxUbN ? kMaxFullSoftmaxUbN : seqLen_;
                if (scoreBufElems < kAlignedBlockN) {
                    scoreBufElems = kAlignedBlockN;
                }
            }
            pipe_.InitBuffer(expBuf_, sizeof(float) * 64);
            pipe_.InitBuffer(scoreHalfBuf_, sizeof(half) * scoreBufElems);
            pipe_.InitBuffer(scoreFloatBuf_, sizeof(float) * scoreBufElems);
            pipe_.InitBuffer(scoreReduceBuf_, sizeof(float) * scoreBufElems);
            pipe_.InitBuffer(valueHalfBuf_, sizeof(half) * kMaxHeadDim);
            pipe_.InitBuffer(valueFloatBuf_, sizeof(float) * kMaxHeadDim);
        }
    }

    __aicore__ inline void Process()
    {
        if (kDebugWriteQkScores) {
            const uint32_t blockNum = AscendC::GetBlockNum();
            const uint32_t rowStride = blockNum * blockM_;
            for (uint32_t row = AscendC::GetBlockIdx() * blockM_; row < rowCount_; row += rowStride) {
                ProcessRow(row);
            }
        } else {
            const uint32_t blockNum = AscendC::GetBlockNum();
            const uint32_t rowStride = blockNum * blockM_;
            for (uint32_t row = AscendC::GetBlockIdx() * blockM_; row < rowCount_; row += rowStride) {
                ProcessTile(row);
            }
        }
        pipe_.Destroy();
    }

private:

    __aicore__ inline float Load(const AscendC::GlobalTensor<half>& tensor, uint64_t index) const
    {
        return static_cast<float>(tensor.GetValue(index));
    }

    __aicore__ inline float AttentionScale() const
    {
        return headDim_ == 64U ? 0.125f : 0.08838834764831845f;
    }

    __aicore__ inline float ScalarExp(float value)
    {
        AscendC::LocalTensor<float> expLocal = expBuf_.Get<float>();
        AscendC::Duplicate(expLocal, value, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(expLocal, expLocal, 64);
        AscendC::PipeBarrier<PIPE_V>();
        return expLocal.GetValue(0);
    }

    __aicore__ inline float ScalarDotScore(uint64_t qBase, uint64_t kBase, float scale) const
    {
        float score = 0.0f;
        for (uint32_t d = 0; d < headDim_; ++d) {
            score += Load(qGm_, qBase + d) * Load(kGm_, kBase + d);
        }
        return score * scale;
    }

    __aicore__ inline AscendC::LocalTensor<float> ScoreFloatLocal()
    {
        return scoreFloatBuf_.Get<float>();
    }

    __aicore__ inline AscendC::LocalTensor<half> ScoreHalfLocal()
    {
        return scoreHalfBuf_.Get<half>();
    }

    __aicore__ inline void LoadScaledScoreRow(uint64_t scoreBase, uint32_t validBlockN, float scale)
    {
        AscendC::LocalTensor<half> scoreHalf = ScoreHalfLocal();
        AscendC::LocalTensor<float> scoreFloat = ScoreFloatLocal();
        AscendC::DataCopy(scoreHalf, scoreGm_[scoreBase], validBlockN);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(scoreFloat, scoreHalf, AscendC::RoundMode::CAST_NONE, validBlockN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(scoreFloat, scoreFloat, scale, validBlockN);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadScaledScoreWholeRow(uint64_t scoreBase, uint32_t validCount, float scale)
    {
        AscendC::LocalTensor<half> scoreHalf = ScoreHalfLocal();
        AscendC::LocalTensor<float> scoreFloat = ScoreFloatLocal();
        AscendC::DataCopy(scoreHalf, scoreGm_[scoreBase], validCount);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(scoreFloat, scoreHalf, AscendC::RoundMode::CAST_NONE, validCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(scoreFloat, scoreFloat, scale, validCount);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float LocalScoreBlockMax(uint32_t validBlockN)
    {
        AscendC::LocalTensor<float> scoreFloat = ScoreFloatLocal();
        if (validBlockN == 64U || validBlockN == 32U || validBlockN == 16U) {
            AscendC::LocalTensor<float> reduceFloat = expBuf_.Get<float>();
            AscendC::Adds(reduceFloat, scoreFloat, 0.0f, validBlockN);
            AscendC::PipeBarrier<PIPE_V>();
            uint32_t count = validBlockN;
            // Only use aligned vector sources. 64->32, 32->16, and 16->8 offsets
            // are 32B aligned for fp32; smaller offsets would fault on dav-c100.
            while (count > 8U) {
                const uint32_t halfCount = count / 2U;
                AscendC::Max(reduceFloat, reduceFloat, reduceFloat[halfCount], static_cast<int32_t>(halfCount));
                AscendC::PipeBarrier<PIPE_V>();
                count = halfCount;
            }
            float blockMax = kNegInf;
            for (uint32_t blockOffset = 0; blockOffset < count; ++blockOffset) {
                const float score = reduceFloat.GetValue(blockOffset);
                blockMax = score > blockMax ? score : blockMax;
            }
            return blockMax;
        }

        float blockMax = kNegInf;
        for (uint32_t blockOffset = 0; blockOffset < validBlockN; ++blockOffset) {
            const float score = scoreFloat.GetValue(blockOffset);
            blockMax = score > blockMax ? score : blockMax;
        }
        return blockMax;
    }

    __aicore__ inline float ShiftExpAndSumLocalScore(float shift, uint32_t validBlockN)
    {
        AscendC::LocalTensor<float> scoreFloat = ScoreFloatLocal();
        AscendC::Adds(scoreFloat, scoreFloat, shift, validBlockN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(scoreFloat, scoreFloat, validBlockN);
        AscendC::PipeBarrier<PIPE_V>();
        if (validBlockN == 64U || validBlockN == 32U || validBlockN == 16U) {
            AscendC::LocalTensor<float> reduceFloat = expBuf_.Get<float>();
            AscendC::Adds(reduceFloat, scoreFloat, 0.0f, validBlockN);
            AscendC::PipeBarrier<PIPE_V>();
            uint32_t count = validBlockN;
            while (count > 8U) {
                const uint32_t halfCount = count / 2U;
                AscendC::Add(reduceFloat, reduceFloat, reduceFloat[halfCount], static_cast<int32_t>(halfCount));
                AscendC::PipeBarrier<PIPE_V>();
                count = halfCount;
            }
            float blockSum = 0.0f;
            for (uint32_t blockOffset = 0; blockOffset < count; ++blockOffset) {
                blockSum += reduceFloat.GetValue(blockOffset);
            }
            return blockSum;
        }

        float blockSum = 0.0f;
        for (uint32_t blockOffset = 0; blockOffset < validBlockN; ++blockOffset) {
            blockSum += scoreFloat.GetValue(blockOffset);
        }
        return blockSum;
    }

    __aicore__ inline AscendC::LocalTensor<float> ScoreReduceLocal()
    {
        return scoreReduceBuf_.Get<float>();
    }

    __aicore__ inline float LocalScoreWholeRowMax(uint32_t validCount)
    {
        AscendC::LocalTensor<float> scoreFloat = ScoreFloatLocal();
        AscendC::LocalTensor<float> reduceFloat = ScoreReduceLocal();
        AscendC::Adds(reduceFloat, scoreFloat, 0.0f, validCount);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t count = validCount;
        while (count > 8U) {
            const uint32_t halfCount = count / 2U;
            // dav-c100 fp32 vector binary ops require 32B-aligned source offsets.
            // halfCount elements * 4B must therefore be a multiple of 32.  Stop
            // early for non-power-of-two multiples of 16 and finish with scalar
            // GetValue below instead of risking an unaligned UUB fault.
            if ((halfCount & 7U) != 0U) {
                break;
            }
            AscendC::Max(reduceFloat, reduceFloat, reduceFloat[halfCount], static_cast<int32_t>(halfCount));
            AscendC::PipeBarrier<PIPE_V>();
            count = halfCount;
        }

        float maxValue = kNegInf;
        for (uint32_t offset = 0; offset < count; ++offset) {
            const float score = reduceFloat.GetValue(offset);
            maxValue = score > maxValue ? score : maxValue;
        }
        return maxValue;
    }

    __aicore__ inline float ShiftExpAndSumWholeRow(float shift, uint32_t validCount)
    {
        AscendC::LocalTensor<float> scoreFloat = ScoreFloatLocal();
        AscendC::LocalTensor<float> reduceFloat = ScoreReduceLocal();
        AscendC::Adds(scoreFloat, scoreFloat, shift, validCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(scoreFloat, scoreFloat, validCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(reduceFloat, scoreFloat, 0.0f, validCount);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t count = validCount;
        while (count > 8U) {
            const uint32_t halfCount = count / 2U;
            if ((halfCount & 7U) != 0U) {
                break;
            }
            AscendC::Add(reduceFloat, reduceFloat, reduceFloat[halfCount], static_cast<int32_t>(halfCount));
            AscendC::PipeBarrier<PIPE_V>();
            count = halfCount;
        }

        float sumValue = 0.0f;
        for (uint32_t offset = 0; offset < count; ++offset) {
            sumValue += reduceFloat.GetValue(offset);
        }
        return sumValue;
    }

    __aicore__ inline void StoreWholeRowProb(uint64_t scoreBase, uint32_t validCount)
    {
        AscendC::LocalTensor<float> scoreFloat = ScoreFloatLocal();
        AscendC::LocalTensor<half> scoreHalf = ScoreHalfLocal();
        AscendC::Cast(scoreHalf, scoreFloat, AscendC::RoundMode::CAST_NONE, validCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(scoreGm_[scoreBase], scoreHalf, validCount);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StoreNormalizedProbRow(uint64_t scoreBase, uint32_t validBlockN, float rowMax, float invSum)
    {
        AscendC::LocalTensor<float> scoreFloat = ScoreFloatLocal();
        AscendC::LocalTensor<half> scoreHalf = ScoreHalfLocal();
        AscendC::Adds(scoreFloat, scoreFloat, -rowMax, validBlockN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(scoreFloat, scoreFloat, validBlockN);
        AscendC::PipeBarrier<PIPE_V>();
        // dav-c100 showed corruption in this streaming+Matmul kernel when a
        // long fp32 vector is multiplied by a tiny per-row invSum before the
        // cast.  Cast exp(score-max) first, then scale the half probabilities;
        // this keeps the PV input fp16, which is the only format CANN Matmul A
        // uses here, while avoiding the unstable fp32 tiny-scalar Muls path.
        AscendC::Cast(scoreHalf, scoreFloat, AscendC::RoundMode::CAST_NONE, validBlockN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(scoreHalf, scoreHalf, static_cast<half>(invSum), validBlockN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(scoreGm_[scoreBase], scoreHalf, validBlockN);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StoreLocalScoreRowAsHalf(uint64_t scoreBase, uint32_t validBlockN)
    {
        AscendC::LocalTensor<float> scoreFloat = ScoreFloatLocal();
        AscendC::LocalTensor<half> scoreHalf = ScoreHalfLocal();
        AscendC::Cast(scoreHalf, scoreFloat, AscendC::RoundMode::CAST_NONE, validBlockN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(scoreGm_[scoreBase], scoreHalf, validBlockN);
        // scoreHalf is a single-row UB staging buffer reused for every row.
        // Wait for UB->GM completion before the next row overwrites it.
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline AscendC::LocalTensor<float> ValueFloatLocal()
    {
        return valueFloatBuf_.Get<float>();
    }

    __aicore__ inline AscendC::LocalTensor<half> ValueHalfLocal()
    {
        return valueHalfBuf_.Get<half>();
    }

    __aicore__ inline void LoadPvRowAsFloat(uint64_t valueBase)
    {
        AscendC::LocalTensor<float> valueFloat = ValueFloatLocal();
        AscendC::DataCopy(valueFloat, pvGm_[valueBase], headDim_);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void StoreOutputRowFromLocal(uint64_t outBase, float invSum)
    {
        AscendC::LocalTensor<float> valueFloat = ValueFloatLocal();
        AscendC::LocalTensor<half> valueHalf = ValueHalfLocal();
        AscendC::Muls(valueFloat, valueFloat, invSum, headDim_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(valueHalf, valueFloat, AscendC::RoundMode::CAST_NONE, headDim_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(outGm_[outBase], valueHalf, headDim_);
        // The output half buffer is reused for the next row immediately. Wait for
        // the store to leave UB before the next row rewrites valueHalf/valueFloat.
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeScoreBlockDebug(uint64_t qBase, uint64_t kBase, uint32_t validBlockN)
    {
        if (kDebugWriteTiling) {
            outGm_.SetValue(qBase + 0, static_cast<half>(1.0f));
            return;
        }
        qkMm_.DisableBias();
        qkMm_.SetOrgShape(blockM_, validBlockN, headDim_, headDim_, headDim_);
        qkMm_.SetSingleShape(blockM_, validBlockN, headDim_);
        qkMm_.SetTensorA(qGm_[qBase], false);
        qkMm_.SetTensorB(kGm_[kBase], true);
        qkMm_.IterateAll<true>(outGm_[qBase], 0, false, true);
        qkMm_.End();
    }

    __aicore__ inline void ComputeScoreBlock(uint64_t qBase, uint64_t kBase, uint32_t validBlockN,
                                             uint64_t scoreOffset)
    {
        qkMm_.DisableBias();
        qkMm_.SetOrgShape(blockM_, validBlockN, headDim_, headDim_, scoreRowStride_);
        qkMm_.SetSingleShape(blockM_, validBlockN, headDim_);
        qkMm_.SetTensorA(qGm_[qBase], false);
        qkMm_.SetTensorB(kGm_[kBase], true);
        qkMm_.IterateAll<true>(scoreGm_[scoreOffset], 0, false, true);
        qkMm_.End();
    }

    __aicore__ inline void ComputeValueBlock(uint64_t probOffset, uint64_t vBase, uint32_t validBlockN,
                                             uint64_t pvOffset)
    {
        pvMm_.DisableBias();
        pvMm_.SetOrgShape(blockM_, headDim_, scoreRowStride_, headDim_, headDim_);
        pvMm_.SetSingleShape(blockM_, headDim_, validBlockN);
        pvMm_.SetTensorA(scoreGm_[probOffset], false);
        pvMm_.SetTensorB(vGm_[vBase], false);
        pvMm_.IterateAll<true>(pvGm_[pvOffset], 0, false, true);
        pvMm_.End();
    }

    __aicore__ inline void ProcessTile(uint32_t rowBase)
    {
        if (mode_ == 1U) {
            ProcessTileFullMatmul(rowBase);
        } else if (kUseOnePassOnline) {
            ProcessTileOnePass(rowBase);
        } else {
            ProcessTileTwoPass(rowBase);
        }
    }

    __aicore__ inline uint32_t FullSoftmaxSegmentElements() const
    {
        return seqLen_ > kFullSoftmaxFastMaxSeq ? kMaxFullSoftmaxLongUbN : kMaxFullSoftmaxFastUbN;
    }

    __aicore__ inline bool UseNormalizedFullMatmulProb() const
    {
        return seqLen_ > kFullSoftmaxFastMaxSeq;
    }

    __aicore__ inline void ProcessTileFullMatmul(uint32_t rowBase)
    {
        float rowSum[kMaxBlockM];
        const uint32_t softmaxSegment = FullSoftmaxSegmentElements();
        const bool normalizedProb = UseNormalizedFullMatmulProb();
        const uint64_t scoreOffset = static_cast<uint64_t>(AscendC::GetBlockIdx()) * blockM_ * scoreRowStride_;
        const uint64_t pvOffset = static_cast<uint64_t>(AscendC::GetBlockIdx()) * kValueTileElements;
        const uint64_t qBase = static_cast<uint64_t>(rowBase) * headDim_;
        const uint64_t kvBase = static_cast<uint64_t>(rowBase / seqLen_) * seqLen_ * headDim_;
        const float scale = AttentionScale();

        ComputeScoreBlock(qBase, kvBase, seqLen_, scoreOffset);
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t m = 0; m < blockM_; ++m) {
            const uint64_t scoreRowBase = scoreOffset + static_cast<uint64_t>(m) * scoreRowStride_;
            float maxValue = kNegInf;
            for (uint32_t keyOffset = 0; keyOffset < seqLen_; keyOffset += softmaxSegment) {
                const uint32_t remaining = seqLen_ - keyOffset;
                const uint32_t validCount = remaining < softmaxSegment ? remaining : softmaxSegment;
                LoadScaledScoreWholeRow(scoreRowBase + keyOffset, validCount, scale);
                const float segmentMax = LocalScoreWholeRowMax(validCount);
                maxValue = segmentMax > maxValue ? segmentMax : maxValue;
            }

            float sumValue = 0.0f;
            for (uint32_t keyOffset = 0; keyOffset < seqLen_; keyOffset += softmaxSegment) {
                const uint32_t remaining = seqLen_ - keyOffset;
                const uint32_t validCount = remaining < softmaxSegment ? remaining : softmaxSegment;
                LoadScaledScoreWholeRow(scoreRowBase + keyOffset, validCount, scale);
                sumValue += ShiftExpAndSumWholeRow(-maxValue, validCount);
                if (!normalizedProb) {
                    // Fast path for validated T<=8192: keep exp(score-max) in fp16,
                    // let PV Matmul consume it, then divide the fp32 PV output by rowSum.
                    StoreWholeRowProb(scoreRowBase + keyOffset, validCount);
                }
            }
            rowSum[m] = sumValue;

            if (normalizedProb) {
                const float invSum = sumValue > 0.0f ? (1.0f / sumValue) : 0.0f;
                for (uint32_t keyOffset = 0; keyOffset < seqLen_; keyOffset += softmaxSegment) {
                    const uint32_t remaining = seqLen_ - keyOffset;
                    const uint32_t validCount = remaining < softmaxSegment ? remaining : softmaxSegment;
                    LoadScaledScoreWholeRow(scoreRowBase + keyOffset, validCount, scale);
                    StoreNormalizedProbRow(scoreRowBase + keyOffset, validCount, maxValue, invSum);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        ComputeValueBlock(scoreOffset, kvBase, seqLen_, pvOffset);
        AscendC::PipeBarrier<PIPE_ALL>();

        for (uint32_t m = 0; m < blockM_; ++m) {
            const uint64_t valueBase = pvOffset + static_cast<uint64_t>(m) * headDim_;
            const uint64_t outBase = qBase + static_cast<uint64_t>(m) * headDim_;
            const float invSum = (!normalizedProb && rowSum[m] > 0.0f) ? (1.0f / rowSum[m]) : 1.0f;
            LoadPvRowAsFloat(valueBase);
            StoreOutputRowFromLocal(outBase, invSum);
        }
    }

    __aicore__ inline void ProcessTileOnePass(uint32_t rowBase)
    {
        const uint32_t tileRows = blockM_ > kDefaultBlockM ? kDefaultBlockM : blockM_;
        float acc[kDefaultBlockM][kMaxHeadDim];
        float rowMax[kDefaultBlockM];
        float rowSum[kDefaultBlockM];
        float rowScale[kDefaultBlockM];
        const uint64_t scoreOffset = static_cast<uint64_t>(AscendC::GetBlockIdx()) * blockM_ * scoreRowStride_;
        const uint64_t pvOffset = static_cast<uint64_t>(AscendC::GetBlockIdx()) * kValueTileElements;

        for (uint32_t m = 0; m < tileRows; ++m) {
            rowMax[m] = kNegInf;
            rowSum[m] = 0.0f;
            rowScale[m] = 0.0f;
            for (uint32_t d = 0; d < headDim_; ++d) {
                acc[m][d] = 0.0f;
            }
        }

        const uint64_t qBase = static_cast<uint64_t>(rowBase) * headDim_;
        const uint64_t kvBase = static_cast<uint64_t>(rowBase / seqLen_) * seqLen_ * headDim_;
        const float scale = AttentionScale();

        for (uint32_t keyBlock = 0; keyBlock < seqLen_; keyBlock += scoreRowStride_) {
            const uint32_t remaining = seqLen_ - keyBlock;
            const uint32_t validBlockN = remaining < scoreRowStride_ ? remaining : scoreRowStride_;
            const uint64_t kBlockBase = kvBase + static_cast<uint64_t>(keyBlock) * headDim_;
            ComputeScoreBlock(qBase, kBlockBase, validBlockN, scoreOffset);
            AscendC::PipeBarrier<PIPE_ALL>();

            for (uint32_t m = 0; m < tileRows; ++m) {
                const uint64_t scoreBase = scoreOffset + static_cast<uint64_t>(m) * kAlignedBlockN;
                LoadScaledScoreRow(scoreBase, validBlockN, scale);
                const float blockMax = LocalScoreBlockMax(validBlockN);
                const float oldMax = rowMax[m];
                const float newMax = blockMax > oldMax ? blockMax : oldMax;
                const float oldScale = keyBlock == 0U ? 0.0f : ScalarExp(oldMax - newMax);
                const float blockSum = ShiftExpAndSumLocalScore(-newMax, validBlockN);
                StoreLocalScoreRowAsHalf(scoreBase, validBlockN);
                rowScale[m] = oldScale;
                rowSum[m] = rowSum[m] * oldScale + blockSum;
                rowMax[m] = newMax;
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            ComputeValueBlock(scoreOffset, kBlockBase, validBlockN, pvOffset);
            AscendC::PipeBarrier<PIPE_ALL>();

            for (uint32_t m = 0; m < tileRows; ++m) {
                const uint64_t valueBase = pvOffset + static_cast<uint64_t>(m) * headDim_;
                const float oldScale = rowScale[m];
                LoadPvRowAsFloat(valueBase);
                AscendC::LocalTensor<float> valueFloat = ValueFloatLocal();
                for (uint32_t d = 0; d < headDim_; ++d) {
                    acc[m][d] = acc[m][d] * oldScale + valueFloat.GetValue(d);
                }
            }
        }

        for (uint32_t m = 0; m < tileRows; ++m) {
            const uint64_t outBase = qBase + static_cast<uint64_t>(m) * headDim_;
            const float invSum = rowSum[m] > 0.0f ? (1.0f / rowSum[m]) : 0.0f;
            AscendC::LocalTensor<float> valueFloat = ValueFloatLocal();
            for (uint32_t d = 0; d < headDim_; ++d) {
                valueFloat.SetValue(d, acc[m][d]);
            }
            AscendC::PipeBarrier<PIPE_V>();
            StoreOutputRowFromLocal(outBase, invSum);
        }
    }

    __aicore__ inline void ProcessTileTwoPass(uint32_t rowBase)
    {
        const uint32_t tileRows = blockM_ > kDefaultBlockM ? kDefaultBlockM : blockM_;
        float acc[kDefaultBlockM][kMaxHeadDim];
        float rowMax[kDefaultBlockM];
        float rowSum[kDefaultBlockM];
        const uint64_t scoreOffset = static_cast<uint64_t>(AscendC::GetBlockIdx()) * blockM_ * scoreRowStride_;
        const uint64_t pvOffset = static_cast<uint64_t>(AscendC::GetBlockIdx()) * kValueTileElements;

        for (uint32_t m = 0; m < tileRows; ++m) {
            rowMax[m] = kNegInf;
            rowSum[m] = 0.0f;
            for (uint32_t d = 0; d < headDim_; ++d) {
                acc[m][d] = 0.0f;
            }
        }

        const uint64_t qBase = static_cast<uint64_t>(rowBase) * headDim_;
        const uint64_t kvBase = static_cast<uint64_t>(rowBase / seqLen_) * seqLen_ * headDim_;
        const float scale = AttentionScale();

        for (uint32_t keyBlock = 0; keyBlock < seqLen_; keyBlock += scoreRowStride_) {
            const uint32_t remaining = seqLen_ - keyBlock;
            const uint32_t validBlockN = remaining < scoreRowStride_ ? remaining : scoreRowStride_;
            const uint64_t kBlockBase = kvBase + static_cast<uint64_t>(keyBlock) * headDim_;
            ComputeScoreBlock(qBase, kBlockBase, validBlockN, scoreOffset);
            AscendC::PipeBarrier<PIPE_ALL>();

            for (uint32_t m = 0; m < tileRows; ++m) {
                const uint64_t scoreBase = scoreOffset + static_cast<uint64_t>(m) * kAlignedBlockN;
                LoadScaledScoreRow(scoreBase, validBlockN, scale);
                const float blockMax = LocalScoreBlockMax(validBlockN);
                const float oldMax = rowMax[m];
                const float newMax = blockMax > oldMax ? blockMax : oldMax;
                const float oldScale = keyBlock == 0U ? 0.0f : ScalarExp(oldMax - newMax);
                const float blockSum = ShiftExpAndSumLocalScore(-newMax, validBlockN);
                rowSum[m] = rowSum[m] * oldScale + blockSum;
                rowMax[m] = newMax;
            }
        }

        for (uint32_t keyBlock = 0; keyBlock < seqLen_; keyBlock += scoreRowStride_) {
            const uint32_t remaining = seqLen_ - keyBlock;
            const uint32_t validBlockN = remaining < scoreRowStride_ ? remaining : scoreRowStride_;
            const uint64_t kBlockBase = kvBase + static_cast<uint64_t>(keyBlock) * headDim_;
            ComputeScoreBlock(qBase, kBlockBase, validBlockN, scoreOffset);
            AscendC::PipeBarrier<PIPE_ALL>();

            for (uint32_t m = 0; m < tileRows; ++m) {
                const uint64_t scoreBase = scoreOffset + static_cast<uint64_t>(m) * kAlignedBlockN;
                const float invSum = rowSum[m] > 0.0f ? (1.0f / rowSum[m]) : 0.0f;
                LoadScaledScoreRow(scoreBase, validBlockN, scale);
                StoreNormalizedProbRow(scoreBase, validBlockN, rowMax[m], invSum);
            }
            AscendC::PipeBarrier<PIPE_ALL>();

            ComputeValueBlock(scoreOffset, kBlockBase, validBlockN, pvOffset);
            AscendC::PipeBarrier<PIPE_ALL>();

            for (uint32_t m = 0; m < tileRows; ++m) {
                const uint64_t valueBase = pvOffset + static_cast<uint64_t>(m) * headDim_;
                for (uint32_t d = 0; d < headDim_; ++d) {
                    acc[m][d] += static_cast<float>(pvGm_.GetValue(valueBase + d));
                }
            }
        }

        for (uint32_t m = 0; m < tileRows; ++m) {
            const uint64_t outBase = qBase + static_cast<uint64_t>(m) * headDim_;
            for (uint32_t d = 0; d < headDim_; ++d) {
                outGm_.SetValue(outBase + d, static_cast<half>(acc[m][d]));
            }
        }
    }


    __aicore__ inline void ProcessRow(uint32_t row)
    {
        const uint64_t qBase = static_cast<uint64_t>(row) * headDim_;
        const uint64_t kvBase = static_cast<uint64_t>(row / seqLen_) * seqLen_ * headDim_;
        if (kDebugWriteQkScores) {
            const uint32_t validBlockN = seqLen_ < scoreRowStride_ ? seqLen_ : scoreRowStride_;
            ComputeScoreBlockDebug(qBase, kvBase, validBlockN);
            for (uint32_t m = 0; m < blockM_; ++m) {
                const uint64_t outBase = qBase + static_cast<uint64_t>(m) * headDim_;
                for (uint32_t d = validBlockN; d < headDim_; ++d) {
                    outGm_.SetValue(outBase + d, static_cast<half>(0.0f));
                }
            }
        }
    }

    AscendC::GlobalTensor<half> qGm_;
    AscendC::GlobalTensor<half> kGm_;
    AscendC::GlobalTensor<half> vGm_;
    AscendC::GlobalTensor<half> outGm_;
    AscendC::GlobalTensor<half> scoreGm_;
    AscendC::GlobalTensor<float> pvGm_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matmulWorkspace_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> expBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scoreHalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scoreFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scoreReduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> valueHalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> valueFloatBuf_;
    uint32_t seqLen_ = 0;
    uint32_t headDim_ = 0;
    uint32_t rowCount_ = 0;
    uint32_t blockM_ = kDefaultBlockM;
    uint32_t mode_ = 0;
    uint32_t scoreRowStride_ = kAlignedBlockN;
};
} // namespace

extern "C" __global__ __aicore__ void streaming_attention(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(StreamingAttentionTilingData);
    GET_TILING_DATA(tilingData, tiling);
    const AscendC::tiling::TCubeTiling* __restrict qkMatmulTiling = &tilingData.qkMatmulTiling;
    const AscendC::tiling::TCubeTiling* __restrict pvMatmulTiling = &tilingData.pvMatmulTiling;
    __gm__ uint8_t* userWorkspace = AscendC::GetUserWorkspace(workspace);
    StreamingAttentionKernel op;
    REGIST_MATMUL_OBJ(&op.pipe_, AscendC::GetSysWorkSpacePtr(),
                      op.qkMm_, qkMatmulTiling,
                      op.pvMm_, pvMatmulTiling);
    op.Init(q, k, v, out, userWorkspace, tilingData);
    op.Process();
}
