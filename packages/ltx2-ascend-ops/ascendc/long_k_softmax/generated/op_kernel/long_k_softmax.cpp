#include "kernel_operator.h"
#include "long_k_softmax_tiling.h"

namespace {
constexpr uint32_t kMaxSegmentElements = 1024;
constexpr float kNegInf = -3.4028234663852886e38f;

class LongKSoftmaxKernel {
public:
    __aicore__ inline LongKSoftmaxKernel() {}

    __aicore__ inline void Init(GM_ADDR scores, GM_ADDR probs, const LongKSoftmaxTilingData& tiling)
    {
        scoresGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scores), tiling.totalElements);
        probsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(probs), tiling.totalElements);
        rowCount_ = tiling.rowCount;
        kLen_ = tiling.kLen;
        segmentElements_ = tiling.segmentElements > kMaxSegmentElements ? kMaxSegmentElements : tiling.segmentElements;
        if (segmentElements_ == 0U) {
            segmentElements_ = kMaxSegmentElements;
        }
        pipe_.InitBuffer(scoreHalfBuf_, sizeof(half) * kMaxSegmentElements);
        pipe_.InitBuffer(scoreFloatBuf_, sizeof(float) * kMaxSegmentElements);
        pipe_.InitBuffer(reduceFloatBuf_, sizeof(float) * kMaxSegmentElements);
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockNum = AscendC::GetBlockNum();
        for (uint32_t row = AscendC::GetBlockIdx(); row < rowCount_; row += blockNum) {
            ProcessRow(row);
        }
        pipe_.Destroy();
    }

private:
    __aicore__ inline AscendC::LocalTensor<half> ScoreHalfLocal()
    {
        return scoreHalfBuf_.Get<half>();
    }

    __aicore__ inline AscendC::LocalTensor<float> ScoreFloatLocal()
    {
        return scoreFloatBuf_.Get<float>();
    }

    __aicore__ inline AscendC::LocalTensor<float> ReduceFloatLocal()
    {
        return reduceFloatBuf_.Get<float>();
    }

    __aicore__ inline void LoadScoreSegment(uint64_t base, uint32_t validCount)
    {
        AscendC::LocalTensor<half> scoreHalf = ScoreHalfLocal();
        AscendC::LocalTensor<float> scoreFloat = ScoreFloatLocal();
        AscendC::DataCopy(scoreHalf, scoresGm_[base], validCount);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Cast(scoreFloat, scoreHalf, AscendC::RoundMode::CAST_NONE, validCount);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float LocalMax(uint32_t validCount)
    {
        AscendC::LocalTensor<float> scoreFloat = ScoreFloatLocal();
        AscendC::LocalTensor<float> reduceFloat = ReduceFloatLocal();
        AscendC::Adds(reduceFloat, scoreFloat, 0.0f, validCount);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t count = validCount;
        while (count > 8U) {
            const uint32_t halfCount = count / 2U;
            // dav-c100 fp32 vector binary ops require 32B-aligned source offsets.
            // If the halved offset is not 8-float aligned, finish the remaining
            // partials with scalar GetValue rather than risking a UUB fault.
            if ((halfCount & 7U) != 0U) {
                break;
            }
            AscendC::Max(reduceFloat, reduceFloat, reduceFloat[halfCount], static_cast<int32_t>(halfCount));
            AscendC::PipeBarrier<PIPE_V>();
            count = halfCount;
        }

        float maxValue = kNegInf;
        for (uint32_t offset = 0; offset < count; ++offset) {
            const float value = reduceFloat.GetValue(offset);
            maxValue = value > maxValue ? value : maxValue;
        }
        return maxValue;
    }

    __aicore__ inline float ScalarExp(float value)
    {
        AscendC::LocalTensor<float> reduceFloat = ReduceFloatLocal();
        AscendC::Duplicate(reduceFloat, value, 64);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(reduceFloat, reduceFloat, 64);
        AscendC::PipeBarrier<PIPE_V>();
        return reduceFloat.GetValue(0);
    }

    __aicore__ inline float ShiftExpAndSum(float shift, uint32_t validCount)
    {
        AscendC::LocalTensor<float> scoreFloat = ScoreFloatLocal();
        AscendC::LocalTensor<float> reduceFloat = ReduceFloatLocal();
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

    __aicore__ inline void StoreNormalizedSegment(uint64_t inputBase, uint64_t outputBase, float rowMax, float invSum,
                                                  uint32_t validCount)
    {
        AscendC::LocalTensor<half> scoreHalf = ScoreHalfLocal();
        AscendC::LocalTensor<float> scoreFloat = ScoreFloatLocal();
        LoadScoreSegment(inputBase, validCount);
        AscendC::Adds(scoreFloat, scoreFloat, -rowMax, validCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Exp(scoreFloat, scoreFloat, validCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(scoreFloat, scoreFloat, invSum, validCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Cast(scoreHalf, scoreFloat, AscendC::RoundMode::CAST_NONE, validCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::DataCopy(probsGm_[outputBase], scoreHalf, validCount);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessRow(uint32_t row)
    {
        const uint64_t rowBase = static_cast<uint64_t>(row) * kLen_;

        // First pass: online fp32 softmax recurrence across <=1024-score
        // segments.  This preserves the same final fp32 row max/sum as the
        // three-pass formulation while dropping one full GM read of scores.
        float rowMax = kNegInf;
        float rowSum = 0.0f;
        for (uint32_t keyOffset = 0; keyOffset < kLen_; keyOffset += segmentElements_) {
            const uint32_t remaining = kLen_ - keyOffset;
            const uint32_t validCount = remaining < segmentElements_ ? remaining : segmentElements_;
            LoadScoreSegment(rowBase + keyOffset, validCount);
            const float segmentMax = LocalMax(validCount);
            const float oldMax = rowMax;
            const float newMax = segmentMax > oldMax ? segmentMax : oldMax;
            const float oldScale = keyOffset == 0U ? 0.0f : ScalarExp(oldMax - newMax);
            const float segmentSum = ShiftExpAndSum(-newMax, validCount);
            rowSum = rowSum * oldScale + segmentSum;
            rowMax = newMax;
        }

        // Second pass: write normalized fp16 probabilities with the final fp32
        // row max/sum.  The input scores are preserved in GM until this pass.
        const float invSum = rowSum > 0.0f ? (1.0f / rowSum) : 0.0f;
        for (uint32_t keyOffset = 0; keyOffset < kLen_; keyOffset += segmentElements_) {
            const uint32_t remaining = kLen_ - keyOffset;
            const uint32_t validCount = remaining < segmentElements_ ? remaining : segmentElements_;
            StoreNormalizedSegment(rowBase + keyOffset, rowBase + keyOffset, rowMax, invSum, validCount);
        }
    }

    AscendC::GlobalTensor<half> scoresGm_;
    AscendC::GlobalTensor<half> probsGm_;
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scoreHalfBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scoreFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceFloatBuf_;
    uint32_t rowCount_ = 0;
    uint32_t kLen_ = 0;
    uint32_t segmentElements_ = kMaxSegmentElements;
};
} // namespace

extern "C" __global__ __aicore__ void long_k_softmax(GM_ADDR scores, GM_ADDR probs, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LongKSoftmaxTilingData);
    GET_TILING_DATA(tilingData, tiling);
    LongKSoftmaxKernel op;
    op.Init(scores, probs, tilingData);
    op.Process();
}
