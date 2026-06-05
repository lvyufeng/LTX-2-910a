
#include "../op_kernel/streaming_attention_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/matmul/matmul_tiling.h"

#include <cstdlib>
#include <cstring>

namespace {
constexpr int64_t kFallbackBlockDim = 8;
constexpr int64_t kMaxAicoreBlockDim = 24;
constexpr int64_t kDefaultFullSoftmaxMaxSeq = 8192;
// AscendC Matmul on dav-m200 uses the CANN system workspace region for KFC
// server/client communication. GetUserWorkspace() reserves 16 MiB at the head,
// so keep a conservative bring-up allocation while the native op remains opt-in.
constexpr size_t kMatmulWorkspaceBytes = 64ULL * 1024ULL * 1024ULL;

int64_t ShapeDim(const gert::StorageShape* shape, size_t idx)
{
    return shape->GetStorageShape().GetDim(static_cast<int32_t>(idx));
}

bool ReadBhsdShape(const gert::StorageShape* shape, int64_t& batch, int64_t& heads, int64_t& seqLen, int64_t& headDim)
{
    if (shape == nullptr || shape->GetStorageShape().GetDimNum() != 4) {
        return false;
    }
    batch = ShapeDim(shape, 0);
    heads = ShapeDim(shape, 1);
    seqLen = ShapeDim(shape, 2);
    headDim = ShapeDim(shape, 3);
    return batch > 0 && heads > 0 && seqLen > 0 && (headDim == 64 || headDim == 128);
}

bool EnvFlagEnabled(const char* name)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return false;
    }
    return std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 || std::strcmp(raw, "TRUE") == 0 ||
           std::strcmp(raw, "on") == 0 || std::strcmp(raw, "ON") == 0 || std::strcmp(raw, "yes") == 0 ||
           std::strcmp(raw, "YES") == 0;
}

bool EnvFlagDisabled(const char* name)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return false;
    }
    return std::strcmp(raw, "0") == 0 || std::strcmp(raw, "false") == 0 || std::strcmp(raw, "FALSE") == 0 ||
           std::strcmp(raw, "off") == 0 || std::strcmp(raw, "OFF") == 0 || std::strcmp(raw, "no") == 0 ||
           std::strcmp(raw, "NO") == 0;
}

int64_t EnvInt(const char* name, int64_t fallback)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw) {
        return fallback;
    }
    return static_cast<int64_t>(value);
}

bool FillSingleCoreMatmulTiling(AscendC::tiling::TCubeTiling& tiling,
                                const platform_ascendc::PlatformAscendC& platform,
                                int32_t m, int32_t n, int32_t k,
                                bool transposeB, bool cToVec,
                                matmul_tiling::DataType cDataType = matmul_tiling::DataType::DT_FLOAT16)
{
    tiling = AscendC::tiling::TCubeTiling{};
    matmul_tiling::MatmulApiTiling matmulTiling(platform);
    if (matmulTiling.SetAType(matmul_tiling::TPosition::GM,
            matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_FLOAT16,
            false) != 0 ||
        matmulTiling.SetBType(matmul_tiling::TPosition::GM,
            matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_FLOAT16,
            transposeB) != 0 ||
        matmulTiling.SetCType(cToVec ? matmul_tiling::TPosition::VECCALC : matmul_tiling::TPosition::GM,
            cToVec ? matmul_tiling::CubeFormat::ND_ALIGN : matmul_tiling::CubeFormat::ND,
            cDataType) != 0 ||
        matmulTiling.SetShape(m, n, k) != 0 ||
        matmulTiling.SetOrgShape(m, n, k) != 0 ||
        matmulTiling.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN) != 0 ||
        matmulTiling.GetTiling(tiling) != 0) {
        return false;
    }
    return true;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    StreamingAttentionTilingData* tiling = context->GetTilingData<StreamingAttentionTilingData>();
    const gert::StorageShape* qShape = context->GetInputShape(0);
    const gert::StorageShape* kShape = context->GetInputShape(1);
    const gert::StorageShape* vShape = context->GetInputShape(2);

    int64_t batch = 0;
    int64_t heads = 0;
    int64_t seqLen = 0;
    int64_t headDim = 0;
    if (!ReadBhsdShape(qShape, batch, heads, seqLen, headDim)) {
        return ge::GRAPH_FAILED;
    }

    int64_t kBatch = 0;
    int64_t kHeads = 0;
    int64_t kSeqLen = 0;
    int64_t kHeadDim = 0;
    int64_t vBatch = 0;
    int64_t vHeads = 0;
    int64_t vSeqLen = 0;
    int64_t vHeadDim = 0;
    if (!ReadBhsdShape(kShape, kBatch, kHeads, kSeqLen, kHeadDim) ||
        !ReadBhsdShape(vShape, vBatch, vHeads, vSeqLen, vHeadDim) ||
        kBatch != batch || vBatch != batch || kHeads != heads || vHeads != heads ||
        kSeqLen != seqLen || vSeqLen != seqLen || kHeadDim != headDim || vHeadDim != headDim) {
        return ge::GRAPH_FAILED;
    }

    const int64_t rowCount = batch * heads * seqLen;
    const int64_t totalElements = rowCount * headDim;
    if (rowCount <= 0 || totalElements <= 0 || rowCount > UINT32_MAX || totalElements > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    tiling->batch = static_cast<uint32_t>(batch);
    tiling->heads = static_cast<uint32_t>(heads);
    tiling->seqLen = static_cast<uint32_t>(seqLen);
    tiling->headDim = static_cast<uint32_t>(headDim);
    tiling->rowCount = static_cast<uint32_t>(rowCount);
    tiling->totalElements = static_cast<uint32_t>(totalElements);
    const bool cubeParallel = EnvFlagEnabled("LTX2_ASCEND_STREAMING_ATTN_CUBE_PARALLEL");
    // Within the already opt-in native backend, default to the fastest validated
    // 910A path found so far: full-sequence QK/PV Matmul with blockM=32. This
    // collapses the per-64-key-block Matmul loop and passed representative
    // T512/T2048/T8192 numeric checks with the Python shape-change warmup guard.
    // Keep explicit fallbacks: LTX2_ASCEND_STREAMING_ATTN_FULL_MATMUL=0 or
    // LTX2_ASCEND_STREAMING_ATTN_BLOCKED=1 forces the safer blocked online path;
    // LTX2_ASCEND_STREAMING_ATTN_MULTICORE=0 forces single-core blocked when blocked.
    const int64_t fullSoftmaxMaxSeq = EnvInt("LTX2_ASCEND_STREAMING_ATTN_FULL_MATMUL_MAX_T", kDefaultFullSoftmaxMaxSeq);
    const bool fullMatmul = seqLen <= fullSoftmaxMaxSeq &&
                            !EnvFlagDisabled("LTX2_ASCEND_STREAMING_ATTN_FULL_MATMUL") &&
                            !EnvFlagEnabled("LTX2_ASCEND_STREAMING_ATTN_BLOCKED");
    const bool legacyMulticore = EnvFlagEnabled("LTX2_ASCEND_STREAMING_ATTN_MULTICORE");
    int64_t blockM = 16;
    if (fullMatmul) {
        // BM32 is the validated default. BM64 is an explicit experiment only:
        // a little faster on medium shapes but exceeded tolerance on T8192/D128
        // in representative validation.
        const int64_t requestedBlockM = EnvInt("LTX2_ASCEND_STREAMING_ATTN_BLOCK_M", 32);
        if (requestedBlockM >= 64 && seqLen % 64 == 0) {
            blockM = 64;
        } else if (requestedBlockM >= 32 && seqLen % 32 == 0) {
            blockM = 32;
        }
    }
    tiling->blockM = static_cast<uint32_t>(blockM);
    tiling->mode = fullMatmul ? 1U : 0U;
    tiling->blockN = fullMatmul ? static_cast<uint32_t>(seqLen) : 64U;
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    if (!FillSingleCoreMatmulTiling(tiling->qkMatmulTiling,
            platform,
            static_cast<int32_t>(blockM),
            static_cast<int32_t>(tiling->blockN),
            static_cast<int32_t>(headDim),
            true,
            false) ||
        !FillSingleCoreMatmulTiling(tiling->pvMatmulTiling,
            platform,
            static_cast<int32_t>(blockM),
            static_cast<int32_t>(headDim),
            static_cast<int32_t>(tiling->blockN),
            false,
            false,
            matmul_tiling::DataType::DT_FLOAT)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t rowTiles = (rowCount + blockM - 1) / blockM;
    int64_t blockDim = 1;
    if (cubeParallel || legacyMulticore) {
        const int64_t requestedCores = EnvInt(
            cubeParallel ? "LTX2_ASCEND_STREAMING_ATTN_CUBE_PARALLEL_CORES"
                         : "LTX2_ASCEND_STREAMING_ATTN_MULTICORE_CORES",
            kMaxAicoreBlockDim);
        int64_t coreCap = requestedCores;
        if (coreCap < 1) {
            coreCap = 1;
        } else if (coreCap > kMaxAicoreBlockDim) {
            coreCap = kMaxAicoreBlockDim;
        }
        blockDim = rowTiles < coreCap ? rowTiles : coreCap;
        if (blockDim < 1) {
            blockDim = 1;
        }
    }
    // Each AiCore owns independent query tiles and invokes CANN Matmul for its
    // local tile.  Do not ask a single Matmul call to cooperate across all launched
    // cores: that reused the old unstable MULTICORE design and caused sparse large
    // shape corruption on 910A (e.g. T2048/D64).  Keep Matmul tiling single-core;
    // parallelism comes from SetBlockDim + disjoint per-core scratch/output ranges.
    tiling->qkMatmulTiling.usedCoreNum = 1;
    tiling->pvMatmulTiling.usedCoreNum = 1;
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = kMatmulWorkspaceBytes;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class StreamingAttention : public OpDef {
public:
    explicit StreamingAttention(const char* name) : OpDef(name)
    {
        this->Input("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
    }
};

OP_ADD(StreamingAttention);
} // namespace ops
