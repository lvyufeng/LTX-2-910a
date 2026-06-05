#include "../op_kernel/long_k_softmax_tiling.h"
#include "register/op_def_registry.h"

#include <cstdint>
#include <limits>

namespace {
constexpr int64_t kMaxAicoreBlockDim = 24;
constexpr uint32_t kSegmentElements = 1024;

int64_t ShapeDim(const gert::StorageShape* shape, size_t idx)
{
    return shape->GetStorageShape().GetDim(static_cast<int32_t>(idx));
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    LongKSoftmaxTilingData* tiling = context->GetTilingData<LongKSoftmaxTilingData>();
    const gert::StorageShape* scoresShape = context->GetInputShape(0);
    if (scoresShape == nullptr || scoresShape->GetStorageShape().GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    const int64_t batch = ShapeDim(scoresShape, 0);
    const int64_t heads = ShapeDim(scoresShape, 1);
    const int64_t qLen = ShapeDim(scoresShape, 2);
    const int64_t kLen = ShapeDim(scoresShape, 3);
    if (batch <= 0 || heads <= 0 || qLen <= 0 || kLen <= 0) {
        return ge::GRAPH_FAILED;
    }

    const int64_t rowCount = batch * heads * qLen;
    const int64_t totalElements = rowCount * kLen;
    const int64_t uint32Max = static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    if (rowCount <= 0 || totalElements <= 0 || rowCount > uint32Max || kLen > uint32Max || totalElements > uint32Max) {
        return ge::GRAPH_FAILED;
    }

    tiling->rowCount = static_cast<uint32_t>(rowCount);
    tiling->kLen = static_cast<uint32_t>(kLen);
    tiling->totalElements = static_cast<uint32_t>(totalElements);
    tiling->segmentElements = kSegmentElements;

    int64_t blockDim = rowCount < kMaxAicoreBlockDim ? rowCount : kMaxAicoreBlockDim;
    if (blockDim < 1) {
        blockDim = 1;
    }
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
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
class LongKSoftmax : public OpDef {
public:
    explicit LongKSoftmax(const char* name) : OpDef(name)
    {
        this->Input("scores")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("probs")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910");
    }
};

OP_ADD(LongKSoftmax);
} // namespace ops
