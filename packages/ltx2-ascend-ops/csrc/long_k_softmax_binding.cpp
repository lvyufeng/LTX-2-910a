#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <dlfcn.h>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Functions.h>
#include <c10/core/ScalarType.h>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "aclnn/acl_meta.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

using LongKSoftmaxGetWorkspaceSizeFn = aclnnStatus (*)(
    const aclTensor* scores,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
using LongKSoftmaxRunFn = aclnnStatus (*)(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

struct LongKSoftmaxApi {
    void* handle = nullptr;
    LongKSoftmaxGetWorkspaceSizeFn get_workspace_size = nullptr;
    LongKSoftmaxRunFn run = nullptr;
    std::string error;
};

LongKSoftmaxApi LoadLongKSoftmaxApi()
{
    LongKSoftmaxApi api;
    const char* override_path = std::getenv("LTX2_ASCEND_LONGK_SOFTMAX_LIB");
    const char* lib_name = (override_path != nullptr && std::strlen(override_path) > 0) ? override_path : "libcust_opapi.so";

    api.handle = dlopen(lib_name, RTLD_NOW | RTLD_LOCAL);
    if (api.handle == nullptr) {
        const char* err = dlerror();
        api.error = std::string("dlopen(") + lib_name + ") failed" + (err != nullptr ? std::string(": ") + err : "");
        return api;
    }

    api.get_workspace_size = reinterpret_cast<LongKSoftmaxGetWorkspaceSizeFn>(
        dlsym(api.handle, "aclnnLongKSoftmaxGetWorkspaceSize"));
    if (api.get_workspace_size == nullptr) {
        const char* err = dlerror();
        api.error = std::string("dlsym(aclnnLongKSoftmaxGetWorkspaceSize) failed") +
            (err != nullptr ? std::string(": ") + err : "");
        return api;
    }

    api.run = reinterpret_cast<LongKSoftmaxRunFn>(dlsym(api.handle, "aclnnLongKSoftmax"));
    if (api.run == nullptr) {
        const char* err = dlerror();
        api.error = std::string("dlsym(aclnnLongKSoftmax) failed") +
            (err != nullptr ? std::string(": ") + err : "");
        return api;
    }

    return api;
}

const LongKSoftmaxApi& GetLongKSoftmaxApi()
{
    static const LongKSoftmaxApi api = LoadLongKSoftmaxApi();
    return api;
}

void CheckAclnnStatus(aclnnStatus status, const char* name)
{
    TORCH_CHECK(status == 0, name, " failed with aclnnStatus=", status);
}

bool EnvFlagEnabled(const char* name)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return false;
    }
    std::string value(raw);
    for (char& ch : value) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

void CheckLongKSoftmaxInputs(const at::Tensor& scores)
{
    TORCH_CHECK(scores.dim() == 4, "ltx2_ascend.long_k_softmax expects scores in B,H,Q,K layout");
    TORCH_CHECK(scores.scalar_type() == at::kHalf, "ltx2_ascend.long_k_softmax supports fp16 scores only");
    TORCH_CHECK(scores.is_contiguous(), "ltx2_ascend.long_k_softmax expects contiguous scores");
    TORCH_CHECK(scores.storage_offset() == 0, "ltx2_ascend.long_k_softmax expects zero storage_offset tensors");
    TORCH_CHECK(scores.size(0) > 0 && scores.size(1) > 0 && scores.size(2) > 0 && scores.size(3) > 0,
        "ltx2_ascend.long_k_softmax expects non-empty B,H,Q,K dimensions");
    TORCH_CHECK(scores.size(3) % 16 == 0,
        "ltx2_ascend.long_k_softmax v1 requires K to be a multiple of 16, got ", scores.size(3));
    TORCH_CHECK(scores.numel() <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        "ltx2_ascend.long_k_softmax v1 total element count exceeds uint32 tiling limit: ", scores.numel());
}

std::vector<int64_t> TensorDims(const at::Tensor& tensor)
{
    return std::vector<int64_t>(tensor.sizes().begin(), tensor.sizes().end());
}

std::vector<int64_t> TensorStrides(const at::Tensor& tensor)
{
    return std::vector<int64_t>(tensor.strides().begin(), tensor.strides().end());
}

struct AclTensorDeleter {
    void operator()(aclTensor* tensor) const
    {
        if (tensor != nullptr) {
            aclDestroyTensor(tensor);
        }
    }
};
using AclTensorPtr = std::unique_ptr<aclTensor, AclTensorDeleter>;

AclTensorPtr MakeAclTensor(
    const at::Tensor& tensor,
    std::vector<int64_t>& dims,
    std::vector<int64_t>& strides,
    std::vector<int64_t>& storage_dims)
{
    dims = TensorDims(tensor);
    strides = TensorStrides(tensor);
    storage_dims = dims;
    aclTensor* acl_tensor = aclCreateTensor(
        dims.data(),
        static_cast<uint64_t>(dims.size()),
        ACL_FLOAT16,
        strides.data(),
        0,
        ACL_FORMAT_ND,
        storage_dims.data(),
        static_cast<uint64_t>(storage_dims.size()),
        const_cast<void*>(tensor.data_ptr()));
    TORCH_CHECK(acl_tensor != nullptr, "aclCreateTensor failed for ltx2_ascend.long_k_softmax");
    return AclTensorPtr(acl_tensor);
}

bool LongKSoftmaxIsAvailable()
{
    const auto& api = GetLongKSoftmaxApi();
    return api.handle != nullptr && api.get_workspace_size != nullptr && api.run != nullptr;
}

at::Tensor LongKSoftmaxForward(const at::Tensor& scores)
{
    CheckLongKSoftmaxInputs(scores);

    const auto& api = GetLongKSoftmaxApi();
    TORCH_CHECK(LongKSoftmaxIsAvailable(),
        "ltx2_ascend.long_k_softmax native ACLNN API is unavailable: ", api.error,
        ". Install/export the generated custom OPP and set ASCEND_CUSTOM_OPP_PATH/LD_LIBRARY_PATH, "
        "or set LTX2_ASCEND_LONGK_SOFTMAX_LIB to the LongKSoftmax libcust_opapi.so.");

    auto out = at::empty_like(scores, scores.options(), at::MemoryFormat::Contiguous);
    TORCH_CHECK(out.storage_offset() == 0, "ltx2_ascend.long_k_softmax output allocation has nonzero storage_offset");

    std::vector<int64_t> scores_dims;
    std::vector<int64_t> scores_strides;
    std::vector<int64_t> scores_storage_dims;
    std::vector<int64_t> out_dims;
    std::vector<int64_t> out_strides;
    std::vector<int64_t> out_storage_dims;
    auto scores_acl = MakeAclTensor(scores, scores_dims, scores_strides, scores_storage_dims);
    auto out_acl = MakeAclTensor(out, out_dims, out_strides, out_storage_dims);

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    CheckAclnnStatus(api.get_workspace_size(scores_acl.get(), out_acl.get(), &workspace_size, &executor),
        "aclnnLongKSoftmaxGetWorkspaceSize");
    TORCH_CHECK(executor != nullptr, "aclnnLongKSoftmaxGetWorkspaceSize returned a null executor");

    at::Tensor workspace;
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        TORCH_CHECK(workspace_size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "ltx2_ascend.long_k_softmax workspace is too large: ", workspace_size);
        workspace = at::empty({static_cast<int64_t>(workspace_size)}, scores.options().dtype(at::kByte));
        workspace_ptr = workspace.data_ptr();
    }

    const auto device_index = static_cast<c10::DeviceIndex>(scores.device().index());
    aclrtStream stream = c10_npu::getCurrentNPUStream(device_index).stream(false);
    CheckAclnnStatus(api.run(workspace_ptr, workspace_size, executor, stream), "aclnnLongKSoftmax");
    // Keep descriptors and the temporary workspace alive until execution finishes.
    // This is an opt-in validation backend, so correctness/lifetime safety wins over
    // shaving a host sync until repeated-call stress proves async is safe.
    if (!EnvFlagEnabled("LTX2_ASCEND_LONGK_SOFTMAX_ASYNC")) {
        CheckAclnnStatus(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    }
    return out;
}

} // namespace

TORCH_LIBRARY(ltx2_ascend_longk, m)
{
    m.def("long_k_softmax(Tensor scores) -> Tensor");
    m.def("long_k_softmax_is_available() -> bool");
}

TORCH_LIBRARY_IMPL(ltx2_ascend_longk, CompositeExplicitAutograd, m)
{
    m.impl("long_k_softmax", TORCH_FN(LongKSoftmaxForward));
    m.impl("long_k_softmax_is_available", TORCH_FN(LongKSoftmaxIsAvailable));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
