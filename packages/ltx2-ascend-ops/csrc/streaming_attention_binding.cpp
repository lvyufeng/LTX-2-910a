#include <cmath>
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

using StreamingAttentionGetWorkspaceSizeFn = aclnnStatus (*)(
    const aclTensor* q,
    const aclTensor* k,
    const aclTensor* v,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
using StreamingAttentionRunFn = aclnnStatus (*)(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);

struct StreamingAttentionApi {
    void* handle = nullptr;
    StreamingAttentionGetWorkspaceSizeFn get_workspace_size = nullptr;
    StreamingAttentionRunFn run = nullptr;
    std::string error;
};

StreamingAttentionApi LoadStreamingAttentionApi()
{
    StreamingAttentionApi api;
    const char* override_path = std::getenv("LTX2_ASCEND_STREAMING_ATTN_LIB");
    const char* lib_name = (override_path != nullptr && std::strlen(override_path) > 0) ? override_path : "libcust_opapi.so";

    api.handle = dlopen(lib_name, RTLD_NOW | RTLD_LOCAL);
    if (api.handle == nullptr) {
        const char* err = dlerror();
        api.error = std::string("dlopen(") + lib_name + ") failed" + (err != nullptr ? std::string(": ") + err : "");
        return api;
    }

    api.get_workspace_size = reinterpret_cast<StreamingAttentionGetWorkspaceSizeFn>(
        dlsym(api.handle, "aclnnStreamingAttentionGetWorkspaceSize"));
    if (api.get_workspace_size == nullptr) {
        const char* err = dlerror();
        api.error = std::string("dlsym(aclnnStreamingAttentionGetWorkspaceSize) failed") +
            (err != nullptr ? std::string(": ") + err : "");
        return api;
    }

    api.run = reinterpret_cast<StreamingAttentionRunFn>(dlsym(api.handle, "aclnnStreamingAttention"));
    if (api.run == nullptr) {
        const char* err = dlerror();
        api.error = std::string("dlsym(aclnnStreamingAttention) failed") +
            (err != nullptr ? std::string(": ") + err : "");
        return api;
    }

    return api;
}

const StreamingAttentionApi& GetStreamingAttentionApi()
{
    static const StreamingAttentionApi api = LoadStreamingAttentionApi();
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

void CheckStreamingAttentionInputs(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double scale,
    int64_t block_m,
    int64_t block_n)
{
    TORCH_CHECK(q.dim() == 4, "ltx2_ascend.streaming_attention expects q in BNSD layout (B,H,T,D)");
    TORCH_CHECK(k.sizes() == q.sizes(), "ltx2_ascend.streaming_attention expects k shape to match q");
    TORCH_CHECK(v.sizes() == q.sizes(), "ltx2_ascend.streaming_attention expects v shape to match q");
    TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf,
        "ltx2_ascend.streaming_attention v1 supports fp16 q/k/v only");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
        "ltx2_ascend.streaming_attention expects q/k/v on the same NPU device");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
        "ltx2_ascend.streaming_attention expects contiguous BNSD q/k/v tensors");
    TORCH_CHECK(q.storage_offset() == 0 && k.storage_offset() == 0 && v.storage_offset() == 0,
        "ltx2_ascend.streaming_attention v1 expects zero storage_offset tensors");

    const int64_t head_dim = q.size(3);
    TORCH_CHECK(head_dim == 64 || head_dim == 128,
        "ltx2_ascend.streaming_attention v1 supports head_dim 64 or 128, got ", head_dim);
    const double expected_scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    TORCH_CHECK(std::abs(scale - expected_scale) <= 1.0e-6,
        "ltx2_ascend.streaming_attention scale must be dim_head**-0.5 for v1; got ", scale,
        ", expected ", expected_scale);
    TORCH_CHECK(block_m >= 0 && block_n >= 0,
        "ltx2_ascend.streaming_attention block_m/block_n must be non-negative");
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
    TORCH_CHECK(acl_tensor != nullptr, "aclCreateTensor failed for ltx2_ascend.streaming_attention");
    return AclTensorPtr(acl_tensor);
}

bool StreamingAttentionIsAvailable()
{
    const auto& api = GetStreamingAttentionApi();
    return api.handle != nullptr && api.get_workspace_size != nullptr && api.run != nullptr;
}

at::Tensor StreamingAttentionForward(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double scale,
    int64_t block_m,
    int64_t block_n)
{
    CheckStreamingAttentionInputs(q, k, v, scale, block_m, block_n);

    const auto& api = GetStreamingAttentionApi();
    TORCH_CHECK(StreamingAttentionIsAvailable(),
        "ltx2_ascend.streaming_attention native ACLNN API is unavailable: ", api.error,
        ". Install the generated custom OPP and export ASCEND_CUSTOM_OPP_PATH/LD_LIBRARY_PATH, "
        "or set LTX2_ASCEND_STREAMING_ATTN_LIB to libcust_opapi.so.");

    auto out = at::empty_like(q, q.options(), at::MemoryFormat::Contiguous);
    TORCH_CHECK(out.storage_offset() == 0, "ltx2_ascend.streaming_attention output allocation has nonzero storage_offset");

    std::vector<int64_t> q_dims;
    std::vector<int64_t> q_strides;
    std::vector<int64_t> q_storage_dims;
    std::vector<int64_t> k_dims;
    std::vector<int64_t> k_strides;
    std::vector<int64_t> k_storage_dims;
    std::vector<int64_t> v_dims;
    std::vector<int64_t> v_strides;
    std::vector<int64_t> v_storage_dims;
    std::vector<int64_t> out_dims;
    std::vector<int64_t> out_strides;
    std::vector<int64_t> out_storage_dims;
    auto q_acl = MakeAclTensor(q, q_dims, q_strides, q_storage_dims);
    auto k_acl = MakeAclTensor(k, k_dims, k_strides, k_storage_dims);
    auto v_acl = MakeAclTensor(v, v_dims, v_strides, v_storage_dims);
    auto out_acl = MakeAclTensor(out, out_dims, out_strides, out_storage_dims);

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    CheckAclnnStatus(api.get_workspace_size(q_acl.get(), k_acl.get(), v_acl.get(), out_acl.get(), &workspace_size, &executor),
        "aclnnStreamingAttentionGetWorkspaceSize");
    TORCH_CHECK(executor != nullptr, "aclnnStreamingAttentionGetWorkspaceSize returned a null executor");

    at::Tensor workspace;
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        TORCH_CHECK(workspace_size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "ltx2_ascend.streaming_attention workspace is too large: ", workspace_size);
        workspace = at::empty({static_cast<int64_t>(workspace_size)}, q.options().dtype(at::kByte));
        workspace_ptr = workspace.data_ptr();
    }

    const auto device_index = static_cast<c10::DeviceIndex>(q.device().index());
    aclrtStream stream = c10_npu::getCurrentNPUStream(device_index).stream(false);
    CheckAclnnStatus(api.run(workspace_ptr, workspace_size, executor, stream), "aclnnStreamingAttention");
    // Default to the conservative synchronous path so ACL tensor descriptors and the
    // temporary workspace tensor cannot be destroyed/reused before the generated
    // ACLNN executor has consumed them. 910A Matmul bring-up showed that subtle
    // lifetime/state bugs can look like random numerical corruption, so the async
    // path is an explicit benchmark/stress-test experiment only.
    if (!EnvFlagEnabled("LTX2_ASCEND_STREAMING_ATTN_ASYNC")) {
        CheckAclnnStatus(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    }
    return out;
}

} // namespace

TORCH_LIBRARY(ltx2_ascend, m)
{
    m.def("streaming_attention(Tensor q, Tensor k, Tensor v, float scale, int block_m=0, int block_n=0) -> Tensor");
    m.def("streaming_attention_is_available() -> bool");
}

TORCH_LIBRARY_IMPL(ltx2_ascend, CompositeExplicitAutograd, m)
{
    m.impl("streaming_attention", TORCH_FN(StreamingAttentionForward));
    m.impl("streaming_attention_is_available", TORCH_FN(StreamingAttentionIsAvailable));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
