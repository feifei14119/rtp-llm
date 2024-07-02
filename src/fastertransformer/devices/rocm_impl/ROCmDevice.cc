#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/devices/rocm_impl/ROCmAllocator.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/kernels/hello_world.h"
#include "src/fastertransformer/utils/ShapeCheck.h"
#include <cstring>
namespace fastertransformer {

ROCmDevice::ROCmDevice(const DeviceInitParams& params): DeviceBase(params) {
    RUNTIME_ASSERT_OP_ARG(params.tp_rank == 0, "rocm device doesn't support nccl");
    // RUNTIME_ASSERT_OP_ARG(params.host_reserve_memory_bytes == 0, "rocm device doesn't reserve host memory");
    // RUNTIME_ASSERT_OP_ARG(params.device_reserve_memory_bytes == 0, "rocm device doesn't reserve device memory");
    allocator_.reset(new Allocator<AllocatorType::ROCM>());
    hostAllocator_.reset(new Allocator<AllocatorType::ROCM_HOST>());
    (void)hipSetDevice(0);
    (void)hipStreamCreate(&stream_);
}

ROCmDevice::~ROCmDevice() {
    if (stream_ != nullptr) {
        (void)hipStreamDestroy(stream_);
    }
}

DeviceProperties ROCmDevice::getDeviceProperties() {
    DeviceProperties props;
    props.type = DeviceType::ROCm;
    return props;
}

void ROCmDevice::copy(const CopyParams& params) {
    FT_CHECK_WITH_INFO(params.src.type() == params.dst.type(),
                       "dst[%d] and src[%d,] need has same type.",
                       params.src.type(),
                       params.dst.type());

    RUNTIME_ASSERT_OP_ARG(!params.dst.isQuantify() && !params.src.isQuantify(),
                          "rocm device doesn't support qint8 copy");

    const auto src_offset  = params.src_offset;
    const auto dst_offset  = params.dst_offset;
    auto       copy_length = params.copy_length;

    if (copy_length < 0) {
        RUNTIME_ASSERT_OP_ARG(params.src.shape()[0] == params.dst.shape()[0],
                              "src and dst 0-dim size mismatch: [%s] vs [%s]",
                              params.src.debugString().c_str(),
                              params.dst.debugString().c_str());
        copy_length = params.src.shape()[0];
    }

    if (copy_length == 0) {
        return;
    }

    const auto src = params.src.view(src_offset, copy_length);
    const auto dst = params.dst.view(dst_offset, copy_length);

    RUNTIME_ASSERT_OP_ARG(src.sizeBytes() == dst.sizeBytes(),
                          "src and dst copy size mismatch: [%s] vs [%s]",
                          src.debugString().c_str(),
                          dst.debugString().c_str());

    if (src.data() == dst.data()) {
        return;
    }

    hipMemcpyKind copyType;
    if (src.where() == MemoryType::MEMORY_GPU && dst.where() != MemoryType::MEMORY_GPU) {
        copyType = hipMemcpyDeviceToHost;
    } else if (src.where() != MemoryType::MEMORY_GPU && dst.where() == MemoryType::MEMORY_GPU) {
        copyType = hipMemcpyHostToDevice;
    } else if (src.where() == MemoryType::MEMORY_GPU && dst.where() == MemoryType::MEMORY_GPU) {
        copyType = hipMemcpyDeviceToDevice;
    } else {
        copyType = hipMemcpyHostToHost;
    }

    (void)hipMemcpyWithStream(dst.data(), src.data(), src.sizeBytes(), copyType, stream_);
    (void)hipStreamSynchronize(stream_);
}

void ROCmDevice::syncAndCheck() {
    (void)hipDeviceSynchronize();
}

BufferPtr ROCmDevice::gemm(const GemmParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}
SelectOutput ROCmDevice::select(const SelectParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

BufferPtr ROCmDevice::embeddingLookup(const EmbeddingLookupParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

LayernormOutput ROCmDevice::layernorm(const LayernormParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

void ROCmDevice::activation(const ActivationParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}

AttentionModuleOutput ROCmDevice::contextAttention(const AttentionModuleParams& params) {
    throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
}
BufferPtr ROCmDevice::testVecAdd(const BufferPtr a, const BufferPtr b) {
    BufferPtr           output;
    DataType            dtype  = a.get()->type();
    std::vector<size_t> dshape = a.get()->shape();

    output = allocateBuffer({dtype, dshape, AllocationType::DEVICE}, {"vec_add_rslt"});
    invokeHelloWorld<float>((const float*)(a.get()->data()),
                            ((const float*)b.get()->data()),
                            ((float*)output.get()->data()),
                            output.get()->size(),
                            stream_);

    return output;
}

RTP_LLM_REGISTER_DEVICE(ROCm);

}  // namespace fastertransformer
