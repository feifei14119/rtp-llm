#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/cuda/Dispatch.h"

using namespace std;

// namespace trt = tensorrt_llm::kernels::cutlass_kernels;

namespace fastertransformer {
using namespace rocm;

inline rocm::QuantType quantTypeConvert(DataType dtype) {
    switch (dtype) {
        case DataType::TYPE_QINT8: {
            return rocm::QuantType::INT8_WEIGHT_ONLY;
        }
        default: {
            FT_CHECK_WITH_INFO(false, "Invalid quant type");
        }
    }
}
/**
 *  Symmetric Per Channle
 *  Inputs: kernel、 scales（optional）、quantType
 *  Limits：kernel（2D、3D）scales（1D）、quantType（int8、int4x2）
 *  Outputs: QBuffer(kernel, scales, zeros(empty))
 *  note：if scales is null, compute scales
 * 
 * **/
BufferPtr ROCmDevice::quantize(const QuantizeParams& params){
    FT_CHECK_WITH_INFO((params.input.dim() == 2 || params.input.dim() == 3),
        "cuda quantize only support 2D or 3D input.");

    FT_CHECK_WITH_INFO((!params.scales.has_value() && !params.zeros.has_value()),
        "cuda quantize only support SymmetricPerChannel without scales.");
    
    FT_CHECK_WITH_INFO((params.input.type() == DataType::TYPE_FP16 || 
                        params.input.type() == DataType::TYPE_FP32 ||
                        params.input.type() == DataType::TYPE_BF16),
        "cuda quantize only support half or float quantize. but get %d.", params.input.type());
    
    FT_CHECK_WITH_INFO((params.axis == (params.input.dim() - 1)),
        "cuda quantize only support last axis.");
    
    FT_CHECK_WITH_INFO((params.input.where() == MemoryType::MEMORY_CPU),
        "cuda quantize only support input with cpu memory.");
    
    size_t axis = params.input.dim() - 1;
    auto scales = allocateBuffer({DataType::TYPE_FP16,
                                  {params.input.shape()[axis]},
                                  getMemAllocationType(params.input.where())},
                                  {"scales"});
    auto kernel = allocateBuffer({DataType::TYPE_INT8,
                                  params.input.shape(),
                                  getMemAllocationType(params.input.where())},
                                  {"kernel"});
    // TODO(lidongjin) The dispatch maro only support multi template type but without data cast,
    // or one template type with data cast, here need multi template type with data cast.
    if (params.input.type() == DataType::TYPE_FP16) {
        rocm::symmetric_quantize(kernel->data<int8_t>(),
                                nullptr,
                                scales->data<half>(),
                                params.input.data<half>(),
                                params.input.shape(),
                                quantTypeConvert(params.qtype));
#if 0
    } else if (params.input.type() == DataType::TYPE_BF16) {
        trt::symmetric_quantize(kernel->data<int8_t>(),
                                nullptr,
                                scales->data<half>(),
                                params.input.data<__nv_bfloat16>(),
                                params.input.shape(),
                                quantTypeConvert(params.qtype));
    } else if (params.input.type() == DataType::TYPE_FP32) {
        trt::symmetric_quantize(kernel->data<int8_t>(),
                                nullptr,
                                scales->data<half>(),
                                params.input.data<float>(),
                                params.input.shape(),
                                quantTypeConvert(params.qtype));
#endif
    } else {
        FT_CHECK_WITH_INFO(false, 
            "ERROR data type [%d] for cuda quantize input.", params.input.type());
    }


    return BufferPtr(new QBuffer(std::move(kernel),
                                 std::move(scales),
                                 std::move(BufferPtr(new Buffer(MemoryType::MEMORY_CPU_PINNED,
                                                                DataType::TYPE_INVALID,
                                                                {0},
                                                                nullptr)))));

    return nullptr;
}

}   // namespace fastertransformer

