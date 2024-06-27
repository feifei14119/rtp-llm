#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/devices/rocm_impl/ROCmAllocator.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/kernels/rmsnormKernels.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/utils/ShapeCheck.h"
#include <cstring>

#include <hip/hip_runtime.h>

// TODO(rocm): Idealy we just link compiler_rt for this symbol.
extern "C" half __truncdfhf2(double a) {
    return (half)(float)a;
}

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

struct GemmArguments {
    std::vector<size_t> Ashape;
    std::vector<size_t> Bshape;
    std::vector<size_t> Cshape;
    std::vector<size_t> Dshape;

    DataType ADtype;
    DataType BDtype;
    DataType CDtype;
    DataType DDtype;

    size_t dim;
    size_t batch_size;
    size_t m;
    size_t k;
    size_t n;

    float alpha = 1.0f;
    float beta  = 0.0f;

    size_t lda;
    size_t stride_a;
    size_t ldb;
    size_t stride_b;
    size_t ldc;
    size_t stride_c;

    GemmArguments(const GemmParams& params) {

        Ashape = params.A.shape();
        Bshape = params.B.shape();

        if (params.transA == TransposeOperation::TRANSPOSE) {
            std::iter_swap(Ashape.end() - 1, Ashape.end() - 2);
        }

        if (params.transB == TransposeOperation::TRANSPOSE) {
            std::iter_swap(Bshape.end() - 1, Bshape.end() - 2);
        }

        if (params.C != std::nullopt) {
            Cshape = params.C.value().get().shape();
        }

        ADtype = params.A.type();
        BDtype = params.A.type();
        if (params.C != std::nullopt) {
            CDtype = params.C.value().get().type();
        }
        DDtype = (params.compute_type == DataType::TYPE_INVALID) ? params.A.type() : params.compute_type;

        dim        = params.A.dim();
        batch_size = std::accumulate(Ashape.begin(), Ashape.end() - 2, (size_t)1, std::multiplies<size_t>());

        m = Ashape[dim - 2];
        k = Ashape[dim - 1];
        n = Bshape[dim - 1];

        Dshape = std::vector<size_t>(Ashape.begin(), Ashape.end() - 2);
        Dshape.insert(Dshape.end(), {m, n});

        lda      = params.A.shape()[dim - 1];
        stride_a = m * k;
        ldb      = params.B.shape()[dim - 1];
        stride_b = k * n;
        ldc      = n;
        stride_c = m * n;
    }

    void dump() {
        std::cerr << "Ashape is : " << ShapeStringView(Ashape) << "\n"
                  << "Bshape is : " << ShapeStringView(Bshape) << "\n"
                  << "Cshape is : " << ShapeStringView(Cshape) << "\n"
                  << "Dshape is : " << ShapeStringView(Dshape) << "\n"
                  << "dim is : " << dim << "\n"
                  << "batch size is : " << batch_size << "\n"
                  << "m is : " << m << "\n"
                  << "n is : " << n << "\n"
                  << "k is : " << k << "\n"
                  << "lda is : " << lda << "\n"
                  << "ldb is : " << ldb << "\n"
                  << "ldc is : " << ldc << "\n"
                  << "stride_a is : " << stride_a << "\n"
                  << "stride_b is : " << stride_b << "\n"
                  << "stride_c is : " << stride_c << "\n"
                  << std::endl;
    }
};

BufferPtr ROCmDevice::gemm(const GemmParams& params) {
    params.check();

    GemmArguments arguments(params);

    std::cerr << "gemm " << int(params.A.type()) << " " << int(params.compute_type) << '\n';
    arguments.dump();

    BufferPtr output;
    if (params.D) {
        output = params.D;
    } else {
        output = allocateBuffer({arguments.DDtype, arguments.Dshape, AllocationType::DEVICE}, {"gemm_output"});
    }
    return output;
}

SelectOutput ROCmDevice::select(const SelectParams& params) {
    if ((params.input.where() != MemoryType::MEMORY_GPU) || (params.dim > 0)) {
        return DeviceBase::select(params);
    }

    RUNTIME_ASSERT_OP_ARG(params.index.type() == DataType::TYPE_INT32, "Select index must be int32.");
    RUNTIME_ASSERT_OP_ARG(params.dim == 0, "select op tmp only support dim == 0");

    const auto& input         = params.input;
    auto        output_shape  = input.shape();
    output_shape[0]           = params.index.size();
    auto num_selected_element = input.size() / input.shape()[0];
    auto output               = allocateBuffer({input.type(), output_shape});
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(input.type(),
                                     invokeLookupHiddenStateOfLastToken,
                                     output->data(),
                                     input.data(),
                                     (int*)params.index.data(),
                                     (int)params.index.size(),
                                     num_selected_element,
                                     0,
                                     stream_);
    return std::move(output);
}

BufferPtr ROCmDevice::embeddingLookup(const EmbeddingLookupParams& params) {
    const auto& tokens           = params.combo_tokens;
    const auto& embedding_table  = params.embedding_table;
    const auto& position_ids     = params.position_ids;
    const auto& postition_table  = params.position_table;
    const auto& token_types      = params.token_types;
    const auto& token_type_table = params.token_type_table;

    const auto token_num   = tokens.size();
    const auto hidden_size = embedding_table.shape()[1];
    const auto data_type   = embedding_table.type();

    auto embeddings = allocateBuffer({data_type, {token_num, hidden_size}}, {"embedding"});

    DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                     invokeEmebeddingLookup,
                                     embeddings->data(),
                                     embedding_table.data(),
                                     params.input_embedding_scalar,
                                     postition_table.has_value() ? postition_table.value().get().data() : nullptr,
                                     token_type_table.has_value() ? token_type_table.value().get().data() : nullptr,
                                     tokens.data<int>(),
                                     position_ids.has_value() ? position_ids.value().get().data<int>() : nullptr,
                                     token_types.has_value() ? token_types.value().get().data<int>() : nullptr,
                                     token_num,
                                     hidden_size,
                                     stream_);

    return std::move(embeddings);
}

LayernormOutput ROCmDevice::layernorm(const LayernormParams& params) {
    const auto& input   = params.input;
    auto&       output  = params.norm_output;
    const auto& weights = params.weights;
    const auto& gamma   = weights ? weights->get().gamma.get()->data() : nullptr;
    const auto& beta    = (weights && weights->get().beta) ? weights->get().beta.get()->data() : nullptr;

    const auto norm_type = params.norm_type;
    const auto data_type = input.type();
    const auto m         = input.shape()[0];
    const auto n         = input.shape()[1];
    const auto eps       = params.eps;

    if (!weights.has_value()) {
        if (params.alpha.has_value() || (norm_type == NormType::alphanorm)) {
            const auto alpha = params.alpha.value_or(1.0f);
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                             invokeAlphaAddBiasResidual,
                                             output.data(),
                                             input.data(),
                                             params.residual1 ? params.residual1.value().get().data() : nullptr,
                                             params.bias ? params.bias.value().get().data() : nullptr,
                                             alpha,
                                             m,
                                             n,
                                             stream_);
        } else if (params.bias.has_value() || params.residual1.has_value() || params.residual2.has_value()) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                             invokeAddBiasResidual,
                                             output.data(),
                                             input.data(),
                                             params.residual1 ? params.residual1.value().get().data() : nullptr,
                                             params.residual2 ? params.residual2.value().get().data() : nullptr,
                                             params.bias.has_value() ? params.bias.value().get().data() : nullptr,
                                             nullptr,  // scale_inter
                                             nullptr,  // scale_out
                                             m,
                                             n,
                                             stream_);
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
        return;
    }

    if (!(norm_type == NormType::layernorm || norm_type == NormType::rmsnorm)) {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
    if (params.residual1.has_value() || params.bias.has_value()) {
        const auto& add_bias_output = params.add_bias_output ? params.add_bias_output.value().get() : output;
        if (params.norm_type == NormType::layernorm) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                             invokeGeneralAddBiasResidualLayerNorm,
                                             add_bias_output.data(),
                                             output.data(),
                                             input.data(),
                                             params.bias ? params.bias.value().get().data() : nullptr,
                                             params.residual1 ? params.residual1.value().get().data() : nullptr,
                                             gamma,
                                             beta,
                                             eps,
                                             m,
                                             n,
                                             stream_,
                                             true,     // use_diff_of_squares
                                             nullptr,  // scale
                                             nullptr,  // dynamic_scale
                                             nullptr   // out_quant
            );
        } else if (params.norm_type == NormType::rmsnorm) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                             invokeAddBiasResidualRmsNorm,
                                             add_bias_output.data(),
                                             output.data(),
                                             input.data(),
                                             params.bias ? params.bias.value().get().data() : nullptr,
                                             params.residual1 ? params.residual1.value().get().data() : nullptr,
                                             gamma,
                                             beta,
                                             eps,
                                             m,
                                             n,
                                             stream_,
                                             nullptr,  // scale
                                             nullptr,  // dynamic_scale
                                             nullptr   // out_quant
            );
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    } else {
        if (params.norm_type == NormType::layernorm) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                             invokeGeneralLayerNorm,
                                             output.data(),
                                             input.data(),
                                             gamma,
                                             beta,
                                             eps,
                                             m,
                                             n,
                                             stream_,
                                             true,     // use_diff_of_squares
                                             nullptr,  // scale
                                             nullptr,  // dynamic_scale
                                             nullptr   // out_quant
            );
        } else if (params.norm_type == NormType::rmsnorm) {
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(data_type,
                                             invokeGeneralRmsNorm,
                                             output.data(),
                                             input.data(),
                                             gamma,
                                             beta,
                                             eps,
                                             m,
                                             n,
                                             stream_,
                                             nullptr,  // scale
                                             nullptr,  // dynamic_scale
                                             nullptr   // out_quant
            );
        } else {
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
        }
    }
    return;
}

#define ARGS_DISPATCH(Atype, Dtype, out, bias, gate, gate_bias, m, n, stream)                                          \
    do {                                                                                                               \
        invokeGenericActivation<Atype>((Dtype*)out,                                                                    \
                                       (const Dtype*)bias,                                                             \
                                       (const Dtype*)gate,                                                             \
                                       (const Dtype*)gate_bias,                                                        \
                                       (const int*)nullptr,                                                            \
                                       (const Dtype*)nullptr,                                                          \
                                       (int)m,                                                                         \
                                       (int)n,                                                                         \
                                       0,                                                                              \
                                       (const float*)nullptr,                                                          \
                                       (const float*)nullptr,                                                          \
                                       (const Dtype*)nullptr,                                                          \
                                       stream);                                                                        \
    } while (0)

#define ATYPE_DISPATCH(Dtype, Atype, KERNEL, ...)                                                                      \
    do {                                                                                                               \
        if (Atype == ActivationType::Silu) {                                                                           \
            KERNEL(SiluActivation, Dtype, __VA_ARGS__);                                                                \
        } else if (Atype == ActivationType::Gelu) {                                                                    \
            KERNEL(GeluActivation, Dtype, __VA_ARGS__);                                                                \
        } else if (Atype == ActivationType::Swiglu) {                                                                  \
            KERNEL(SiluActivation, Dtype, __VA_ARGS__);                                                                \
        } else {                                                                                                       \
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);                                                       \
        }                                                                                                              \
    } while (0)

#define DTYPE_DISPATCH(Dtype, ...)                                                                                     \
    do {                                                                                                               \
        if (Dtype == DataType::TYPE_FP16) {                                                                            \
            ATYPE_DISPATCH(half, __VA_ARGS__);                                                                         \
        } else {                                                                                                       \
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);                                                       \
        }                                                                                                              \
    } while (0)

#define DISPATCH(Dtype, Atype, ...)                                                                                    \
    do {                                                                                                               \
        DTYPE_DISPATCH(Dtype, Atype, ARGS_DISPATCH, __VA_ARGS__);                                                      \
    } while (0)

void ROCmDevice::activation(const ActivationParams& params) {
    const auto& states = params.states;
    size_t      m      = states.shape()[0];
    size_t      n      = states.shape()[1];

    void* bias      = nullptr;
    void* gate      = nullptr;
    void* gate_bias = nullptr;

    if (params.bias) {
        bias = params.bias.value().get().data();
    }

    if (params.gate) {
        gate = params.gate.value().get().data();
    }

    if (params.gate_bias) {
        gate_bias = params.gate_bias.value().get().data();
    }

    DISPATCH(states.type(), params.atype, states.data(), bias, gate, gate_bias, m, n, stream_);
}

AttentionModuleOutput ROCmDevice::contextAttention(const AttentionModuleParams& params) {
    std::cerr << "contextAttention\n";
}

RTP_LLM_REGISTER_DEVICE(ROCm);

}  // namespace fastertransformer
