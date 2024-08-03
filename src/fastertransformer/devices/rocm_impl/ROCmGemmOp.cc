#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"
#include "src/fastertransformer/devices/rocm_impl/ROCmAllocator.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/devices/CommonDefines.h"
#include "src/fastertransformer/utils/ShapeCheck.h"
#include "autil/StringUtil.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/rocm/quantizePreprocessors.h"
#include "src/fastertransformer/kernels/quantization_tensor.h"

#include <numeric>
#include <utility>

using namespace std;

namespace fastertransformer {
using namespace rocm;

hipblasOperation_t opConvert(TransposeOperation op) {
    switch (op) {
        case TransposeOperation::NONE:
            return hipblasOperation_t::HIPBLAS_OP_N;
        case TransposeOperation::TRANSPOSE:
            return hipblasOperation_t::HIPBLAS_OP_T;
        default:
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
};

hipblasDatatype_t dtypeConvert(DataType dtype) {
    switch (dtype) {
        case DataType::TYPE_FP16:
            return hipblasDatatype_t::HIPBLAS_R_16F;
        case DataType::TYPE_FP32:
            return hipblasDatatype_t::HIPBLAS_R_32F;
        default:
            throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
};

struct ROCmGemmDispatch {

    enum GemmImplementType {
        hipblas_basic_gemm,
        hipblas_batch_gemm,
        WeightOnlyQuantMatmulPlugin,
        invalid,
    };

    static GemmImplementType dispatch(const GemmParams& params) {
        size_t dim = params.A.dim();
        if (params.C != std::nullopt) {
            return GemmImplementType::invalid;
        }
        if (dim == 2 && params.A.isFloat() && params.B.isFloat()) {

            return GemmImplementType::hipblas_basic_gemm;
        } else if (dim > 2 && params.A.isFloat() && params.B.isFloat()) {

            return GemmImplementType::hipblas_batch_gemm;
        } else if (dim == 2 && (params.A.type() == DataType::TYPE_FP16 || params.A.type() == DataType::TYPE_BF16)
                   && params.B.type() == DataType::TYPE_QINT8) {
            return GemmImplementType::WeightOnlyQuantMatmulPlugin;
        }
        return GemmImplementType::invalid;
    }
};

struct ROCmGemmArguments {
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

    ROCmGemmArguments(const GemmParams& params) {

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
        std::cout << "Ashape is : " << ShapeStringView(Ashape) << "\n"
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

/// @brief   basic gemm ops
/// @details D = alpha * op(A) * op(B) + beta * C
///          A [b, ..., m, k]
///          B [b, ..., k, n]
///          C [b, ..., m, n]

float cpu_half2float(uint16_t h)
{
  unsigned sign = ((((uint16_t)h) >> 15) & 1);
  unsigned exponent = ((((uint16_t)h) >> 10) & 0x1f);
  unsigned mantissa = ((((uint16_t)h) & 0x3ff) << 13);

  if (exponent == 0x1f) { /* NaN or Inf */
    mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
    exponent = 0xff;
  } else if (!exponent) { /* Denorm or Zero */
    if (mantissa) {
      unsigned int msb;
      exponent = 0x71;
      do {
        msb = (mantissa & 0x400000);
        mantissa <<= 1; /* normalize */
        --exponent;
      } while (!msb);
      mantissa &= 0x7fffff; /* 1.mantissa is implicit */
    }
  } else {
    exponent += 0x70;
  }

  int temp = ((sign << 31) | (exponent << 23) | mantissa);
  return *((float*)((void*)&temp));
}
BufferPtr ROCmDevice::gemm(const GemmParams& params) {
    params.check();

    using GemmImplementType = ROCmGemmDispatch::GemmImplementType;
    ROCmGemmArguments arguments(params);

    BufferPtr output;
    if (params.D) {
        output = params.D;
        RUNTIME_ASSERT_OP_ARG((arguments.DDtype == params.D->type()) && (arguments.Dshape == params.D->shape()),
                              "Gemm output D shape and dtype mismatch: expected [%d][%s] but got [%s]",
                              arguments.DDtype,
                              autil::StringUtil::toString(arguments.Dshape).c_str(),
                              params.D->debugString().c_str());
    } else {
        output = allocateBuffer({arguments.DDtype, arguments.Dshape, AllocationType::DEVICE}, {"gemm_output"});
    }


    if (params.dispatch() == GemmType::BufferA_QBufferB_BufferC_2DGemm) {
        if (reinterpret_cast<const QBuffer&>(params.B).zerosData() != nullptr) {
            FT_CHECK(reinterpret_cast<const QBuffer&>(params.B). scales().dim() == 2);
            size_t kernel_dim0 = params.B.shape()[0];
            size_t scales_dim0 = reinterpret_cast<const QBuffer&>(params.B). scales().shape()[0];
            FT_CHECK((kernel_dim0 % scales_dim0 == 0));
            size_t group_size = (kernel_dim0 / scales_dim0);
            FT_CHECK((group_size == 64 || group_size == 128));
            size_t type_bits = getTypeBits(params.B.type());
            FT_CHECK((type_bits == 4 || type_bits == 8));

            BUFFER_DTYPE_CHECK(params.A, {DataType::TYPE_FP16, DataType::TYPE_BF16});
            BUFFER_DTYPE_CHECK(params.B, {DataType::TYPE_QINT4X2});

            const QBuffer& QB  = reinterpret_cast<const QBuffer&>(params.B);
            auto           fpB = allocateBuffer({params.A.type(), {params.B.shape()}, AllocationType::DEVICE}, {"fpB"});

            printf("[GEMM]BufferA_QBufferB_BufferC_2DGemm:TYPE_QINT4X2\n");
            printf("[GEMM] A = %s\n", params.A.debugString().c_str());
            printf("[GEMM] kernel = %s\n", QB.kernel().debugString().c_str());
            printf("[GEMM] scales = %s\n", QB.scales().debugString().c_str());
            printf("[GEMM] zeros = %s\n", QB.zeros().debugString().c_str());

            // dequant B
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(params.A.type(),
                                             invokePerColDequantizationInt4x2,
                                             fpB.get()->data(),
                                             (int8_t*)(QB.kernel().data()),
                                             arguments.k,
                                             arguments.n,
                                             QB.scales().data<half>(),
                                             QB.zeros().data<half>(),
                                             group_size,
                                             stream_);
            sync_check_cuda_error();

            const auto A = params.A.data();
            const auto B = fpB.get()->data();
            auto       D = output->data();

            auto a_op = opConvert(params.transA);
            auto b_op = opConvert(params.transB);

            auto A_data_type = dtypeConvert(arguments.ADtype);
            auto B_data_type = dtypeConvert(fpB.get()->type());
            auto D_data_type = dtypeConvert(arguments.DDtype);
            auto computeType = dtypeConvert(arguments.DDtype);
            
            printf("[GEMM] n = %d, m = %d, k = %d\n", arguments.n, arguments.m, arguments.k);
            printf("[GEMM] ldc = %d, stride_c = %d, batch_size = %d\n", arguments.ldc, arguments.stride_c, arguments.batch_size);
            /*printf("------------------------------------\n");
            BufferPtr hB = clone({QB.kernel(), AllocationType::HOST});
            int8_t* phB = (int8_t*)(hB.get()->data());
            size_t Bsz = QB.kernel().size();
            for(size_t i = 0; i < Bsz; i++)
            {
                if(i%8 == 0) printf("\n");
                int8_t tmpu8 = phB[i];
                printf("%d, ", tmpu8);
            }
            printf("\n------------------------------------\n");*/
            /*printf("------------------------------------\n");
            printf("[GEMM] int4 = %s\n", QB.kernel().debugString().c_str());
            BufferPtr hB = clone({QB.kernel(), AllocationType::HOST});
            uint8_t* phB = (uint8_t*)(hB.get()->data());
            size_t Bsz = QB.kernel().size();
            for(size_t i = 0; i < Bsz/2; i++)
            {
                if(i%(QB.kernel().shape()[1]/2) == 0) printf("\n");
                uint8_t tmpu8 = phB[i];
                uint8_t tmpu4l = tmpu8 & 0x0F;
                uint8_t tmpu4h = (tmpu8 & 0xF0) >> 4;
                if(tmpu4l & 0x08) tmpu4l|= 0xF0;
                if(tmpu4h & 0x08) tmpu4h|= 0xF0;
                int8_t tmpi4l = tmpu4l;
                int8_t tmpi4h = tmpu4h;
                printf("%d, %d, ", tmpi4l, tmpi4h);
            }
            printf("\n------------------------------------\n");*/
            /*printf("------------------------------------\n");
            printf("[GEMM] scale = %s\n", QB.scales().debugString().c_str());
            BufferPtr hB = clone({QB.scales(), AllocationType::HOST});
            uint16_t* phB = (uint16_t*)(hB.get()->data());
            size_t Bsz = QB.scales().size();
            for(size_t i = 0; i < Bsz; i++)
            {
                if(i%(QB.scales().shape()[1]) == 0) printf("\n");
                printf("%.2e, ", cpu_half2float(phB[i]));
            }
            printf("\n------------------------------------\n");*/
            /*printf("------------------------------------\n");
            printf("[GEMM] dequantB = %s\n", fpB.get()->debugString().c_str());
            BufferPtr hB = clone({*fpB.get(), AllocationType::HOST});
            uint16_t* phB = (uint16_t*)(hB.get()->data());
            size_t Bsz = fpB.get()->size();
            for(size_t i = 0; i < Bsz; i++)
            {
                if(i%(fpB.get()->shape()[1]) == 0) printf("\n");
                printf("%.2e, ", cpu_half2float(phB[i]));
            }
            printf("\n------------------------------------\n");*/
            //std::exit(0);

#if 1 // B * A
            hipblas_mm_wrapper_->stridedBatchedGemm(b_op,
                                                    a_op,
                                                    arguments.n,
                                                    arguments.m,
                                                    arguments.k,
                                                    arguments.alpha,
                                                    B,
                                                    B_data_type,
                                                    arguments.ldb,
                                                    arguments.stride_b,
                                                    A,
                                                    A_data_type,
                                                    arguments.lda,
                                                    arguments.stride_a,
                                                    arguments.beta,
                                                    D,
                                                    D_data_type,
                                                    arguments.ldc,
                                                    arguments.stride_c,
                                                    arguments.batch_size,
                                                    computeType);
#else
            hipblas_mm_wrapper_->stridedBatchedGemm(a_op,
                                                    b_op,
                                                    arguments.m,
                                                    arguments.n,
                                                    arguments.k,
                                                    arguments.alpha,
                                                    A,
                                                    A_data_type,
                                                    arguments.lda,
                                                    arguments.stride_a,
                                                    B,
                                                    B_data_type,
                                                    arguments.ldb,
                                                    arguments.stride_b,
                                                    arguments.beta,
                                                    D,
                                                    D_data_type,
                                                    arguments.ldc,
                                                    arguments.stride_c,
                                                    arguments.batch_size,
                                                    computeType);
#endif                                                    


            /*printf("------------------------------------\n");
            printf("[GEMM] output = %s\n", output.get()->debugString().c_str());
            BufferPtr hD = clone({*output.get(), AllocationType::HOST});
            uint16_t* phD = (uint16_t*)(hD.get()->data());
            size_t Dsz = output.get()->size();
            for(size_t i = 0; i < Dsz; i++)
            {
                if(i%(output.get()->shape()[1]) == 0) printf("\n");
                printf("%.2e, ", cpu_half2float(phD[i]));
            }
            printf("\n------------------------------------\n");*/
            //std::exit(0);

            sync_check_cuda_error();
            return move(output);
        }
        else {
            printf("[GEMM]GEMM_ERROR: BufferA_QBufferB_BufferC_2DGemm\n");
        }
    }

#if 0
    if (params.dispatch() == GemmType::BufferA_QBufferB_BufferC_2DGemm) {
        printf("[GEMM]BufferA_QBufferB_BufferC_2DGemm\n");
        if (params.B.type() == DataType::TYPE_QINT8) {
            printf("[GEMM]BufferA_QBufferB_BufferC_2DGemm:TYPE_QINT8\n");
            BUFFER_DTYPE_CHECK(params.A, {DataType::TYPE_FP16, DataType::TYPE_BF16});
            BUFFER_DTYPE_CHECK(params.B, {DataType::TYPE_QINT8});

            const QBuffer& QB  = reinterpret_cast<const QBuffer&>(params.B);
            auto           fpB = allocateBuffer({params.A.type(), {params.B.shape()}, AllocationType::DEVICE}, {"fpB"});

            // dequant B
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(params.A.type(),
                                             invokePerColDequantizationInt8,
                                             fpB.get()->data(),
                                             QB.kernel().data<int8_t>(),
                                             arguments.k,
                                             arguments.n,
                                             QB.scales().data<half>(),
                                             nullptr,
                                             nullptr,
                                             stream_);
            sync_check_cuda_error();

            const auto A = params.A.data();
            const auto B = fpB.get()->data();
            auto       D = output->data();

            auto a_op = opConvert(params.transA);
            auto b_op = opConvert(params.transB);

            auto A_data_type = dtypeConvert(arguments.ADtype);
            auto B_data_type = dtypeConvert(fpB.get()->type());
            auto D_data_type = dtypeConvert(arguments.DDtype);
            auto computeType = dtypeConvert(arguments.DDtype);
            
            hipblas_mm_wrapper_->stridedBatchedGemm(b_op,
                                                    a_op,
                                                    arguments.n,
                                                    arguments.m,
                                                    arguments.k,
                                                    arguments.alpha,
                                                    B,
                                                    B_data_type,
                                                    arguments.ldb,
                                                    arguments.stride_b,
                                                    A,
                                                    A_data_type,
                                                    arguments.lda,
                                                    arguments.stride_a,
                                                    arguments.beta,
                                                    D,
                                                    D_data_type,
                                                    arguments.ldc,
                                                    arguments.stride_c,
                                                    arguments.batch_size,
                                                    computeType);

            sync_check_cuda_error();
            return move(output);
        }
        if (params.B.type() == DataType::TYPE_QINT4X2) {
            printf("[GEMM]BufferA_QBufferB_BufferC_2DGemm:TYPE_QINT4X2\n");
            BUFFER_DTYPE_CHECK(params.A, {DataType::TYPE_FP16, DataType::TYPE_BF16});
            BUFFER_DTYPE_CHECK(params.B, {DataType::TYPE_QINT4X2});

            const QBuffer& QB  = reinterpret_cast<const QBuffer&>(params.B);
            auto           fpB = allocateBuffer({params.A.type(), {params.B.shape()}, AllocationType::DEVICE}, {"fpB"});

            printf("[GEMM] kernel = %s\n", QB.kernel().debugString().c_str());
            printf("[GEMM] scales = %s\n", QB.scales().debugString().c_str());
            if(QB.zeros().size() > 0)
                printf("[GEMM] zeros = %s\n", QB.zeros().debugString().c_str());
            else
                printf("[GEMM] zeros = null_zeros\n");

            if(QB.kernel().shape()[0] != QB.scales().shape()[0] * 128)
                printf("[GEMM] GEMM_ERROR group_scale not 128\n");

            // dequant B
            DISPATCH_CUDA_FUNCTION_DATA_TYPE(params.A.type(),
                                             invokePerColDequantizationInt4x2,
                                             fpB.get()->data(),
                                             (int8_t*)(QB.kernel().data()),
                                             arguments.k,
                                             arguments.n,
                                             QB.scales().data<half>(),
                                             nullptr,
                                             nullptr,
                                             stream_);
            sync_check_cuda_error();

            const auto A = params.A.data();
            const auto B = fpB.get()->data();
            auto       D = output->data();

            auto a_op = opConvert(params.transA);
            auto b_op = opConvert(params.transB);

            auto A_data_type = dtypeConvert(arguments.ADtype);
            auto B_data_type = dtypeConvert(fpB.get()->type());
            auto D_data_type = dtypeConvert(arguments.DDtype);
            auto computeType = dtypeConvert(arguments.DDtype);
            
            printf("[GEMM] b_op = %d, a_op = %d\n", params.transB, params.transA);
            printf("[GEMM] n = %d, m = %d, k = %d\n", arguments.n, arguments.m, arguments.k);
            hipblas_mm_wrapper_->stridedBatchedGemm(b_op,
                                                    a_op,
                                                    arguments.n,
                                                    arguments.m,
                                                    arguments.k,
                                                    arguments.alpha,
                                                    B,
                                                    B_data_type,
                                                    arguments.ldb,
                                                    arguments.stride_b,
                                                    A,
                                                    A_data_type,
                                                    arguments.lda,
                                                    arguments.stride_a,
                                                    arguments.beta,
                                                    D,
                                                    D_data_type,
                                                    arguments.ldc,
                                                    arguments.stride_c,
                                                    arguments.batch_size,
                                                    computeType);

            printf("[GEMM] D = %s\n", output.get()->debugString().c_str());

            sync_check_cuda_error();
            return move(output);
        }
        else{
            printf("[GEMM]GEMM_ERROR: BufferA_QBufferB_BufferC_2DGemm\n");
        }
    }
#endif
    auto A_data_type = dtypeConvert(arguments.ADtype);
    auto B_data_type = dtypeConvert(arguments.BDtype);
    auto D_data_type = dtypeConvert(arguments.DDtype);
    auto computeType = HIPBLAS_R_32F;

    if (params.compute_type == DataType::TYPE_INVALID) {
        computeType = HIPBLAS_R_32F;
        hipblasMMWrapperPtr()->setGemmConfig(A_data_type, B_data_type, D_data_type, HIPBLAS_R_32F);
    } else {
        computeType = dtypeConvert(arguments.DDtype);
        hipblasMMWrapperPtr()->setGemmConfig(A_data_type, B_data_type, D_data_type, dtypeConvert(params.compute_type));
    }

    if (ROCmGemmDispatch::dispatch(params) == GemmImplementType::hipblas_basic_gemm) {

        const auto A    = params.A.data();
        const auto B    = params.B.data();
        auto       D    = output->data();
        auto       a_op = opConvert(params.transA);
        auto       b_op = opConvert(params.transB);

        hipblas_mm_wrapper_->Gemm(
            b_op, a_op, arguments.n, arguments.m, arguments.k, B, arguments.ldb, A, arguments.lda, D, arguments.ldc);
        sync_check_hip_error();

        printf("[GEMM]hipblas_basic_gemm\n");
        printf("[GEMM] A = %s\n", params.A.debugString().c_str());
        printf("[GEMM] B = %s\n", params.B.debugString().c_str());
        printf("[GEMM] D = %s\n", output.get()->debugString().c_str());
        printf("[GEMM] n = %d, m = %d, k = %d\n", arguments.n, arguments.m, arguments.k);
        printf("[GEMM] ldc = %d, stride_c = %d, batch_size = %d\n", arguments.ldc, arguments.stride_c, arguments.batch_size);
        /*printf("------------------------------------\n");
        printf("[GEMM] output = %s\n", params.B.debugString().c_str());
        BufferPtr hB = clone({params.B, AllocationType::HOST});
        uint16_t* phB = (uint16_t*)(hB.get()->data());
        size_t Bsz = params.B.size();
        for(size_t i = 0; i < Bsz; i++)
        {
            if(i%(hB.get()->shape()[1]) == 0) printf("\n");
            printf("%.2e, ", cpu_half2float(phB[i]));
        }
        printf("\n------------------------------------\n");*/
        /*printf("------------------------------------\n");
        printf("[GEMM] output = %s\n", output.get()->debugString().c_str());
        BufferPtr hB = clone({*output.get(), AllocationType::HOST});
        uint16_t* phB = (uint16_t*)(hB.get()->data());
        size_t Bsz = output.get()->size();
        for(size_t i = 0; i < Bsz; i++)
        {
            if(i%(hB.get()->shape()[1]) == 0) printf("\n");
            printf("%.2e, ", cpu_half2float(phB[i]));
        }
        printf("\n------------------------------------\n");*/
        //std::exit(0);

        return std::move(output);
    } else if (ROCmGemmDispatch::dispatch(params) == GemmImplementType::hipblas_batch_gemm) {

        // convert buffers to ptrs
        const auto A = params.A.data();
        const auto B = params.B.data();
        auto       D = output->data();

        auto a_op = opConvert(params.transA);
        auto b_op = opConvert(params.transB);

        auto A_data_type = dtypeConvert(arguments.ADtype);
        auto B_data_type = dtypeConvert(arguments.BDtype);
        auto D_data_type = dtypeConvert(arguments.DDtype);
        auto computeType = dtypeConvert(arguments.DDtype);

        hipblas_mm_wrapper_->stridedBatchedGemm(b_op,
                                                a_op,
                                                arguments.n,
                                                arguments.m,
                                                arguments.k,
                                                arguments.alpha,
                                                B,
                                                B_data_type,
                                                arguments.ldb,
                                                arguments.stride_b,
                                                A,
                                                A_data_type,
                                                arguments.lda,
                                                arguments.stride_a,
                                                arguments.beta,
                                                D,
                                                D_data_type,
                                                arguments.ldc,
                                                arguments.stride_c,
                                                arguments.batch_size,
                                                computeType);
        sync_check_hip_error();
        return std::move(output);
    } else {
        throw OpException(OpErrorType::ERROR_UNIMPLEMENTED);
    }
    printf("[GEMM]GEMM_ERROR\n");
    return std::move(output);
}

}  // namespace fastertransformer
