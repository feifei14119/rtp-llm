#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/devices/base_tests/GemmOpTest.hpp"
#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"

using namespace std;
using namespace fastertransformer;

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

class ROCmGemmOpTest: public GemmOpTest {
public:
    struct QGemmOpTestInput {
        torch::Tensor A;
        torch::Tensor Wk;
        torch::Tensor Ws;
        torch::Tensor Wz;
    };
    /*GemmOpTestInput PrepareQGemmOpInput(size_t m,
                                       size_t n,
                                       size_t k,
                                       DataType type)
    {
        auto dtype = dataTypeToTorchType(type);
        auto A = torch::rand({(int)m, (int)k}, torch::Device(torch::kCPU)).to(dtype);
        auto B = torch::rand({(int)k, (int)n}, torch::Device(torch::kCPU)).to(dtype);

        torch::Tensor qweight_unprocessed = torch::randint((int)0xFFFFFFFF, (int)0x0FFFFFFF, {(long)k/8, (long)n});
        torch::print(qweight_unprocessed);
        //scale = torch.rand((gemm_k // group_size, gemm_n), dtype=compute_type) * 2
        //zeros = torch.rand((gemm_k // group_size, gemm_n), dtype=compute_type) * 2
        torch::Tensor qweight_int8 = self.woq_groupwise_extract_int4(qweight_unprocessed, uint4_input).char()

      w_packed_int8 = w_packed.T.contiguous().view(torch.uint8)
      w_unpacked_int4 = torch.stack(((w_packed_int8 % 16).view(-1, 1), (w_packed_int8 // 16).view(-1, 1)), dim=1)

      # Unpacked uint4s
      w_unpacked_int4 = w_unpacked_int4.flatten().view(w_packed.shape[1], -1).T.contiguous().int()

        return GemmOpTestInput({A, B});
    }*/

    /*GemmOpTestOutput RocmQ8GemmOpRun(GemmOpTestInput& input) {
        auto       A   = tensorToBuffer(input.A);
        auto       B   = tensorToBuffer(input.B);
        auto       Q8B = device_->quantize({*B, DataType::TYPE_QINT8, 1});
        auto       D   = device_->allocateBuffer({A->type(), {A->shape()[0], Q8B->shape()[1]}});
        GemmParams params{*A, *Q8B, std::nullopt, D};
        device_->gemm(params);
        return GemmOpTestOutput({bufferToTensor(*D)});
    }
    GemmOpTestOutput RocmQ4x2GemmOpRun(GemmOpTestInput& input) {
        auto       A   = tensorToBuffer(input.A);
        auto       B   = tensorToBuffer(input.B);
        auto       Q4B = device_->quantize({*B, DataType::TYPE_QINT4X2, 1});
        auto       D   = device_->allocateBuffer({A->type(), {A->shape()[0], Q4B->shape()[1]}});
        GemmParams params{*A, *Q4B, std::nullopt, D};
        device_->gemm(params);
        return GemmOpTestOutput({bufferToTensor(*D)});
    }
    void RocmQ8GemmOpTest(size_t m, size_t n, size_t k, DataType dtype) {
        auto input      = PrepareGemmOpInput(m, n, k, dtype);
        auto result     = RocmQ8GemmOpRun(input);
        auto result_ref = BasicGemmTorchRefRun(input);
        assertTensorClose(result.C.to(result_ref.C.type()), result_ref.C, 1e-1, 1e-1);
    }
    void RocmQ4x2GemmOpTest0(size_t m, size_t n, size_t k, DataType dtype) {
        auto input      = PrepareGemmOpInput(m, n, k, dtype);
        auto result     = RocmQ4x2GemmOpRun(input);
        auto result_ref = BasicGemmTorchRefRun(input);
        assertTensorClose(result.C.to(result_ref.C.type()), result_ref.C, 1, 1);
    }*/

    void RocmQ4x2GemmOpTest(size_t m, size_t n, size_t k, DataType dtype) {
        auto ptype = dataTypeToTorchType(dtype);
        torch::Tensor A = torch::rand({(int)m, (int)k}, torch::Device(torch::kCPU)).to(ptype);
        printf("\n---------------------------------------\n");
        torch::print(A);
        BufferPtr bpA = tensorToBuffer(A, AllocationType::HOST);
        printf("\n---------------------------------------\n");
        printf("bpA:\n %s\n", bpA.get()->debugString().c_str());
        printf("------------------------------------\n");
        uint16_t* phA = (uint16_t*)(bpA.get()->data());
        size_t Asz = bpA.get()->size();
        for(size_t i = 0; i < Asz; i++)
        {
            if(i%8 == 0) printf("\n");
            printf("%.2e, ", cpu_half2float(phA[i]));
        }
        printf("\n------------------------------------\n");
    }
};

TEST_F(ROCmGemmOpTest, QuantGemmOpTest) {
    //RocmQ8GemmOpTest(64, 64, 64, DataType::TYPE_FP16);
    //RocmQ4x2GemmOpTest(64, 64, 64, DataType::TYPE_FP16);
    RocmQ4x2GemmOpTest(1, 8, 16, DataType::TYPE_FP16);
    //BasicGemmOpTest(2, 4, 8, DataType::TYPE_FP16);
}

#if 0
TEST_F(ROCmGemmOpTest, BasicGemmOpTest) {
    BasicGemmOpTest(2, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(8, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(1024, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(4096, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(2, 1024, 2048, DataType::TYPE_FP32);
    BasicGemmOpTest(8, 1024, 2048, DataType::TYPE_FP32);
    BasicGemmOpTest(1024, 1024, 2048, DataType::TYPE_FP32);
    BasicGemmOpTest(4096, 1024, 2048, DataType::TYPE_FP32);
}

TEST_F(ROCmGemmOpTest, QuantGemmOpTest) {
    RocmQ8GemmOpTest(64, 64, 64, DataType::TYPE_FP16);
    RocmQ8GemmOpTest(2, 1024, 2048, DataType::TYPE_FP16);
    RocmQ4x2GemmOpTest(64, 64, 64, DataType::TYPE_FP16);
    RocmQ4x2GemmOpTest(2, 1024, 2048, DataType::TYPE_FP16);
}

TEST_F(ROCmGemmOpTest, TransposeGemmOpTest) {
    auto   tran = TransposeOperation::TRANSPOSE;
    auto   none = TransposeOperation::NONE;
    size_t m    = 5;
    size_t n    = 1024;
    size_t k    = 4096;
    TransposeGemmOpTest(none, none, m, k, k, n, DataType::TYPE_FP16);
    TransposeGemmOpTest(none, tran, m, k, n, k, DataType::TYPE_FP16);
    TransposeGemmOpTest(tran, tran, k, m, n, k, DataType::TYPE_FP16);
    TransposeGemmOpTest(tran, none, k, m, k, n, DataType::TYPE_FP16);
    TransposeGemmOpTest(none, none, m, k, k, n, DataType::TYPE_FP32);
    TransposeGemmOpTest(none, tran, m, k, n, k, DataType::TYPE_FP32);
    TransposeGemmOpTest(tran, tran, k, m, n, k, DataType::TYPE_FP32);
    TransposeGemmOpTest(tran, none, k, m, k, n, DataType::TYPE_FP32);
}

TEST_F(ROCmGemmOpTest, BatchGemmOpTest) {
    BatchGemmOpTest(1, 8, 16, 32, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(2, 8, 16, 32, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(4, 8, 16, 32, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(8, 8, 8, 8, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(1, 8, 16, 32, DataType::TYPE_FP32);
    BatchGemmOpTest(2, 8, 16, 32, DataType::TYPE_FP32);
    BatchGemmOpTest(4, 8, 16, 32, DataType::TYPE_FP32);
    BatchGemmOpTest(8, 8, 8, 8, DataType::TYPE_FP32);
}

TEST_F(ROCmGemmOpTest, TransposeBatchGemmOpTest) {
    auto   tran = TransposeOperation::TRANSPOSE;
    auto   none = TransposeOperation::NONE;
    size_t b    = 128;
    size_t m    = 64;
    size_t n    = 8;
    size_t k    = 16;
    BatchTransposeGemmOp(none, none, b, m, k, k, n, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchTransposeGemmOp(none, tran, b, m, k, n, k, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchTransposeGemmOp(tran, tran, b, k, m, n, k, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchTransposeGemmOp(tran, none, b, k, m, k, n, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchTransposeGemmOp(none, none, b, m, k, k, n, DataType::TYPE_FP32);
    BatchTransposeGemmOp(none, tran, b, m, k, n, k, DataType::TYPE_FP32);
    BatchTransposeGemmOp(tran, tran, b, k, m, n, k, DataType::TYPE_FP32);
    BatchTransposeGemmOp(tran, none, b, k, m, k, n, DataType::TYPE_FP32);
}

TEST_F(ROCmGemmOpTest, TransposeBatchMixFloatGemmOP) {
    auto   tran = TransposeOperation::TRANSPOSE;
    auto   none = TransposeOperation::NONE;
    size_t b    = 128;
    size_t m    = 64;
    size_t n    = 8;
    size_t k    = 16;
    MixtureBatchTransposeGemmOp(none, none, b, m, k, k, n, DataType::TYPE_FP16, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(none, tran, b, m, k, n, k, DataType::TYPE_FP16, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(tran, tran, b, k, m, n, k, DataType::TYPE_FP16, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(tran, none, b, k, m, k, n, DataType::TYPE_FP16, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(none, none, b, m, k, k, n, DataType::TYPE_FP32, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(none, tran, b, m, k, n, k, DataType::TYPE_FP32, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(tran, tran, b, k, m, n, k, DataType::TYPE_FP32, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(tran, none, b, k, m, k, n, DataType::TYPE_FP32, DataType::TYPE_FP32);
}
#endif
