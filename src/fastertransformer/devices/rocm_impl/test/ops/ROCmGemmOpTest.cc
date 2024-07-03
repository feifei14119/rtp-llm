#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/devices/base_tests/GemmOpTest.hpp"
#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"

using namespace std;
using namespace fastertransformer;

class ROCmGemmOpTest: public GemmOpTest {

public:
    GemmOpTestOutput BasicROCMQGemmOpRun(GemmOpTestInput& input) {
        auto A = tensorToBuffer(input.A);
        auto host_B = tensorToBuffer(input.B, AllocationType::HOST);
        auto QB = device_->quantize({*host_B,
                                    std::nullopt,
                                    std::nullopt,
                                    DataType::TYPE_QINT8,
                                    1});
        auto device_B  = device_->clone({*QB});
        auto D = device_->allocateBuffer({A->type(), {A->shape()[0], device_B->shape()[1]}});
        GemmParams params {*A, *device_B, std::nullopt, D};
        device_->gemm(params);
        return GemmOpTestOutput({bufferToTensor(*D)});
    }

    void BasicQGemmOpTest(size_t m,
                        size_t n,
                        size_t k,
                        DataType dtype)
    {
        auto input = PrepareGemmOpInput(m, n, k, dtype);
        auto result = BasicROCMQGemmOpRun(input);
        auto result_ref = BasicGemmTorchRefRun(input);
        assertTensorClose(result.C.to(result_ref.C.type()), result_ref.C, 1e-2, 1e-2);

    }
};

TEST_F(ROCmGemmOpTest, BasicGemmOpTest) {
    BasicGemmOpTest(2, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(8, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(1024, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(4096, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(2, 1024, 2048, DataType::TYPE_FP32);
    BasicGemmOpTest(8, 1024, 2048, DataType::TYPE_FP32);
    BasicGemmOpTest(1024, 1024, 2048, DataType::TYPE_FP32);
    BasicGemmOpTest(4096, 1024, 2048, DataType::TYPE_FP32);
    // BasicQGemmOpTest(64, 64, 64, DataType::TYPE_FP16);
    // BasicQGemmOpTest(2, 1024, 2048, DataType::TYPE_FP16);
    // BasicQGemmOpTest(2, 2048, 4096, DataType::TYPE_FP16);
}

TEST_F(ROCmGemmOpTest, TransposeGemmOpTest) {
    auto tran = TransposeOperation::TRANSPOSE;
    auto none = TransposeOperation::NONE;
    size_t m = 5;
    size_t n = 1024;
    size_t k = 4096;
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
    auto tran = TransposeOperation::TRANSPOSE;
    auto none = TransposeOperation::NONE;
    size_t b = 128;
    size_t m = 64;
    size_t n = 8;
    size_t k = 16;
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
    auto tran = TransposeOperation::TRANSPOSE;
    auto none = TransposeOperation::NONE;
    size_t b = 128;
    size_t m = 64;
    size_t n = 8;
    size_t k = 16;
    MixtureBatchTransposeGemmOp(none, none, b, m, k, k, n, DataType::TYPE_FP16, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(none, tran, b, m, k, n, k, DataType::TYPE_FP16, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(tran, tran, b, k, m, n, k, DataType::TYPE_FP16, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(tran, none, b, k, m, k, n, DataType::TYPE_FP16, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(none, none, b, m, k, k, n, DataType::TYPE_FP32, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(none, tran, b, m, k, n, k, DataType::TYPE_FP32, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(tran, tran, b, k, m, n, k, DataType::TYPE_FP32, DataType::TYPE_FP32);
    MixtureBatchTransposeGemmOp(tran, none, b, k, m, k, n, DataType::TYPE_FP32, DataType::TYPE_FP32);
}
