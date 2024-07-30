#include "src/fastertransformer/devices/base_tests/GemmOpTest.hpp"
#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"

using namespace std;
using namespace fastertransformer;

class CudaGemmOpTest: public GemmOpTest {

public:
    GemmOpTestInput ffPrepareGemmOpInput(size_t m, size_t n, size_t k, DataType type)
    {
        auto dtype = dataTypeToTorchType(type);
        auto A = torch::rand({(int)m, (int)k}, torch::Device(torch::kCPU)).to(dtype);
        auto B = torch::rand({(int)k, (int)n}, torch::Device(torch::kCPU)).to(dtype);
        //auto A = torch::full({(int)m, (int)k}, 1.0).to(dtype);
        //auto B = torch::full({(int)k, (int)n}, 1.0).to(dtype);
        
        printf("\n---------------------------------------\n");
        torch::print(A);
        printf("\n---------------------------------------\n");
        torch::print(B);

        return GemmOpTestInput({A, B});
    }
    GemmOpTestOutput ffQ8GemmOpRun(GemmOpTestInput& input)
    {
        CudaDevice*     cudaDev       = static_cast<CudaDevice*>(device_);
        auto A  = tensorToBuffer(input.A);
        auto B  = tensorToBuffer(input.B);
        auto Q8B = device_->quantize({*B, DataType::TYPE_QINT8, 1});
        auto Q4B = device_->quantize({*B, DataType::TYPE_QINT4X2, 1});
        auto D0  = device_->allocateBuffer({A->type(), {A->shape()[0], Q8B->shape()[1]}});
        auto D1  = device_->allocateBuffer({A->type(), {A->shape()[0], Q8B->shape()[1]}});
        auto D2  = device_->allocateBuffer({A->type(), {A->shape()[0], Q8B->shape()[1]}});
#if 0
        printf("\n***************************************************\n");
        GemmParams paramsq {*A, *Q8B, std::nullopt, D0};
        device_->gemm(paramsq);
        {
            QBuffer*  d_q8B   = static_cast<QBuffer*>(Q8B.get());
            Buffer    d_q8B_k = d_q8B->kernel();
            Buffer    d_q8B_s = d_q8B->scales();
            torch::Tensor t_q8B_k = bufferToTensor(d_q8B_k);
            printf("\n---------------------------------------\n");
            printf("d_q8B_k:\n %s\n", d_q8B_k.debugString().c_str());
            torch::print(t_q8B_k);
            printf("\n---------------------------------------\n");
            printf("d_q8B_s:\n %s\n", d_q8B_s.debugString().c_str());
            torch::Tensor t_q8B_s = bufferToTensor(d_q8B_s);
            torch::print(t_q8B_s);
            printf("\n---------------------------------------\n");
            torch::Tensor t_D = bufferToTensor(*D0);
            printf("D0:\n %s\n", D0.get()->debugString().c_str());
            torch::print(t_D);
        }
#endif
        printf("\n***************************************************\n");
        GemmParams paramsq4 {*A, *Q4B, std::nullopt, D1};
        device_->gemm(paramsq4);
        {
            QBuffer*  d_q4B   = static_cast<QBuffer*>(Q4B.get());
            Buffer    d_q4B_k = d_q4B->kernel();
            Buffer    d_q4B_s = d_q4B->scales();
            torch::Tensor t_q4B_k = bufferToTensor(d_q4B_k);
            printf("\n---------------------------------------\n");
            printf("d_q4B_k:\n %s\n", d_q4B_k.debugString().c_str());
            torch::print(t_q4B_k);
            printf("\n---------------------------------------\n");
            printf("d_q4B_s:\n %s\n", d_q4B_s.debugString().c_str());
            torch::Tensor t_q4B_s = bufferToTensor(d_q4B_s);
            torch::print(t_q4B_s);
            printf("\n---------------------------------------\n");
            torch::Tensor t_D = bufferToTensor(*D1);
            printf("D1:\n %s\n", D1.get()->debugString().c_str());
            torch::print(t_D);
        }

        GemmParams params0 {*A, *B, std::nullopt, D2};
        device_->gemm(params0);        
        {
            torch::Tensor t_B = bufferToTensor(*B);
            printf("t_B:\n %s\n", B.get()->debugString().c_str());
            torch::print(t_B);
            printf("\n---------------------------------------\n");
            printf("D1:\n %s\n", D1.get()->debugString().c_str());
            torch::Tensor t_D = bufferToTensor(*(D2.get()));
            torch::print(t_D);
        }
        return GemmOpTestOutput({bufferToTensor(*D1)});
    }
    void ffQ8GemmOpTest(size_t m, size_t n, size_t k, DataType dtype)
    {
        auto input = ffPrepareGemmOpInput(m, n, k, dtype);
        printf("\n***************************************************\n");
        auto result = ffQ8GemmOpRun(input);
        printf("\n***************************************************\n");
        auto result_ref = BasicGemmTorchRefRun(input);
        assertTensorClose(result.C.to(result_ref.C.type()), result_ref.C, 1e-2, 1e-2);
    }
};

TEST_F(CudaGemmOpTest, BasicGemmOpTest) {
    ffQ8GemmOpTest(4, 4, 4, DataType::TYPE_FP16); // need to do add_bias_and_interleave_quantized_tensor_inplace(1)
    //ffQ8GemmOpTest(1, 4, 8, DataType::TYPE_FP16);
    return;

    unsigned long   len           = 24;
    CudaDevice*     cudaDev       = static_cast<CudaDevice*>(device_);
    c10::ScalarType torchDataType = dataTypeToTorchType(DataType::TYPE_FP32);
    float scale = 0.09729f;
    float zero_point = 0;

    printf("\n***************************************************\n");
    printf("t_A:\n");
    //torch::Tensor t_A = torch::full({(long)1, (long)len}, 12.456).to(torchDataType);
    //torch::Tensor t_A = torch::rand({(long)4, (long)len}).to(torchDataType);    
    //torch::Tensor t_A = torch::full({(long)64, (long)64}, 12.456).to(torchDataType);
    torch::Tensor t_A = torch::rand({(long)64, (long)64}).to(torchDataType);
    torch::print(t_A);
    printf("\n---------------------------------------\n");
    BufferPtr h_A = tensorToBuffer(t_A, AllocationType::HOST);
    printf("h_A:\n %s\n", h_A.get()->debugString().c_str());
    printf("\n---------------------------------------\n");
    BufferPtr d_A = tensorToBuffer(t_A, AllocationType::DEVICE);
    printf("d_A:\n %s\n", d_A.get()->debugString().c_str());
    printf("\n---------------------------------------\n");
    torch::Tensor t_f16A = t_A.to(dataTypeToTorchType(DataType::TYPE_FP16));
    printf("t_f16A:\n");
    //torch::print(t_f16A);
    printf("\n---------------------------------------\n");
    BufferPtr h_fp16A = tensorToBuffer(t_f16A, AllocationType::HOST);
    printf("h_fp16A:\n %s\n", h_fp16A.get()->debugString().c_str());
    torch::Tensor t_fp16A_h = bufferToTensor(*(h_fp16A.get()));
    //torch::print(t_fp16A_h);
    printf("\n---------------------------------------\n");
    BufferPtr d_fp16A = tensorToBuffer(t_f16A, AllocationType::DEVICE);
    printf("d_fp16A:\n %s\n", d_fp16A.get()->debugString().c_str());
    torch::Tensor t_fp16A_d = bufferToTensor(*(d_fp16A.get()));
    //torch::print(t_fp16A_d);
#if 0    
    printf("\n================ TORCH =======================\n");
    printf("t_q8A:\n");
    torch::Tensor t_q8A = torch::quantize_per_tensor(t_A, scale, zero_point, at::ScalarType::QInt8);
    //torch::print(t_q8A);
    printf("\n---------------------------------------\n");
    printf("t_q4A:\n");
    //torch::Tensor t_q4A = torch::quantize_per_tensor(t_A, scale, zero_point, at::ScalarType::QUInt4x2);
    //torch::print(t_q4A);
    printf("\n---------------------------------------\n");
    printf("t_dq8A:\n");
    torch::Tensor t_dq8A = torch::dequantize(t_q8A);
    //torch::print(t_dq8A);
#endif    
#if 0
    printf("\n=============== CPU 8 ========================\n");
    BufferPtr h_q8A0  = cudaDev->quantize({*h_A, DataType::TYPE_QINT8, 1});
    QBuffer*  h_q8A   = static_cast<QBuffer*>(h_q8A0.get());
    Buffer    h_q8A_k = h_q8A->kernel();
    Buffer    h_q8A_s = h_q8A->scales();
    Buffer    h_q8A_z = h_q8A->zeros();
    printf("h_q8A:\n %s\n", h_q8A->debugString().c_str());
    printf("\n---------------------------------------\n");
    printf("h_q8A_k:\n %s\n", h_q8A_k.debugString().c_str());
    torch::Tensor th_q8A_k = bufferToTensor(h_q8A_k);
    //torch::print(th_q8A_k);
    printf("\n---------------------------------------\n");
    printf("h_q8A_s:\n %s\n", h_q8A_s.debugString().c_str());
    torch::Tensor th_q8A_s = bufferToTensor(h_q8A_s);
    //torch::print(th_q8A_s);
    printf("\n---------------------------------------\n");
    printf("h_q8A_z:\n %s\n", h_q8A_z.debugString().c_str());
#endif
#if 0
    printf("\n=============== CPU 4 ========================\n");
    BufferPtr h_q4A0  = cudaDev->quantize({*h_fp16A, DataType::TYPE_QINT4X2, 1});
    QBuffer*  h_q4A   = static_cast<QBuffer*>(h_q4A0.get());
    Buffer    h_q4A_k = h_q4A->kernel();
    Buffer    h_q4A_s = h_q4A->scales();
    Buffer    h_q4A_z = h_q4A->zeros();
    printf("h_q4A:\n %s\n", h_q4A->debugString().c_str());
    printf("\n---------------------------------------\n");
    printf("h_q4A_k:\n %s\n", h_q4A_k.debugString().c_str());
    char* ph_q4A_k = (char*)(h_q4A_k.data());
    torch::Tensor th_q4A_k = bufferToTensor(h_q4A_k);
    torch::print(th_q4A_k);
    printf("\n---------------------------------------\n");
    printf("h_q4A_s:\n %s\n", h_q4A_s.debugString().c_str());
    torch::Tensor th_q4A_s = bufferToTensor(h_q4A_s);
    torch::print(th_q4A_s);
    printf("\n---------------------------------------\n");
    printf("h_q4A_z:\n %s\n", h_q4A_z.debugString().c_str());
#endif
#if 0
    printf("\n=============== GPU 8 ========================\n");
    BufferPtr d_q8A0  = cudaDev->quantize({*d_fp16A, DataType::TYPE_QINT8, 1});
    QBuffer*  d_q8A   = static_cast<QBuffer*>(d_q8A0.get());
    Buffer    d_q8A_k = d_q8A->kernel();
    Buffer    d_q8A_s = d_q8A->scales();
    Buffer    d_q8A_z = d_q8A->zeros();
    printf("d_q8A:\n %s\n", d_q8A->debugString().c_str());
    printf("\n---------------------------------------\n");
    printf("d_q8A_k:\n %s\n", d_q8A_k.debugString().c_str());
    torch::Tensor td_q8A_k = bufferToTensor(d_q8A_k);
    //torch::print(td_q8A_k);
    assertTensorClose(td_q8A_k, th_q8A_k);
    printf("\n---------------------------------------\n");
    printf("d_q8A_s:\n %s\n", d_q8A_s.debugString().c_str());
    torch::Tensor td_q8A_s = bufferToTensor(d_q8A_s);
    //torch::print(td_q8A_s);
    printf("\n---------------------------------------\n");
    printf("d_q8A_z:\n %s\n", d_q8A_z.debugString().c_str());
#endif
#if 1
    printf("\n=============== GPU 4 ========================\n");
    BufferPtr d_q4A0  = cudaDev->quantize({*d_fp16A, DataType::TYPE_QINT4X2, 1});
    assertTensorClose(t_f16A, bufferToTensor(*(d_fp16A.get())), 1e-2, 1e-2);
    QBuffer*  d_q4A   = static_cast<QBuffer*>(d_q4A0.get());
    Buffer    d_q4A_k = d_q4A->kernel();
    Buffer    d_q4A_s = d_q4A->scales();
    Buffer    d_q4A_z = d_q4A->zeros();
    printf("d_q4A:\n %s\n", d_q4A->debugString().c_str());
    printf("\n---------------------------------------\n");
    {
        BufferPtr dp_q4A_k_t = cudaDev->allocateBuffer({d_q4A_k.type(), d_q4A_k.shape(), AllocationType::HOST});
        cudaDev->copy({*(dp_q4A_k_t.get()), d_q4A_k});
        //printf("d_q4A_k_t:\n %s\n", dp_q4A_k_t.get()->debugStringWithData<int8_t>().c_str());
    }
    printf("\n---------------------------------------\n");
    printf("d_q4A_s:\n %s\n", d_q4A_s.debugString().c_str());
    torch::Tensor td_q4A_s = bufferToTensor(d_q4A_s);
    //torch::print(td_q4A_s);
    printf("\n---------------------------------------\n");
    printf("d_q4A_z:\n %s\n", d_q4A_z.debugString().c_str());
#endif
    printf("\n***************************************************\n");
}

#if 0
TEST_F(CudaGemmOpTest, BasicGemmOpTest) {
    BasicGemmOpTest(2, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(8, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(1024, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(4096, 1024, 2048, DataType::TYPE_FP16);
    BasicGemmOpTest(2, 1024, 2048, DataType::TYPE_FP32);
    BasicGemmOpTest(8, 1024, 2048, DataType::TYPE_FP32);
    BasicGemmOpTest(1024, 1024, 2048, DataType::TYPE_FP32);
    BasicGemmOpTest(4096, 1024, 2048, DataType::TYPE_FP32);
    BasicQGemmOpTest(64, 64, 64, DataType::TYPE_FP16);
    BasicQGemmOpTest(2, 1024, 2048, DataType::TYPE_FP16);
    BasicQGemmOpTest(2, 2048, 4096, DataType::TYPE_FP16);
    // 结果正确，但int8 gemm跟float gemm之间的精度差较大，应改为int gemm对比
    qInt8QInt82DGemmOpTest(64, 64, 64);
    qInt8QInt82DGemmOpTest(2, 2048, 2048);
    qInt8QInt82DGemmOpTest(2, 4096, 4096);
}

TEST_F(CudaGemmOpTest, TransposeGemmOpTest) {
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

TEST_F(CudaGemmOpTest, BatchGemmOpTest) {
    BatchGemmOpTest(1, 8, 16, 32, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(2, 8, 16, 32, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(4, 8, 16, 32, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(8, 8, 8, 8, DataType::TYPE_FP16, 1e-2, 1e-2);
    BatchGemmOpTest(1, 8, 16, 32, DataType::TYPE_FP32);
    BatchGemmOpTest(2, 8, 16, 32, DataType::TYPE_FP32);
    BatchGemmOpTest(4, 8, 16, 32, DataType::TYPE_FP32);
    BatchGemmOpTest(8, 8, 8, 8, DataType::TYPE_FP32);
}

TEST_F(CudaGemmOpTest, TransposeBatchGemmOpTest) {
    auto tran = TransposeOperation::TRANSPOSE;
    auto none = TransposeOperation::NONE;
    size_t b = 128;
    size_t m = 64;
    size_t n = 8;
    size_t k = 16;
    BatchTransposeGemmOp(none, none, b, m, k, k, n, DataType::TYPE_FP16);
    BatchTransposeGemmOp(none, tran, b, m, k, n, k, DataType::TYPE_FP16);
    BatchTransposeGemmOp(tran, tran, b, k, m, n, k, DataType::TYPE_FP16);
    BatchTransposeGemmOp(tran, none, b, k, m, k, n, DataType::TYPE_FP16);
    BatchTransposeGemmOp(none, none, b, m, k, k, n, DataType::TYPE_FP32);
    BatchTransposeGemmOp(none, tran, b, m, k, n, k, DataType::TYPE_FP32);
    BatchTransposeGemmOp(tran, tran, b, k, m, n, k, DataType::TYPE_FP32);
    BatchTransposeGemmOp(tran, none, b, k, m, k, n, DataType::TYPE_FP32);
}

TEST_F(CudaGemmOpTest, TransposeBatchMixFloatGemmOP) {
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
#endif
