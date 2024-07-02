#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"

using namespace std;
using namespace fastertransformer;

class RocmOpsTest: public DeviceTestBase {};

TEST_F(RocmOpsTest, hello) {
    GTEST_LOG_(INFO) << "\nhello world from RocmOpsTest\n";

    auto devProp = device_->getDeviceProperties();
    GTEST_LOG_(INFO) << "\ndevice type = " << std::to_string((int)devProp.type);
    GTEST_LOG_(INFO) << "\ndevice id   = " << std::to_string((int)devProp.id);
}

TEST_F(RocmOpsTest, testCopy) {
    GTEST_LOG_(INFO) << "\n**************** memcpy test *******************\n";

    vector<float> expected = {12, 223, 334, 4, 5, 6};
    auto A = createHostBuffer({2, 3}, expected.data());
    auto B = device_->allocateBuffer({DataType::TYPE_FP32, {2, 3}, AllocationType::DEVICE}, {});
    auto C = device_->allocateBuffer({DataType::TYPE_FP32, {2, 3}, AllocationType::HOST}, {});
    device_->copy({*B, *A});
    device_->copy({*C, *B});

    assertBufferValueEqual(*C, expected);
}

TEST_F(RocmOpsTest, TestOp) {
    GTEST_LOG_(INFO) << "\n**************** TestOp(vector add) *******************\n";

    int         len           = 64;
    ROCmDevice* RocmDev       = static_cast<ROCmDevice*>(device_);
    auto        torchDataType = dataTypeToTorchType(DataType::TYPE_FP32);

    torch::Tensor tensorA = torch::rand({1, len}, torch::Device(torch::kCPU)).to(torchDataType);
    torch::Tensor tensorB = torch::rand({1, len}, torch::Device(torch::kCPU)).to(torchDataType);
    torch::print(tensorA);
    torch::print(tensorB);
    torch::Tensor tensorC_ref = torch::add(tensorA, tensorB);

    BufferPtr d_bufferA = tensorToBuffer(tensorA);
    BufferPtr d_bufferB = tensorToBuffer(tensorB);
    BufferPtr d_bufferC = RocmDev->testVecAdd(d_bufferA, d_bufferB);
    std::cout << d_bufferC.get()->debugStringWithData<float>() << std::endl;

    torch::Tensor tensorC_rslt = bufferToTensor(*d_bufferC);
    assertTensorClose(tensorC_rslt, tensorC_ref);
}
