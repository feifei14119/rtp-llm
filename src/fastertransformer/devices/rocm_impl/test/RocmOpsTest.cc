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
    vector<float> expected = {12, 223, 334, 4, 5, 6};
    auto A = createHostBuffer({2, 3}, expected.data());
    auto B = device_->allocateBuffer({DataType::TYPE_FP32, {2, 3}, AllocationType::DEVICE}, {});
    auto C = device_->allocateBuffer({DataType::TYPE_FP32, {2, 3}, AllocationType::HOST}, {});
    device_->copy({*B, *A});
    device_->copy({*C, *B});

    assertBufferValueEqual(*C, expected);
}

