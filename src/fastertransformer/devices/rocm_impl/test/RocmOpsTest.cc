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

TEST_F(RocmOpsTest, testSelect) {
    auto src = createBuffer<float>({6, 5}, {
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24,
        25, 26, 27, 28, 29
    });
    auto index = createBuffer<int32_t>({3}, {0, 2, 3});

    auto result = device_->select({*src, *index});
    auto expected = torch::tensor({
        {0, 1, 2, 3, 4},
        {10, 11, 12, 13, 14},
        {15, 16, 17, 18, 19}
    }, torch::kFloat32);
    assertTensorClose(bufferToTensor(*result), expected, 1e-6, 1e-6);

    auto src2 = device_->clone({*src, AllocationType::HOST});
    auto index2 = device_->clone({*index, AllocationType::HOST});
    auto result2 = device_->select({*src2, *index2});
    assertTensorClose(bufferToTensor(*result2), expected, 1e-6, 1e-6);
}

TEST_F(RocmOpsTest, testSelect1d) {
    auto src = createBuffer<float>({2, 6}, {
        0, 1, 2, 3, 4, 5,
        10, 11, 12, 13, 14, 15
    });
    auto index = createBuffer<int32_t>({3}, {0, 4, 5}, AllocationType::HOST);

    auto result = device_->select({*src, *index, 1});
    auto expected = torch::tensor({
        {0, 4, 5},
        {10, 14, 15}
    }, torch::kFloat32);
    assertTensorClose(bufferToTensor(*result), expected, 1e-6, 1e-6);

    src = createBuffer<float>({2, 5, 3},{
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9, 10, 11,
        12, 13, 14,
        15, 16, 17,
        18, 19, 20,
        21, 22, 23,
        24, 25, 26,
        27, 28, 29
    });
    index = createBuffer<int32_t>({4}, {0, 1, 3, 4}, AllocationType::HOST);
    result = device_->select({*src, *index, 1});
    expected = torch::tensor({
        {0, 1, 2},
        {3, 4, 5},
        {9, 10, 11},
        {12, 13, 14},
        {15, 16, 17},
        {18, 19, 20},
        {24, 25, 26},
        {27, 28, 29}
    }, torch::kFloat32);
}