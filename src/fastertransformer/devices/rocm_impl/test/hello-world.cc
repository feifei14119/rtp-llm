#include "src/fastertransformer/devices/testing/TestBase.h"
#include "src/fastertransformer/devices/rocm_impl/ROCmDevice.h"

using namespace std;
using namespace fastertransformer;

class RocmOpsTest: public DeviceTestBase {
public:
};
#if 0
#include <hip/hip_runtime.h>
#include <iostream>



#define HIP_CHECK(error) \
    do { \
        hipError_t _err = (error); \
        if (_err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(_err) << " at line " << __LINE__ << std::endl; \
            exit(-1); \
        } \
    } while (0)


__global__ void gpu_print() {
    printf("Hello World from thread %d\n", threadIdx.x);
}


__global__ void add_array(int N, float* a, float* b) {

    int i = threadIdx.x + (blockIdx.x * blockDim.x);

    // Check for maximum workers count
    if (i<N) {
        b[i] += b[i]*a[i];
    }

}

TEST_F(RocmOpsTest, hellotest) {

    gpu_print<<<dim3(1), dim3(1), 0, 0>>>();

    int    N = 10000000;
    float* a = new float[N];
    float* b = new float[N];

    for (int i = 0; i < N; i++) {
        a[i] = b[i] = i;
    }

    float* Ga;
    float* Gb;
    size_t Nbytes = N * sizeof(float);
    HIP_CHECK(hipMalloc(&Ga, Nbytes));
    HIP_CHECK(hipMalloc(&Gb, Nbytes));

    HIP_CHECK(hipMemcpy(Ga, a, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Gb, b, Nbytes, hipMemcpyHostToDevice));

    dim3 threads(256, 1, 1);
    dim3 blocks(((N - 1) / 256) + 1, 1, 1);

    hipLaunchKernelGGL(add_array, blocks, threads, 0, 0, N, Ga, Gb);

    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(b, Gb, Nbytes, hipMemcpyDeviceToHost));

    delete[] a;
    delete[] b;
    HIP_CHECK(hipFree(Ga));
    HIP_CHECK(hipFree(Gb));

    std::cout << "Done." << std::endl;

}
#endif
