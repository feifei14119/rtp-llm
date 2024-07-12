#pragma once

#include <cstdint>
#include "amd_bfloat16.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel_rocm.h>

#define __nv_bfloat16 amd_bfloat16
#define __nv_bfloat162 amd_bfloat162
#define __nv_bfloat162 amd_bfloat162

static inline __device__ __host__ __nv_bfloat162 __float2bfloat162_rn(float x) {
    return {__nv_bfloat16(x), __nv_bfloat16(x)};
}
static inline __device__ __host__ __nv_bfloat162 __floats2bfloat162_rn(float x, float y) {
    return {__nv_bfloat16(x), __nv_bfloat16(y)};
}
static inline __device__ __host__ __nv_bfloat162 __ldg(const __nv_bfloat162* ptr) {
    return *ptr;
}
static inline __device__ __host__ __nv_bfloat16 __ldg(const __nv_bfloat16* ptr) {
    return *ptr;
}

template<typename T>
__device__ inline T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = 32) {
    (void)mask;
    return __shfl_xor(var, laneMask, width);
}

template<typename T>
__device__ inline T __shfl_sync(unsigned mask, T var, int laneMask, int width = 32) {
    (void)mask;
    return __shfl(var, laneMask, width);
}

template<typename T_OUT, typename T_IN> __host__ __device__ inline T_OUT special_cast(T_IN val) { return val; }
#ifdef ENABLE_BF16
template<> __host__ __device__ inline amd_bfloat16 special_cast<amd_bfloat16, float>(float val) { return __float2bfloat16(val); };
template<> __host__ __device__ inline float special_cast<float, amd_bfloat16>(amd_bfloat16 val) { return __bfloat162float(val); };
#endif

#define check_cuda_error check_hip_error
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaGetDevice hipGetDevice
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMaxSharedMemoryPerMultiprocessor hipDeviceAttributeMaxSharedMemoryPerMultiprocessor
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaMemcpy hipMemcpy
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define sync_check_cuda_error() rocm::syncAndCheck(__FILE__, __LINE__)
#define curandState_t hiprandState_t
#define cudaDeviceProp hipDeviceProp_t

    // Taken from cuda_utils.h