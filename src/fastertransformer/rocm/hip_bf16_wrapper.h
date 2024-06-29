#pragma once

namespace fastertransformer {
template<typename T_OUT, typename T_IN> __host__ __device__ inline T_OUT special_cast(T_IN val) { return val; }
}

#if ENABLE_BF16
//#include <cuda_bf16.h>
//#include "hip/hip_bf16.h"
#include <hip/hip_common.h>
#include "amd_hip_bf16_new.h"
#ifdef ENABLE_FP8
#include "src/fastertransformer/utils/cuda_fp8_utils.h"
#endif

namespace fastertransformer {

template<> __host__ __device__ inline __hip_bfloat16 special_cast<__hip_bfloat16, float>(float val) { return __float2bfloat16(val); };
template<> __host__ __device__ inline float special_cast<float, __hip_bfloat16>(__hip_bfloat16 val) { return __bfloat162float(val); };
#ifdef ENABLE_FP8
template<> __host__ __device__ inline float special_cast<float, hipblaslt_f8>(hipblaslt_f8 val) { return (float)val; };
template<> __host__ __device__ inline hipblaslt_f8 special_cast<hipblaslt_f8, float>(float val) { return hipblaslt_f8(val); };
#endif

//template<typename T_OUT, typename T_IN> __host__ inline T_OUT special_cast(T_IN val) { return val; }
//template<> __host__ inline __hip_bfloat16 special_cast<__hip_bfloat16, float>(float val) { return __float2bfloat16(val); };
//template<> __host__ inline float special_cast<float, __hip_bfloat16>(__hip_bfloat16 val) { return __bfloat162float(val); };


__device__ __hip_bfloat16 operator-(const __hip_bfloat16 &h) { return __hneg(h); }
__device__  __hip_bfloat16 operator*(const __hip_bfloat16 &lh, const __hip_bfloat16 &rh) { return __hmul(lh, rh); }
__device__  __hip_bfloat16 operator+(const __hip_bfloat16 &lh, const __hip_bfloat16 &rh) { return __hadd(lh, rh); }
__device__  __hip_bfloat162 operator*(const __hip_bfloat162 &lh, const __hip_bfloat162 &rh) { return __hmul2(lh, rh); }

}


#endif
