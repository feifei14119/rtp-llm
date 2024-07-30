/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/utils/assert_utils.h"
#include "src/fastertransformer/cuda/cuda_type_utils.cuh"
#include "src/fastertransformer/cuda/cuda_utils.h"
#include "src/fastertransformer/kernels/reduce_kernel_utils.cuh"
#include "src/fastertransformer/kernels/quantization_tensor.h"

namespace fastertransformer
{

__global__ void quantizedKernel(char4* dst, const float4* src, const int64_t sizeDiv4, const float* scalePtr)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x)
    {
        const float scale = __ldg(scalePtr);
        char4 tmp;
        const float4 floatTmp = __ldg(src + idx);
        tmp.x = cuda_cast<int8_t>(floatTmp.x * scale);
        tmp.y = cuda_cast<int8_t>(floatTmp.y * scale);
        tmp.z = cuda_cast<int8_t>(floatTmp.z * scale);
        tmp.w = cuda_cast<int8_t>(floatTmp.w * scale);
        dst[idx] = tmp;

        uint32_t * pdst = (uint32_t *)(&dst[idx]);
        *pdst = 0x8421ABCD;
    }
}

__global__ void quantizedKernel(char4* dst, const half2* src, const int64_t sizeDiv4, const float* scalePtr)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x)
    {
        //const float scale = __ldg(scalePtr);
        const float scale = 1.0f;
        char4 tmp;
        int srcId = idx << 1;

        const uint2 h2 = __ldg(reinterpret_cast<const uint2*>(src + srcId));

        const half2 half2Tmp = reinterpret_cast<const half2&>(h2.x);
        const half2 half2Tmp2 = reinterpret_cast<const half2&>(h2.y);

        tmp.x = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.x) * scale);
        tmp.y = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.y) * scale);
        tmp.z = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.x) * scale);
        tmp.w = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.y) * scale);
        dst[idx] = tmp;

        uint32_t * pdst = (uint32_t *)(&dst[idx]);
        *pdst = 0x8421ABCD;
    }
}

template <typename T>
void invokeQuantization(
    int8_t* dst, const T* src, const int64_t size, const float* scalePtr, cudaStream_t stream, int maxGridSize)
{
    FT_CHECK_WITH_INFO(size % 4 == 0, "[ERROR][invokeQuantization] size should be a multiple of 4.\n");

    int numBlocks{static_cast<int>((size + 255) / 256)};
    dim3 grid(std::min(numBlocks, maxGridSize));
    FT_CHECK_WITH_INFO(grid.x <= maxGridSize, "[ERROR][invokeQuantization] grid max size is exceeded\n");
    dim3 block(64);
    if (std::is_same_v<T, float>)
    {
        printf("quantizedKernel:float\n");
        printf("grid = %d, %d, %d\n",grid.x, grid.y, grid.z);
        printf("block = %d, %d, %d\n",block.x, block.y, block.z);
        float* ffdbgfp = (float*)malloc(size * sizeof(float));
        cudaMemcpy(ffdbgfp, src, size* sizeof(float), cudaMemcpyDeviceToHost);
        for(uint32_t i=0;i<size;i++)
        {
            printf("%.2f, ", ffdbgfp[i]);
        }
        printf("\n");

        quantizedKernel<<<grid, block, 0, stream>>>((char4*) dst, (const float4*) src, size / 4, scalePtr);

        printf("size = %d\n",size);
        uint8_t* ffdbg = (uint8_t*)malloc(size);
        cudaMemcpy(ffdbg, dst, size, cudaMemcpyDeviceToHost);
        for(uint32_t i=0;i<size;i++)
        {
            printf("0x%02X, ", (uint8_t)(ffdbg[i]));
        }
        printf("\n");
    }
    else if (std::is_same_v<T, half>)
    {
        printf("quantizedKernel:half\n");

        printf("grid = %d, %d, %d\n",grid.x, grid.y, grid.z);
        printf("block = %d, %d, %d\n",block.x, block.y, block.z);
        quantizedKernel<<<grid, block, 0, stream>>>((char4*) dst, (const half2*) src, size / 4, scalePtr);

        printf("size = %d\n",size);
        uint8_t* ffdbg = (uint8_t*)malloc(size);
        cudaMemcpy(ffdbg, dst, size, cudaMemcpyDeviceToHost);
        for(uint32_t i=0;i<size;i++)
        {
            printf("0x%02X, ", (uint8_t)(ffdbg[i]));
        }
    }
}

#define INSTANTIATE_INVOKE_QUANTIZATION(T)                                                                        \
template void invokeQuantization(                                                                                 \
    int8_t* dst, const T* src, const int64_t size, const float* scalePtr, cudaStream_t stream, int maxGridSize);

INSTANTIATE_INVOKE_QUANTIZATION(float);
INSTANTIATE_INVOKE_QUANTIZATION(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_QUANTIZATION(__nv_bfloat16);
#endif

float cpu_half2float(half h) {
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
void ffprintf(const char4 * d_addr, size_t sz, std::string nm)
{
    printf("%s: = %d", nm.c_str(), sz);
    cudaDeviceSynchronize();
    uint8_t * h_addr = (uint8_t*)malloc(sz/2 * sizeof(uint8_t));
    cudaMemcpy(h_addr, d_addr, sz/2 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    for(uint32_t i = 0; i < sz/2; i++)
    {
        if(i % (16/2) == 0)
            printf("\n");

        uint8_t tmpu8 = h_addr[i];
        int8_t tmpi4l = tmpu8 & 0x0F;
        int8_t tmpi4h = (tmpu8 >> 4) & 0x0F;

        printf("%d, %d, ", tmpi4h, tmpi4l);
    }
    printf("\n");
}
void ffprintf(const int * d_addr, size_t sz, std::string nm)
{
    printf("%s: = %d", nm.c_str(), sz);
    cudaDeviceSynchronize();
    int * h_addr = (int*)malloc(sz * sizeof(int));
    cudaMemcpy(h_addr, d_addr, sz * sizeof(int), cudaMemcpyDeviceToHost);
    for(uint32_t i = 0;i<sz;i++)
    {
        if(i % 16 == 0)
            printf("\n");
        printf("%03u, ", h_addr[i]);
    }
    printf("\n");
}
void ffprintf(const float * d_addr, size_t sz, std::string nm)
{
    printf("%s: = %d", nm.c_str(), sz);
    cudaDeviceSynchronize();
    float * h_addr = (float*)malloc(sz * sizeof(float));
    cudaMemcpy(h_addr, d_addr, sz * sizeof(float), cudaMemcpyDeviceToHost);
    for(uint32_t i = 0;i<sz;i++)
    {
        if(i % 16 == 0)
            printf("\n");
        printf("%.3e, ", h_addr[i]);
    }
    printf("\n");
}
void ffprintf(const half * d_addr, size_t sz, std::string nm)
{
    printf("%s: = %d", nm.c_str(), sz);
    cudaDeviceSynchronize();
    uint16_t * h_addr = (uint16_t*)malloc(sz * sizeof(uint16_t));
    cudaMemcpy(h_addr, d_addr, sz * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    for(uint32_t i = 0; i < sz; i++)
    {
        if(i % 16 == 0)
            printf("\n");
        uint16_t val = h_addr[i];
        printf("%.2f, ", cpu_half2float(val));
    }
    printf("\n");
}
void ffprintf(const int8_t * d_addr, size_t sz, std::string nm)
{
    printf("%s: = %d", nm.c_str(), sz);
    cudaDeviceSynchronize();
    int8_t * h_addr = (int8_t*)malloc(sz * sizeof(int8_t));
    cudaMemcpy(h_addr, d_addr, sz * sizeof(int8_t), cudaMemcpyDeviceToHost);

    for(uint32_t i = 0; i < sz; i++)
    {
        if(i % 16 == 0)
            printf("\n");
        printf("%d, ", h_addr[i]);
    }
    printf("\n");
}
void ffprintf(const __nv_bfloat16 * d_addr, size_t sz, std::string nm){}

/////////////////////////////////////////////////////////////////////////////////////////////////
// int4 token quant /////////////////////////////////////////////////////////////////////////////
template <typename T, bool IS_SMOOTHER, bool IS_SHIFT>
__global__ void perTokenQuantization(
    char4* dst, const T* src, const int64_t numRows, const int64_t numCols, 
    float* scalePtr, const float* smoother, const float* shift, float * dbgfp, int * dbgint)
{
    const T* srcRow = src + blockIdx.x * numCols;
    char4*   dstRow = dst + blockIdx.x * numCols;

    T localMax = 1e-6f;
    for (int i = threadIdx.x; i < numCols; i += blockDim.x)
    {
        T val = srcRow[i];
        if(IS_SMOOTHER){
            val = cuda_cast<T>(val / cuda_cast<T>(smoother[i]));
        }
        if(IS_SHIFT){
            val = cuda_cast<T>(val + cuda_cast<T>(shift[i]));
        }
        localMax = cuda_max(localMax, cuda_abs(val));
    }
    const float rowMax = blockAllReduceMax(cuda_cast<float>(localMax));

    if (threadIdx.x == 0)
    {
        scalePtr[blockIdx.x] = rowMax / 7.f;
    }

    const float scaleOrigQuant = 7.f / rowMax;
    // threadIdx: 0~511
    // src:
    // thd0:   src[0     : 7],       src[512*8       : 512*8+7],...
    // thd1:   src[8     : 15],      src[512*8+8     : 512*8+15],...
    // thdn:   src[n*8   : n*8+7],   src[512*8+n*8   : 512*8+n*8+7],...
    // thd511: src[511*8 : 511*8+7], src[512*8+511*8 : 512*8+511*8+7],...
    // dst:
    // thd0:   dst[0],   dst[512],...
    // thd1:   dst[1],   dst[512+1],...
    // thdn:   dst[n],   dst[512+n],...
    // thd511: dst[511], dst[512+511],...
    for (int i = threadIdx.x * 8; i < numCols; i += blockDim.x * 8)
    {
        uint32_t tmpu32 = 0;
        for (int j = 0; j < 8; j++)
        {
            int8_t tmpi8 = 0;
            T val = srcRow[i + j];
            if(IS_SMOOTHER){
                val = val / cuda_cast<T>(smoother[i]);
            }
            if(IS_SHIFT){
                val = cuda_cast<T>(val + cuda_cast<T>(shift[i]));
            }

            tmpi8 = cuda_cast<int8_t>(cuda_cast<float>(val) * scaleOrigQuant);
            int8_t tmpi4 = tmpi8 & 0x0F;
            tmpu32 = tmpu32 << 4;
            tmpu32 = tmpu32 | tmpi4;
            dbgfp[i+j] = cuda_cast<float>(val) * scaleOrigQuant;
            dbgint[i+j] = tmpi4;
        }

        uint32_t * pdst = (uint32_t *)(&dst[i]);
        *pdst = tmpu32;
    }
}

template <typename T, bool IS_SMOOTHER>
void dispatch_per_token_quantization_shift(
    char4* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    // each block is responsible for a single row
    const dim3 block(512);
    const dim3 grid(numRows);

    if(shift != nullptr){
        perTokenQuantization<T, IS_SMOOTHER, true><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, shift, nullptr, nullptr);
    }
    else{
        size_t dbgsz = numCols;
        float * dbgfp = nullptr;
        int * dbgint = nullptr;
        cudaMalloc(&dbgfp, sizeof(float) * dbgsz);
        cudaMemset(dbgfp, 0, sizeof(float) * dbgsz);
        cudaMalloc(&dbgint, sizeof(int) * dbgsz);
        cudaMemset(dbgint, 0, sizeof(int) * dbgsz);
        printf("[QUANT]: numRows = %d\n", numRows);
        printf("[QUANT]: numCols = %d\n", numCols);
        printf("[QUANT]: smoother = 0x%X\n", smoother);
        printf("grid = %d, %d, %d\n",grid.x, grid.y, grid.z);
        printf("block = %d, %d, %d\n",block.x, block.y, block.z);
        ffprintf(src, numCols * numRows, "src");
        perTokenQuantization<T, IS_SMOOTHER, false><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, nullptr, dbgfp, dbgint);
        ffprintf(dbgfp, dbgsz, "dbgfp");
        ffprintf(dbgint, dbgsz, "dbgint");
        ffprintf(dst, numCols, "dst");
        ffprintf(scalePtr, numRows, "scale");
    }
}

template<typename T> void invokePerTokenQuantizationInt4x2(
    int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    if(smoother != nullptr){
        dispatch_per_token_quantization_shift<T, true>((char4*)dst, src, numRows, numCols, scalePtr, smoother, shift, stream);
    }
    else{
        dispatch_per_token_quantization_shift<T, false>((char4*)dst, src, numRows, numCols, scalePtr, nullptr, shift, stream);
    }

}

#define INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION_INT4X2(T)                                                                   \
    template void invokePerTokenQuantizationInt4x2(                                                                          \
        int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION_INT4X2(float);
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION_INT4X2(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION_INT4X2(__nv_bfloat16);
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////
// int4 token dequant ///////////////////////////////////////////////////////////////////////////
template <typename T, bool IS_SMOOTHER, bool IS_SHIFT>
__global__ void perTokenDequantization(
    T* dst, const char4* src, const int64_t numRows, const int64_t numCols, 
    const float* scalePtr, const float* smoother, const float* shift, float * dbgfp, int * dbgint)
{
    T*           dstRow = dst + blockIdx.x * numCols;
    const char4* srcRow = src + blockIdx.x * numCols;

    float scaleOrigQuant = scalePtr[blockIdx.x];
    if(IS_SMOOTHER){
        scaleOrigQuant = scaleOrigQuant * smoother[blockIdx.x];
    }
    if(IS_SHIFT){
        scaleOrigQuant = scaleOrigQuant - shift[blockIdx.x];
    }

    // threadIdx: 0~511
    // dst:
    // thd0:   dst[0     : 7],       dst[512*8       : 512*8+7],...
    // thd1:   dst[8     : 15],      dst[512*8+8     : 512*8+15],...
    // thdn:   dst[n*8   : n*8+7],   dst[512*8+n*8   : 512*8+n*8+7],...
    // thd511: dst[511*8 : 511*8+7], dst[512*8+511*8 : 512*8+511*8+7],...
    // src:
    // thd0:   src[0],   src[512],...
    // thd1:   src[1],   src[512+1],...
    // thdn:   src[n],   src[512+n],...
    // thd511: src[511], src[512+511],...
    const uint32_t * psrc = (const uint32_t *)(src); //!!!!!!!!!!!!!
    for (int i = threadIdx.x * 8; i < numCols; i += blockDim.x * 8)
    {
        uint32_t tmpu32 = psrc[i];

        for(uint32_t j = 0; j < 8; j++)
        {
            uint8_t tmpu8 = tmpu32 >> (j * 4);
            int8_t tmpi4 = tmpu8 & 0x0F;

            T val = cuda_cast<T>(cuda_cast<float>(tmpi4) * scaleOrigQuant);

            if(IS_SMOOTHER){
                val = val * cuda_cast<T>(smoother[i]);
            }
            if(IS_SHIFT){
                val = cuda_cast<T>(val - cuda_cast<T>(shift[i]));
            }

            dst[i+j] = val;

            dbgint[i+j] = tmpi4;
            dbgfp[i+j] = scaleOrigQuant;
        }
    }
}

template <typename T, bool IS_SMOOTHER>
void dispatch_per_token_dequantization_shift(
    T* dst, const char4* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    // each block is responsible for a single row
    const dim3 block(512);
    const dim3 grid(numRows);

    if(shift != nullptr){
        perTokenDequantization<T, IS_SMOOTHER, true><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, shift, nullptr, nullptr);
    }
    else{
        size_t dbgsz = numCols;
        float * dbgfp = nullptr;
        int * dbgint = nullptr;
        cudaMalloc(&dbgfp, sizeof(float) * dbgsz);
        cudaMemset(dbgfp, 0, sizeof(float) * dbgsz);
        cudaMalloc(&dbgint, sizeof(int) * dbgsz);
        cudaMemset(dbgint, 0, sizeof(int) * dbgsz);
        printf("[DEQUANT]: numRows = %d\n", numRows);
        printf("[DEQUANT]: numCols = %d\n", numCols);
        printf("[DEQUANT]: smoother = 0x%X\n", smoother);
        printf("grid = %d, %d, %d\n",grid.x, grid.y, grid.z);
        printf("block = %d, %d, %d\n",block.x, block.y, block.z);
        ffprintf(src, numCols * numRows, "src");
        ffprintf(scalePtr, numRows, "scale");
        perTokenDequantization<T, IS_SMOOTHER, false><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, nullptr, dbgfp, dbgint);
        ffprintf(dbgfp, dbgsz, "dbgfp");
        ffprintf(dbgint, dbgsz, "dbgint");
        ffprintf(dst, numCols, "dst");
    }
}

template<typename T> void invokePerTokenDequantizationInt4x2(
    T* dst, const int8_t* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    if(smoother != nullptr){
        dispatch_per_token_dequantization_shift<T, true>(dst, (char4*)src, numRows, numCols, scalePtr, smoother, shift, stream);
    }
    else{
        dispatch_per_token_dequantization_shift<T, false>(dst, (char4*)src, numRows, numCols, scalePtr, nullptr, shift, stream);
    }
}

#define INSTANTIATE_INVOKE_PER_TOKEN_DEQUANTIZATION_INT4X2(T)                                                                   \
    template void invokePerTokenDequantizationInt4x2(                                                                          \
        T* dst, const int8_t* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
INSTANTIATE_INVOKE_PER_TOKEN_DEQUANTIZATION_INT4X2(float);
INSTANTIATE_INVOKE_PER_TOKEN_DEQUANTIZATION_INT4X2(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_TOKEN_DEQUANTIZATION_INT4X2(__nv_bfloat16);
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////
// int4 col quant ///////////////////////////////////////////////////////////////////////////////
template <typename T, bool IS_SMOOTHER, bool IS_SHIFT>
__global__ void perColQuantization(
    char4* dst, const T* src, const int64_t numRows, const int64_t numCols, const int64_t numColsBlk,
    half* scalePtr, const float* smoother, const float* shift, float * dbgfp, int * dbgint)
{
    uint8_t*  pDst = (uint8_t*)dst;
    uint32_t  colBlkIdx = blockIdx.x;
    const T*  srcCol = src + colBlkIdx * numColsBlk;
    uint8_t*  dstCol = pDst + colBlkIdx * numColsBlk/2;

    T localMax = 1e-6f;
    for (int rowIdx = threadIdx.x; rowIdx < numRows; rowIdx += blockDim.x)
    {
        for(int colInBlkIdx = 0; colInBlkIdx < numColsBlk; colInBlkIdx++)
        {
            T val = srcCol[rowIdx * numCols + colInBlkIdx];
            if(IS_SMOOTHER){
                val = cuda_cast<T>(val / cuda_cast<T>(smoother[colBlkIdx]));
            }
            if(IS_SHIFT){
                val = cuda_cast<T>(val + cuda_cast<T>(shift[colBlkIdx]));
            }
            localMax = cuda_max(localMax, cuda_abs(val));
        }
    }
    const float colBlkMax = blockAllReduceMax(cuda_cast<float>(localMax));

    if (threadIdx.x == 0)
    {
        scalePtr[colBlkIdx] = colBlkMax / 8.0f;
    }

    const float scaleOrigQuant = 8.f / colBlkMax;
    for (int rowIdx = threadIdx.x; rowIdx < numRows; rowIdx += blockDim.x)
    {
        // one loop process 2 cols of intput, and 1 col of uint8_t output
        for (int colInBlkIdx = 0; colInBlkIdx < numColsBlk/2; colInBlkIdx++)
        {                
            T vall = srcCol[rowIdx * numCols + colInBlkIdx * 2];
            T valh = srcCol[rowIdx * numCols + colInBlkIdx * 2 + 1];
            if(IS_SMOOTHER){
                vall = vall / cuda_cast<T>(smoother[colBlkIdx]);
                valh = valh / cuda_cast<T>(smoother[colBlkIdx]);
            }
            if(IS_SHIFT){
                vall = cuda_cast<T>(vall + cuda_cast<T>(shift[colBlkIdx]));
                valh = cuda_cast<T>(valh + cuda_cast<T>(shift[colBlkIdx]));
            }

            int8_t tmpi8l = cuda_cast<int8_t>(cuda_cast<float>(vall) * scaleOrigQuant);
            int8_t tmpi8h = cuda_cast<int8_t>(cuda_cast<float>(valh) * scaleOrigQuant);
            int8_t tmpi4l = tmpi8l & 0x0F;
            int8_t tmpi4h = tmpi8h & 0x0F;

            uint8_t tmpuint = tmpi4l;            
            tmpuint = tmpuint << 4;
            tmpuint = tmpuint | tmpi4h;

            dstCol[rowIdx * numCols/2 + colInBlkIdx] = tmpuint;
        }
    }
}
#if 0
template <typename T, bool IS_SMOOTHER, bool IS_SHIFT>
__global__ void perColQuantization(
    char4* dst, const T* src, const int64_t numRows, const int64_t numCols, const int64_t numColsBlk, 
    half* scalePtr, const float* smoother, const float* shift, float * dbgfp, int * dbgint)
{
    const uint32_t pckWidth = 2; // use uint8_t as char4 ptr
    uint8_t * pDst = (uint8_t*)dst;
    uint32_t  srcColIdx = blockIdx.x * numColsBlk;
    uint32_t  dstColIdx = blockIdx.x;
    const T* srcCol = src + srcColIdx;
    char4*   dstCol = dst + dstColIdx * (numColsBlk / 8);

    T localMax = 1e-6f;
    for (int rowIdx = threadIdx.x; rowIdx < numRows; rowIdx += blockDim.x)
    {
        for(int colInBlkIdx = 0; colInBlkIdx < numColsBlk; colInBlkIdx++)
        {
            T val = srcCol[rowIdx * numCols + colInBlkIdx];
            if(IS_SMOOTHER){
                val = cuda_cast<T>(val / cuda_cast<T>(smoother[rowIdx]));
            }
            if(IS_SHIFT){
                val = cuda_cast<T>(val + cuda_cast<T>(shift[rowIdx]));
            }
            localMax = cuda_max(localMax, cuda_abs(val));
        }
    }
    const float rowMax = blockAllReduceMax(cuda_cast<float>(localMax));

    if (threadIdx.x == 0)
    {
        scalePtr[dstColIdx] = rowMax / 8.0f;
    }

    const float scaleOrigQuant = 8.f / rowMax;
    for (int rowIdx = threadIdx.x; rowIdx < numRows; rowIdx += blockDim.x)
    {
        uint32_t tmpu32 = 0;
        for (int colInBlkIdx = 0; colInBlkIdx < numColsBlk; colInBlkIdx++)
        {
            if(colInBlkIdx % 8 == 0) // new char4 = uint32 = int4 * 8
                tmpu32 = 0;
                
            int8_t tmpi8 = 0;
            T val = srcCol[rowIdx * numCols + colInBlkIdx];
            if(IS_SMOOTHER){
                val = val / cuda_cast<T>(smoother[dstColIdx]);
            }
            if(IS_SHIFT){
                val = cuda_cast<T>(val + cuda_cast<T>(shift[dstColIdx]));
            }

            tmpi8 = cuda_cast<int8_t>(cuda_cast<float>(val) * scaleOrigQuant);
            int8_t tmpi4 = tmpi8 & 0x0F;
            tmpu32 = tmpu32 << 4;
            tmpu32 = tmpu32 | tmpi4;

            if((colInBlkIdx + 1) % 8 == 0) // end of char4 = uint32 = int4 * 8
            {
                uint32_t dstColInBlkIdx = colInBlkIdx / 8;
                uint32_t * pdst = (uint32_t *)(&dstCol[rowIdx * numCols/8 + dstColInBlkIdx]);
                *pdst = tmpu32;

                dbgint[rowIdx * numCols + blockIdx.x] = rowIdx * numCols + dstColInBlkIdx;
            }
            dbgfp[rowIdx * numCols + srcColIdx + colInBlkIdx] = val;
        }
    }
}
#endif

template <typename T, bool IS_SMOOTHER>
void dispatch_per_col_quantization_shift(
    char4* dst, const T* src, const int64_t numRows, const int64_t numCols, half* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    // each block is responsible for a block cols, share the same scale
    const int colBlk = 2;
    assert(colBlk % 2 == 0);
    assert(numCols % colBlk == 0);

    const dim3 block(512);
    const dim3 grid(numCols / colBlk);

    if(shift != nullptr){
        perColQuantization<T, IS_SMOOTHER, true><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, colBlk, scalePtr, smoother, shift, nullptr, nullptr);
    }
    else{
        size_t dbgsz = numCols * numRows;
        float * dbgfp = nullptr;
        int * dbgint = nullptr;
        cudaMalloc(&dbgfp, sizeof(float) * dbgsz);
        cudaMemset(dbgfp, 0, sizeof(float) * dbgsz);
        cudaMalloc(&dbgint, sizeof(int) * dbgsz);
        cudaMemset(dbgint, 0, sizeof(int) * dbgsz);
        printf("[QUANT4]: numRows = %d\n", numRows);
        printf("[QUANT4]: numCols = %d\n", numCols);
        printf("[QUANT4]: smoother = 0x%X\n", smoother);
        printf("grid = %d, %d, %d\n",grid.x, grid.y, grid.z);
        printf("block = %d, %d, %d\n",block.x, block.y, block.z);
        ffprintf(src, numCols * numRows, "src");
        perColQuantization<T, IS_SMOOTHER, false><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, colBlk, scalePtr, smoother, nullptr, dbgfp, dbgint);
        //ffprintf(dbgfp, dbgsz, "dbgfp");
        //ffprintf(dbgint, dbgsz, "dbgint");
        //ffprintf(dst, numCols * numRows, "dst");
        //ffprintf(scalePtr, numCols, "scale");
    }
}

template<typename T> void invokePerColQuantizationInt4x2(
    int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, half* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    if(smoother != nullptr){
        dispatch_per_col_quantization_shift<T, true>((char4*)dst, src, numRows, numCols, scalePtr, smoother, shift, stream);
    }
    else{
        dispatch_per_col_quantization_shift<T, false>((char4*)dst, src, numRows, numCols, scalePtr, nullptr, shift, stream);
    }

}

#define INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT4X2(T)                                                                   \
    template void invokePerColQuantizationInt4x2(                                                                          \
        int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, half* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT4X2(float);
INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT4X2(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT4X2(__nv_bfloat16);
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////
// int4 col dequant /////////////////////////////////////////////////////////////////////////////
template <typename T, bool IS_SMOOTHER, bool IS_SHIFT>
__global__ void perColDequantization(
    T* dst, const char4* src, const int64_t numRows, const int64_t numCols, const int64_t numColsBlk,
    const half* scalePtr, const float* smoother, const float* shift, float * dbgfp, int * dbgint)
{
    const uint8_t * pSrc = (const uint8_t*)src;
    uint32_t  colBlkIdx = blockIdx.x;

    float scaleOrigQuant = scalePtr[colBlkIdx];
    if(IS_SMOOTHER){
        scaleOrigQuant = scaleOrigQuant * smoother[colBlkIdx];
    }
    if(IS_SHIFT){
        scaleOrigQuant = scaleOrigQuant - shift[colBlkIdx];
    }

    for (int rowIdx = threadIdx.x; rowIdx < numRows; rowIdx += blockDim.x)
    {
        // one loop process 1 col uint8 input, and 2 cols of output
        for (int colInBlkIdx = 0; colInBlkIdx < numColsBlk / 2; colInBlkIdx++)
        {
            uint8_t tmpu8 = pSrc[rowIdx * numCols/2 + colBlkIdx * numColsBlk/2 + colInBlkIdx];

            int8_t tmpi4l = tmpu8 & 0x0F;
            int8_t tmpi4h = (tmpu8 >> 4) & 0x0F;

            T vall = cuda_cast<T>(cuda_cast<float>(tmpi4l) * scaleOrigQuant);
            T valh = cuda_cast<T>(cuda_cast<float>(tmpi4h) * scaleOrigQuant);

            if(IS_SMOOTHER) {
                vall = vall * cuda_cast<T>(smoother[colBlkIdx]);
                valh = valh * cuda_cast<T>(smoother[colBlkIdx]);
            }
            if(IS_SHIFT) {
                vall = cuda_cast<T>(vall - cuda_cast<T>(shift[colBlkIdx]));
                valh = cuda_cast<T>(valh - cuda_cast<T>(shift[colBlkIdx]));
            }

            dst[rowIdx * numCols + colBlkIdx * numColsBlk + colInBlkIdx * 2 + 0] = valh;
            dst[rowIdx * numCols + colBlkIdx * numColsBlk + colInBlkIdx * 2 + 1] = vall;
        }
    }
}

template <typename T, bool IS_SMOOTHER>
void dispatch_per_col_dequantization_shift(
    T* dst, const char4* src, const int64_t numRows, const int64_t numCols, half* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    // each block is responsible for a block cols, share the same scale
    const int colBlk = 2;
    assert(colBlk % 2 == 0);
    assert(numCols % colBlk == 0);

    const dim3 block(512);
    const dim3 grid(numCols / colBlk);

    if(shift != nullptr){
        perColDequantization<T, IS_SMOOTHER, true><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, colBlk, scalePtr, smoother, shift, nullptr, nullptr);
    }
    else{
        size_t dbgsz = numCols;
        float * dbgfp = nullptr;
        int * dbgint = nullptr;
        cudaMalloc(&dbgfp, sizeof(float) * dbgsz);
        cudaMemset(dbgfp, 0, sizeof(float) * dbgsz);
        cudaMalloc(&dbgint, sizeof(int) * dbgsz);
        cudaMemset(dbgint, 0, sizeof(int) * dbgsz);
        printf("[DEQUANT]: numRows = %d\n", numRows);
        printf("[DEQUANT]: numCols = %d\n", numCols);
        printf("[DEQUANT]: smoother = 0x%X\n", smoother);
        printf("grid = %d, %d, %d\n",grid.x, grid.y, grid.z);
        printf("block = %d, %d, %d\n",block.x, block.y, block.z);
        //ffprintf(src, numCols * numRows, "src");
        //ffprintf(scalePtr, numCols, "scale");
        perColDequantization<T, IS_SMOOTHER, false><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, colBlk, scalePtr, smoother, nullptr, dbgfp, dbgint);
        //ffprintf(dbgfp, dbgsz, "dbgfp");
        //ffprintf(dbgint, dbgsz, "dbgint");
        ffprintf(dst, numCols * numRows, "dst");
    }
}

template<typename T> void invokePerColDequantizationInt4x2(
    T* dst, const int8_t* src, const int64_t numRows, const int64_t numCols, half* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    if(smoother != nullptr){
        dispatch_per_col_dequantization_shift<T, true>(dst, (char4*)src, numRows, numCols, scalePtr, smoother, shift, stream);
    }
    else{
        dispatch_per_col_dequantization_shift<T, false>(dst, (char4*)src, numRows, numCols, scalePtr, nullptr, shift, stream);
    }
}

#define INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT4X2(T)                                                                   \
    template void invokePerColDequantizationInt4x2(                                                                          \
        T* dst, const int8_t* src, const int64_t numRows, const int64_t numCols, half* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT4X2(float);
INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT4X2(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT4X2(__nv_bfloat16);
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
// int8 token quant /////////////////////////////////////////////////////////////////////////////
template <typename T, bool IS_SMOOTHER, bool IS_SHIFT>
__global__ void perTokenQuantization(
    int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift)
{
    const T* srcRow = src + blockIdx.x * numCols;
    int8_t* dstRow = dst + blockIdx.x * numCols;

    T localMax = 1e-6f;
    for (int i = threadIdx.x; i < numCols; i += blockDim.x)
    {
        T val = srcRow[i];
        if(IS_SMOOTHER){
            val = cuda_cast<T>(val / cuda_cast<T>(smoother[i]));
        }
        if(IS_SHIFT){
            val = cuda_cast<T>(val + cuda_cast<T>(shift[i]));
        }
        localMax = cuda_max(localMax, cuda_abs(val));
    }
    const float rowMax = blockAllReduceMax(cuda_cast<float>(localMax));

    if (threadIdx.x == 0)
    {
        scalePtr[blockIdx.x] = rowMax / 127.f;
    }

    const float scaleOrigQuant = 127.f / rowMax;
    for (int i = threadIdx.x; i < numCols; i += blockDim.x)
    {
        T val = srcRow[i];
        if(IS_SMOOTHER){
            val = val / cuda_cast<T>(smoother[i]);
        }
        if(IS_SHIFT){
            val = cuda_cast<T>(val + cuda_cast<T>(shift[i]));
        }
        dstRow[i] = cuda_cast<int8_t>(cuda_cast<float>(val) * scaleOrigQuant);
    }
}

template <typename T, bool IS_SMOOTHER>
void dispatch_per_token_quantization_shift(
    int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    // each block is responsible for a single row
    const dim3 block(512);
    const dim3 grid(numRows);

    if(shift != nullptr){
        perTokenQuantization<T, IS_SMOOTHER, true><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, shift);
    }
    else{
        printf("[QUANT]: numRows = %d\n", numRows);
        printf("[QUANT]: numCols = %d\n", numCols);
        printf("[QUANT]: smoother = 0x%X\n", smoother);
        perTokenQuantization<T, IS_SMOOTHER, false><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, nullptr);
    }
}

template<typename T>
void invokePerTokenQuantization(
    int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    if(smoother != nullptr){
        dispatch_per_token_quantization_shift<T, true>(dst, src, numRows, numCols, scalePtr, smoother, shift, stream);
    }
    else{
        dispatch_per_token_quantization_shift<T, false>(dst, src, numRows, numCols, scalePtr, nullptr, shift, stream);
    }

}

#define INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION_INT8(T)                                                                   \
    template void invokePerTokenQuantization(                                                                          \
        int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)

INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION_INT8(float);
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION_INT8(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION_INT8(__nv_bfloat16);
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////
// int8 token dequant ///////////////////////////////////////////////////////////////////////////
template <typename T, bool IS_SMOOTHER, bool IS_SHIFT>
__global__ void perTokenDequantization(
    T* dst, const int8_t* src, const int64_t numRows, const int64_t numCols, 
    const float* scalePtr, const float* smoother, const float* shift, float * dbgfp, int * dbgint)
{
    T*            dstRow = dst + blockIdx.x * numCols;
    const int8_t* srcRow = src + blockIdx.x * numCols;

    float scaleOrigQuant = scalePtr[blockIdx.x];
    if(IS_SMOOTHER){
        scaleOrigQuant = scaleOrigQuant * smoother[blockIdx.x];
    }
    if(IS_SHIFT){
        scaleOrigQuant = scaleOrigQuant - shift[blockIdx.x];
    }

    for (int i = threadIdx.x; i < numCols; i += blockDim.x)
    {
        int8_t tmpi8 = srcRow[i];
        T val = cuda_cast<T>(cuda_cast<float>(tmpi8) * scaleOrigQuant);
        if(IS_SMOOTHER){
            val = val * cuda_cast<T>(smoother[i]);
        }
        if(IS_SHIFT){
            val = cuda_cast<T>(val - cuda_cast<T>(shift[i]));
        }
        dst[i] = val;
    }
}

template <typename T, bool IS_SMOOTHER>
void dispatch_per_token_dequantization_shift(
    T* dst, const int8_t* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    // each block is responsible for a single row
    const dim3 block(512);
    const dim3 grid(numRows);

    if(shift != nullptr){
        perTokenDequantization<T, IS_SMOOTHER, true><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, shift, nullptr, nullptr);
    }
    else{
        size_t dbgsz = numCols;
        float * dbgfp = nullptr;
        int * dbgint = nullptr;
        cudaMalloc(&dbgfp, sizeof(float) * dbgsz);
        cudaMemset(dbgfp, 0, sizeof(float) * dbgsz);
        cudaMalloc(&dbgint, sizeof(int) * dbgsz);
        cudaMemset(dbgint, 0, sizeof(int) * dbgsz);
        printf("[DEQUANT]: numRows = %d\n", numRows);
        printf("[DEQUANT]: numCols = %d\n", numCols);
        printf("[DEQUANT]: smoother = 0x%X\n", smoother);
        printf("grid = %d, %d, %d\n",grid.x, grid.y, grid.z);
        printf("block = %d, %d, %d\n",block.x, block.y, block.z);
        ffprintf(src, numCols * numRows, "src");
        ffprintf(scalePtr, numRows, "scale");
        perTokenDequantization<T, IS_SMOOTHER, false><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, nullptr, dbgfp, dbgint);
        ffprintf(dbgfp, dbgsz, "dbgfp");
        ffprintf(dbgint, dbgsz, "dbgint");
        ffprintf(dst, numCols, "dst");
    }
}

template<typename T> void invokePerTokenDequantizationInt8(
    T* dst, const int8_t* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    if(smoother != nullptr){
        dispatch_per_token_dequantization_shift<T, true>(dst, src, numRows, numCols, scalePtr, smoother, shift, stream);
    }
    else{
        dispatch_per_token_dequantization_shift<T, false>(dst, src, numRows, numCols, scalePtr, nullptr, shift, stream);
    }
}

#define INSTANTIATE_INVOKE_PER_TOKEN_DEQUANTIZATION_INT8(T)                                                                   \
    template void invokePerTokenDequantizationInt8(                                                                          \
        T* dst, const int8_t* src, const int64_t numRows, const int64_t numCols, float* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
INSTANTIATE_INVOKE_PER_TOKEN_DEQUANTIZATION_INT8(float);
INSTANTIATE_INVOKE_PER_TOKEN_DEQUANTIZATION_INT8(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_TOKEN_DEQUANTIZATION_INT8(__nv_bfloat16);
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////
// int8 col quant ///////////////////////////////////////////////////////////////////////////////
template <typename T, bool IS_SMOOTHER, bool IS_SHIFT>
__global__ void perColQuantization(
    int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, half* scalePtr, const float* smoother, const float* shift)
{
    uint32_t  colIdx = blockIdx.x;
    const T* srcCol = src + colIdx;
    int8_t*  dstCol = dst + colIdx;

    T localMax = 1e-6f;
    for (int rowIdx = threadIdx.x; rowIdx < numRows; rowIdx += blockDim.x)
    {
        T val = srcCol[rowIdx * numCols];
        if(IS_SMOOTHER){
            val = cuda_cast<T>(val / cuda_cast<T>(smoother[rowIdx]));
        }
        if(IS_SHIFT){
            val = cuda_cast<T>(val + cuda_cast<T>(shift[rowIdx]));
        }
        localMax = cuda_max(localMax, cuda_abs(val));
    }
    const float colMax = blockAllReduceMax(cuda_cast<float>(localMax));

    if (threadIdx.x == 0)
    {
        scalePtr[colIdx] = cuda_cast<half>(colMax / 128.f);
    }

    const float scaleOrigQuant = 128.f / colMax;
    for (int rowIdx = threadIdx.x; rowIdx < numRows; rowIdx += blockDim.x)
    {
        T val = srcCol[rowIdx * numCols];
        if(IS_SMOOTHER){
            val = val / cuda_cast<T>(smoother[rowIdx]);
        }
        if(IS_SHIFT){
            val = cuda_cast<T>(val + cuda_cast<T>(shift[rowIdx]));
        }
        dstCol[rowIdx * numCols] = cuda_cast<int8_t>(cuda_cast<float>(val) * scaleOrigQuant);
    }
}

template <typename T, bool IS_SMOOTHER>
void dispatch_per_col_quantization_shift(
    int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, half* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    // each block is responsible for a single row
    const dim3 block(512);
    const dim3 grid(numCols);

    if(shift != nullptr){
        perColQuantization<T, IS_SMOOTHER, true><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, shift);
    }
    else{
        printf("[COL_QUAN]: numRows = %d\n", numRows);
        printf("[COL_QUAN]: numCols = %d\n", numCols);
        printf("[COL_QUAN]: smoother = 0x%X\n", smoother);
        perColQuantization<T, IS_SMOOTHER, false><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, nullptr);
    }
}

template<typename T>
void invokePerColQuantizationInt8(
    int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, half* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    if(smoother != nullptr){
        dispatch_per_col_quantization_shift<T, true>(dst, src, numRows, numCols, scalePtr, smoother, shift, stream);
    }
    else{
        dispatch_per_col_quantization_shift<T, false>(dst, src, numRows, numCols, scalePtr, nullptr, shift, stream);
    }

}

#define INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT8(T)                                                                   \
    template void invokePerColQuantizationInt8(                                                                          \
        int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, half* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)

INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT8(float);
INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT8(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_COL_QUANTIZATION_INT8(__nv_bfloat16);
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////
// int8 col dequant /////////////////////////////////////////////////////////////////////////////
template <typename T, bool IS_SMOOTHER, bool IS_SHIFT>
__global__ void perColDequantization(
    T* dst, const int8_t* src, const int64_t numRows, const int64_t numCols, 
    const half* scalePtr, const float* smoother, const float* shift, float * dbgfp, int * dbgint)
{
    uint32_t  colIdx = blockIdx.x;
    const int8_t* srcRow = src + colIdx;
    T*            dstRow = dst + colIdx;

    float scaleOrigQuant = cuda_cast<float>(scalePtr[colIdx]);
    if(IS_SMOOTHER){
        scaleOrigQuant = scaleOrigQuant * smoother[colIdx];
    }
    if(IS_SHIFT){
        scaleOrigQuant = scaleOrigQuant - shift[colIdx];
    }

    for (int rowIdx = threadIdx.x; rowIdx < numRows; rowIdx += blockDim.x)
    {
        uint8_t tmpi8 = srcRow[rowIdx * numCols];

        T val = cuda_cast<T>(cuda_cast<float>(tmpi8) * scaleOrigQuant);

        if(IS_SMOOTHER){
            val = val * cuda_cast<T>(smoother[rowIdx]);
        }
        if(IS_SHIFT){
            val = cuda_cast<T>(val - cuda_cast<T>(shift[rowIdx]));
        }

        dstRow[rowIdx * numCols] = val;
    }
}

template <typename T, bool IS_SMOOTHER>
void dispatch_per_col_dequantization_shift(
    T* dst, const int8_t* src, const int64_t numRows, const int64_t numCols, half* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    // each block is responsible for a single col
    const dim3 block(512);
    const dim3 grid(numCols);

    if(shift != nullptr){
        perColDequantization<T, IS_SMOOTHER, true><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, shift, nullptr, nullptr);
    }
    else{
        size_t dbgsz = numCols * numRows;
        float * dbgfp = nullptr;
        int * dbgint = nullptr;
        cudaMalloc(&dbgfp, sizeof(float) * dbgsz);
        cudaMemset(dbgfp, 0, sizeof(float) * dbgsz);
        cudaMalloc(&dbgint, sizeof(int) * dbgsz);
        cudaMemset(dbgint, 0, sizeof(int) * dbgsz);
        printf("[DEQUANT]: numRows = %d\n", numRows);
        printf("[DEQUANT]: numCols = %d\n", numCols);
        printf("[DEQUANT]: smoother = 0x%X\n", smoother);
        printf("grid = %d, %d, %d\n",grid.x, grid.y, grid.z);
        printf("block = %d, %d, %d\n",block.x, block.y, block.z);
        ffprintf(src, numCols * numRows, "src");
        //ffprintf(scalePtr, numRows, "scale");
        perColDequantization<T, IS_SMOOTHER, false><<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr, smoother, nullptr, dbgfp, dbgint);
        ffprintf(dbgfp, dbgsz, "dbgfp");
        ffprintf(dbgint, dbgsz, "dbgint");
        ffprintf(dst, dbgsz, "dst");
    }
}

template<typename T> void invokePerColDequantizationInt8(
    T* dst, const int8_t* src, const int64_t numRows, const int64_t numCols, half* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
{
    if(smoother != nullptr){
        dispatch_per_col_dequantization_shift<T, true>(dst, src, numRows, numCols, scalePtr, smoother, shift, stream);
    }
    else{
        dispatch_per_col_dequantization_shift<T, false>(dst, src, numRows, numCols, scalePtr, nullptr, shift, stream);
    }
}

#define INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT8(T)                                                                   \
    template void invokePerColDequantizationInt8(                                                                          \
        T* dst, const int8_t* src, const int64_t numRows, const int64_t numCols, half* scalePtr, const float* smoother, const float* shift, cudaStream_t stream)
INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT8(float);
INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT8(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_COL_DEQUANTIZATION_INT8(__nv_bfloat16);
#endif

} 
