#pragma once

#include "hip_utils.h"
#include "src/fastertransformer/core/allocator.h"
#include "src/fastertransformer/utils/quantization.h"

namespace fastertransformer {
namespace rocm {

enum class WeightTypeId
{
    INT8 = 1,
    INT4 = 2,
};
enum class WeightDataType : int32_t
{
    kFLOAT = 0,
    kHALF = 1,
    kINT8 = 2,
    kINT32 = 3,
    kBOOL = 4,
    kUINT8 = 5,
    kFP8 = 6,
    kBF16 = 7,
    kINT64 = 8,
};

constexpr int32_t INT8_BITS = 8;
constexpr int32_t INT4_BITS = 4;
constexpr int32_t INT8_INT4_RATIO = INT8_BITS / INT4_BITS;

inline int32_t getWeightTypeMultiplier(WeightTypeId weightTypeId)
{
    return weightTypeId == WeightTypeId::INT8 ? 1 : INT8_INT4_RATIO;
}

// TODO: 1 step: only implement fastertransformer::kernels::weight_only_batched_gemv_launcher, 
//               NOT implement m_weightOnlyGemmRunner
//       2 step: m_weightOnlyGemmRunner

// using WeightOnlyGemmRunner = tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
// using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;

class WeightOnlyQuantMatmulPlugin
{
public:
    WeightOnlyQuantMatmulPlugin() = default;
    WeightOnlyQuantMatmulPlugin(WeightDataType type, WeightTypeId weightTypeId);
    ~WeightOnlyQuantMatmulPlugin() = default;

    int  initialize() noexcept;
    void init(WeightDataType type, WeightTypeId weightTypeId);
    size_t getWorkspaceSize(const int m, const int n, const int k) noexcept;
    int    enqueue(const void*  inputs,
                   const void*  weights,
                   const void*  scales,
                   void*        outputs,
                   void*        workspace,
                   const int    m,
                   const int    n,
                   const int    k,
                   hipStream_t stream) noexcept;

private:
    WeightDataType mType;
    WeightTypeId mWeightTypeId;
    // WeightOnlyGemmRunnerPtr m_weightOnlyGemmRunner; TODO: step2
    bool mHipKernelEnabled;
    size_t m_workspaceMaxSize;

    static constexpr int SMALL_M_FAST_PATH = 4;

#if 0

private:
    void configGemm();

private:
    WeightOnlyGemmRunnerPtr m_weightOnlyGemmRunner;

#endif    
};




}  // namespace rocm
}  // namespace fastertransformer


#if 0
#include "src/fastertransformer/utils/quantization.h"
#include "src/fastertransformer/cutlass/interface.h"
#include "src/fastertransformer/cuda/trt_utils.h"

#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <vector>

// The blank line here is to avoid clang-format -sort-includes option reordering these two cutlass header files and
// breaking dependencies
#include "cutlass/numeric_types.h"

namespace tensorrt_llm::plugins
{
enum class WeightTypeId
{
    INT8 = 1,
    INT4 = 2,
};

constexpr int32_t INT8_BITS = 8;
constexpr int32_t INT4_BITS = 4;
constexpr int32_t INT8_INT4_RATIO = INT8_BITS / INT4_BITS;

inline int32_t getWeightTypeMultiplier(WeightTypeId weightTypeId)
{
    return weightTypeId == WeightTypeId::INT8 ? 1 : INT8_INT4_RATIO;
}

using WeightOnlyGemmRunner = tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;

class WeightOnlyQuantMatmulPlugin
{
public:
    // using PluginProfilerPtr = std::shared_ptr<WeightOnlyQuantGemmPluginProfiler>;
    WeightOnlyQuantMatmulPlugin() = default;

    WeightOnlyQuantMatmulPlugin(nvinfer1::DataType type, WeightTypeId weightTypeId);

    ~WeightOnlyQuantMatmulPlugin() = default;

    size_t getWorkspaceSize(const int m, const int n, const int k) noexcept;
    int    enqueue(const void*  inputs,
                   const void*  weights,
                   const void*  scales,
                   void*        outputs,
                   void*        workspace,
                   const int    m,
                   const int    n,
                   const int    k,
                   cudaStream_t stream) noexcept;

    int  initialize() noexcept;

    void init(nvinfer1::DataType type, WeightTypeId weightTypeId);

private:
    void configGemm();

private:
    WeightOnlyGemmRunnerPtr m_weightOnlyGemmRunner;
    size_t m_workspaceMaxSize;
    nvinfer1::DataType mType;
    WeightTypeId mWeightTypeId;
    bool mCudaKernelEnabled;

    static constexpr int SMALL_M_FAST_PATH = 4;
};

} // namespace tensorrt_llm::plugins
#endif
