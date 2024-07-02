#include "kernelLauncher.h"

namespace fastertransformer {
namespace rocm {
namespace weight {

template <WeightOnlyQuantType QType, typename Arch, typename WeightOnlyFlag, template <typename T> class ActOp, bool Zero, bool Bias,
    int BATCH, int BLOCK_SIZE>
struct WeightOnlyBatchedGemvKernelLauncher
{
    static void run(const WeightOnlyParams& params, hipStream_t stream);
};

#if 0
template <WeightOnlyQuantType QType, typename Arch, typename WeightOnlyFlag, template <typename T> class ActOp, bool Zero, bool Bias,
    int BATCH, int BLOCK_SIZE>
struct WeightOnlyBatchedGemvKernelSm70Launcher
{
    static void run(const WeightOnlyParams& params, hipStream_t stream);
};

#define SELECT_VOLTA_BLOCKSIZE(BLOCK_SIZE)                                                                                  \
    WeightOnlyBatchedGemvKernelSm70Launcher<QType,                                                                     \
                                            cutlass::arch::Sm70,                                                       \
                                            WeightOnlyFlag,                                                            \
                                            ActOp,                                                                     \
                                            false,                                                                     \
                                            false,                                                                     \
                                            BATCH,                                                                     \
                                            BLOCK_SIZE>::run(params, stream)
#endif                                            

template<WeightOnlyQuantType QType, typename WeightOnlyFlag, template<typename T> class ActOp, int BATCH>
void select_arch(const WeightOnlyParams& params, hipStream_t stream)
{
    if (params.sm == 70) {
        if (params.k <= 20480) {
            if (params.n >= 64) {
                if (params.n <= 5120 && (params.k % (256 * 2) == 0)) {
                    //SELECT_VOLTA_BLOCKSIZE(256);
                }
                else if (params.n <= 11264 && (params.k % (128 * 2) == 0)) {
                    //SELECT_VOLTA_BLOCKSIZE(128);
                }
                else if (params.n <= 25600 && (params.k % (64 * 2) == 0)) {
                    //SELECT_VOLTA_BLOCKSIZE(64);
                }
                else {
                    throw std::runtime_error("For N > 25600, fpa_intb_gemm may be better choice");
                }
            }
            else {
                throw std::runtime_error("N < 64 is not supported by weight_only_batched_gemv on V100");
            }
        }
        else {
            throw std::runtime_error("For K > 20480, fpa_intb_gemm may be better choice");
        }
    }
    else {
        WeightOnlyBatchedGemvKernelLauncher<QType,
                                            int,// cutlass::arch::Sm75,
                                            WeightOnlyFlag,
                                            ActOp,
                                            false,
                                            false,
                                            BATCH,
                                            256>::run(params, stream);
    }
}

void weight_only_batched_gemv_launcher(const WeightOnlyParams& params, hipStream_t stream)
{
    assert(params.act_func_type == WeightOnlyActivationFunctionType::Identity);
    assert(params.weight_only_type == WeightOnlyType::PerChannel);
    assert(params.quant_type== WeightOnlyQuantType::Int8b);
    if (params.weight_only_type == WeightOnlyType::PerChannel && params.quant_type == WeightOnlyQuantType::Int8b) {
        switch (params.m) {
            case 1: {
                select_arch<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, IdentityActivation, 1>(params, stream);
                break;
            }
            case 2: {
                select_arch<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, IdentityActivation, 2>(params, stream);
                break;
            }
            case 3: {
                select_arch<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, IdentityActivation, 3>(params, stream);
                break;
            }
            case 4: {
                select_arch<WeightOnlyQuantType::Int8b, WeightOnlyPerChannel, IdentityActivation, 4>(params, stream);
                break;
            }
            default: {
                throw std::runtime_error("Weight only cuda kernel only supported bs <= 4");
                break;
            }
        }
    }
}


} // namespace weight
} // namespace rocm
} // namespace fastertransformer

