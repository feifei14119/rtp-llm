#pragma once

#include "common.h"
#include "utility.h"

namespace fastertransformer {
namespace rocm {
namespace weight {

void weight_only_batched_gemv_launcher(const WeightOnlyParams& params, hipStream_t stream);

// for compile, impl in enable.cc
extern bool isWeightOnlyBatchedGemvEnabled(WeightOnlyQuantType qtype);

} // namespace weight
} // namespace rocm
} // namespace fastertransformer

