#pragma once

#include "src/fastertransformer/rocm/hip_utils.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace fastertransformer {
namespace rocm {
namespace weight {

enum class WeightOnlyQuantType
{
    Int4b,
    Int8b
};
enum class WeightOnlyType
{
    PerChannel,
    GroupWise
};

enum class WeightOnlyActivationFunctionType
{
    Gelu,
    Relu,
    Identity,
    InvalidType
};

enum class WeightOnlyActivationType
{
    FP16,
    BF16
};

struct WeightOnlyPerChannel;
template <int GS>
struct WeightOnlyGroupWise;

struct WeightOnlyParams
{
    using ActType = void;
    const uint8_t*                   qweight;
    const ActType*                   scales;
    const ActType*                   zeros;
    const ActType*                   in;
    const ActType*                   bias;
    ActType*                         out;
    const int                        m;
    const int                        n;
    const int                        k;
    const int                        group_size;
    WeightOnlyQuantType              quant_type;
    WeightOnlyType                   weight_only_type;
    WeightOnlyActivationFunctionType act_func_type;
    WeightOnlyActivationType         act_type;
    const int                        sm;

    WeightOnlyParams(const uint8_t* _qweight,
                     const ActType*    _scales,
                     const ActType*    _zeros,
                     const ActType*    _in,
                     const ActType*    _bias,
                     ActType*          _out,
                     const int      _m,
                     const int      _n,
                     const int      _k,
                     const int      _group_size,
                     const WeightOnlyQuantType              _quant_type,
                     const WeightOnlyType                   _weight_only_type,
                     const WeightOnlyActivationFunctionType _act_func_type,
                     const WeightOnlyActivationType         _act_type):
        qweight(_qweight),
        scales(_scales),
        zeros(_zeros),
        in(_in),
        bias(_bias),
        out(_out),
        m(_m),
        n(_n),
        k(_k),
        group_size(_group_size),
        quant_type(_quant_type),
        weight_only_type(_weight_only_type),
        act_func_type(_act_func_type),
        act_type(_act_type),
        sm(fastertransformer::rocm::getSMVersion())
    {
    }
};

}  // namespace weight
}  // namespace rocm
}  // namespace fastertransformer

