#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

template<>
struct HIP_vector_base<hip_bfloat16, 2> {
    using Native_vec_ = hip_bfloat16[2];

    union {
        Native_vec_ data;
        struct {
            hip_bfloat16 x;
            hip_bfloat16 y;
        };
    };

    using value_type = hip_bfloat16;

    __host__ __device__ HIP_vector_base() = default;
    __host__            __device__ explicit constexpr HIP_vector_base(hip_bfloat16 x_) noexcept: data{x_, x_} {}
    __host__            __device__ constexpr HIP_vector_base(hip_bfloat16 x_, hip_bfloat16 y_) noexcept: data{x_, y_} {}
    __host__            __device__ constexpr HIP_vector_base(const HIP_vector_base&) = default;
    __host__            __device__ constexpr HIP_vector_base(HIP_vector_base&&)      = default;
    __host__            __device__ ~HIP_vector_base()                                = default;
    __host__ __device__ HIP_vector_base& operator=(const HIP_vector_base&)           = default;
};

template<>
struct HIP_vector_base<half, 4> {
    using Native_vec_ = half[4];

    union {
        Native_vec_ data;
        struct {
            half x;
            half y;
            half z;
            half w;
        };
    };

    using value_type = half;

    __host__ __device__ HIP_vector_base() = default;
    __host__            __device__ explicit constexpr HIP_vector_base(half x_) noexcept: data{x_, x_, x_, x_} {}
    __host__ __device__ constexpr HIP_vector_base(half x_, half y_, half z_, half w_) noexcept: data{x_, y_, z_, w_} {}
    __host__ __device__ constexpr HIP_vector_base(const HIP_vector_base&)  = default;
    __host__ __device__ constexpr HIP_vector_base(HIP_vector_base&&)       = default;
    __host__ __device__ ~HIP_vector_base()                                 = default;
    __host__ __device__ HIP_vector_base& operator=(const HIP_vector_base&) = default;
};

namespace std {
template<>
struct is_convertible<double, hip_bfloat16>: std::true_type {};
}  // namespace std

#include <hip/amd_detail/amd_hip_vector_types.h>
#include <hip/hip_runtime.h>

struct bfloat162: public HIP_vector_type<hip_bfloat16, 2> {
    using HIP_vector_type<hip_bfloat16, 2>::HIP_vector_type;
    friend __host__ __device__ inline constexpr bfloat162 operator/(bfloat162 x, const bfloat162& y) noexcept {
        for (auto i = 0u; i != 2; ++i)
            x.data[i] /= y.data[i];
        return x;
    }
};

DECLOP_MAKE_TWO_COMPONENT(hip_bfloat16, bfloat162);

using half4 = HIP_vector_type<half, 4>;
DECLOP_MAKE_FOUR_COMPONENT(half, half4);

#define __nv_bfloat16 hip_bfloat16
#define __nv_bfloat162 bfloat162

static inline __device__ __host__ float __bfloat162float(__nv_bfloat16 x) {
    return x;
}
static inline __device__ __host__ float __low2float(__nv_bfloat162 x) {
    return x.x;
}
static inline __device__ __host__ float __high2float(__nv_bfloat162 x) {
    return x.y;
}
static inline __device__ __host__ __nv_bfloat162 __floats2bfloat162_rn(float x, float y) {
    return {__nv_bfloat16(x), __nv_bfloat16(y)};
}
static inline __device__ __host__ __nv_bfloat16 __float2bfloat16(float a) {
    return __nv_bfloat16(a);
}
static inline __device__ __host__ __nv_bfloat162 __float2bfloat162_rn(float x) {
    return __floats2bfloat162_rn(x, x);
}
static inline __device__ __host__ __nv_bfloat162 __ldg(const __nv_bfloat162* ptr) {
    return *ptr;
}
static inline __device__ __host__ __nv_bfloat16 __ldg(const __nv_bfloat16* ptr) {
    return *ptr;
}
static inline __device__ __host__ __nv_bfloat162 __habs2(__nv_bfloat162 a) {
    return __floats2bfloat162_rn(__builtin_fabsf(a.x), __builtin_fabsf(a.y));
}
static inline __device__ __host__ __nv_bfloat16 __habs(__nv_bfloat16 a) {
    return __float2bfloat16(__builtin_fabsf(a));
}

template<typename T>
__device__ inline T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width = 32) {
    (void)mask;
    return __shfl_xor(var, laneMask, width);
}

#define cudaStream_t hipStream_t

// Taken from cuda_utils.h

template<typename T>
struct packed_type_2;
template<>
struct packed_type_2<float> {
    using type = float;
};  // we don't need to pack float by default
template<>
struct packed_type_2<half> {
    using type = half2;
};

#ifdef ENABLE_BF16
template<>
struct packed_type_2<__nv_bfloat16> {
    using type = __nv_bfloat162;
};
#endif

template<typename T>
struct num_elems;
template<>
struct num_elems<float> {
    static constexpr int value = 1;
};
template<>
struct num_elems<float2> {
    static constexpr int value = 2;
};
template<>
struct num_elems<float4> {
    static constexpr int value = 4;
};

template<>
struct num_elems<half> {
    static constexpr int value = 1;
};
template<>
struct num_elems<half2> {
    static constexpr int value = 2;
};
template<>
struct num_elems<uint32_t> {
    static constexpr int value = 2;
};
template<>
struct num_elems<int32_t> {
    static constexpr int value = 2;
};
template<>
struct num_elems<int64_t> {
    static constexpr int value = 4;
};
template<>
struct num_elems<uint2> {
    static constexpr int value = 4;
};
template<>
struct num_elems<uint4> {
    static constexpr int value = 8;
};

#ifdef ENABLE_BF16
template<>
struct num_elems<__nv_bfloat16> {
    static constexpr int value = 1;
};
template<>
struct num_elems<__nv_bfloat162> {
    static constexpr int value = 2;
};
#endif

template<typename T, int num>
struct packed_as;
template<typename T>
struct packed_as<T, 1> {
    using type = T;
};
template<>
struct packed_as<half, 2> {
    using type = half2;
};
template<>
struct packed_as<float, 2> {
    using type = float2;
};
template<>
struct packed_as<int8_t, 2> {
    using type = int16_t;
};
template<>
struct packed_as<int32_t, 2> {
    using type = int2;
};
template<>
struct packed_as<half2, 1> {
    using type = half;
};
template<>
struct packed_as<float2, 1> {
    using type = float;
};
#ifdef ENABLE_BF16
template<>
struct packed_as<__nv_bfloat16, 2> {
    using type = __nv_bfloat162;
};
template<>
struct packed_as<__nv_bfloat162, 1> {
    using type = __nv_bfloat16;
};
#endif