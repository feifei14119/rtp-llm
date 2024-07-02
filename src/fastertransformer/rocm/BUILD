load("//:def.bzl", "rocm_copts",)


cc_library(
    name = "weight_only_kernel",
    srcs = glob([
            "weightOnlyBatchedGemv/weightOnlyBatchedGemvInt8b.cc",
        ]),
    hdrs = glob([
            "*.h",
            "weightOnlyBatchedGemv/*.h",
        ]),
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rocm",
        "//src/fastertransformer/utils:logger",
        "//src/fastertransformer/utils:assert_utils",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/core:allocator",
    ],
    copts = rocm_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
    alwayslink = True,
)


cc_library(
    name = "rocm_utils",
    srcs = glob([
        "*.cc",
        "weightOnlyBatchedGemv/kernelLauncher.cc",
    ]),
    hdrs = glob([
        "*.h",
        "*.cuh",
        "weightOnlyBatchedGemv/*.h",
        "weightOnlyBatchedGemv/common.h",
        "weightOnlyBatchedGemv/kernelLauncher.h",
    ]),
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rocm",
        "@local_config_rocm//rocm:rocblas",
        "//src/fastertransformer/utils:logger",
        "//src/fastertransformer/utils:assert_utils",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/core:allocator",
        "//src/fastertransformer/core:types",
        ":ck_mha_lib",
    ],
    copts=rocm_copts() + [
        "-Isrc/fastertransformer/rocm/composable_kernel/include",
        "-Isrc/fastertransformer/rocm/composable_kernel/library/include",
        "-Isrc/fastertransformer/rocm/composable_kernel/example/ck_tile/01_fmha",
    ],
    include_prefix = "src",
    visibility = ["//visibility:public"],
)

cc_library(
    name="memory_utils",
    srcs=[
        # "memory_utils.cu",
    ],

    hdrs=[
        # "memory_utils.h",
    ],

    deps=[
        ":rocm_utils",
        "//src/fastertransformer/utils:utils",
        "//src/fastertransformer/core:Tensor",
        "//src/fastertransformer/core:allocator",
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rocm",
    ],
    copts=rocm_copts(),
    include_prefix="src",
    visibility=["//visibility:public"],
)

cc_library(
    name="ck_mha_lib",
    srcs=glob(["ck_bd/fmha_*wd*.cpp"]),
    hdrs=glob(["composable_kernel/**/*.hpp"]),

    copts=rocm_copts() + [
        "-Isrc/fastertransformer/rocm/composable_kernel/include",
        "-Isrc/fastertransformer/rocm/composable_kernel/library/include",
        "-Isrc/fastertransformer/rocm/composable_kernel/example/ck_tile/01_fmha",
    ],
    deps=[
        "@local_config_rocm//rocm:rocm_headers",
        "@local_config_rocm//rocm:rocm",
    ],
    visibility=["//visibility:public"],
    linkopts=["-pthread"]
)

cc_binary(
    name="ck_mha_test1",
    srcs=["composable_kernel/example/ck_tile/01_fmha/fmha_fwd.cpp",],
    deps=[
        ":ck_mha_lib",
    ],
    copts=rocm_copts() + ["-Isrc/fastertransformer/rocm/composable_kernel/include",
                          "-Isrc/fastertransformer/rocm/composable_kernel/library/include",
                          "-Isrc/fastertransformer/rocm/composable_kernel/example/ck_tile/01_fmha",
                          ],
)