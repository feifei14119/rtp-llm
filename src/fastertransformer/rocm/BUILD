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
    ],
    copts = rocm_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)
