load("//:def.bzl", "rocm_copts",)

cc_library(
    name = "cuda_shims",
    hdrs = [
        "cuda_shims.h",
    ],
    deps = [
        "@local_config_rocm//rocm:rocm_headers",
    ],
    copts = rocm_copts(),
    include_prefix = "src",
    visibility = ["//visibility:public"],
)
