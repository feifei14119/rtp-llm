#pragma once

#include "src/fastertransformer/utils/logger.h"
#include "src/fastertransformer/utils/assert_utils.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#ifdef ENABLE_BF16
#include <hip/hip_bf16.h>
#endif

#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#ifdef SPARSITY_ENABLED
// #include <cusparseLt.h>
#endif

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fastertransformer {
#define cudaStream_t hipStream_t
#define sync_check_cuda_error sync_check_hip_error
#define check_cuda_error check_hip_error

namespace rocm {

/* **************************** type definition ***************************** */
enum HipblasDataType {
    FLOAT_DATATYPE    = 0,
    HALF_DATATYPE     = 1,
    BFLOAT16_DATATYPE = 2,
    INT8_DATATYPE     = 3,
    FP8_DATATYPE      = 4
};

enum FtHipDataType {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
    FP8  = 4
};

enum class OperationType {
    FP32,
    FP16,
    BF16,
    INT8,
    FP8
};

/* **************************** debug tools ********************************* */
static const char* _hipGetErrorEnum(hipError_t error) {
    return hipGetErrorString(error);
}

static const char* _hipGetErrorEnum(hipblasStatus_t error) {
    return hipblasStatusToString(error);
}

template<typename T>
void check(T result, char const* const func, const char* const file, int const line) {
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] ROCM runtime error: ") + (_hipGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
}
#define check_hip_error(val) rocm::check((val), #val, __FILE__, __LINE__)

inline void syncAndCheck(const char* const file, int const line) {
    // When FT_DEBUG_LEVEL=DEBUG, must check error
    static char* level_name = std::getenv("FT_DEBUG_LEVEL");
    if (level_name != nullptr) {
        static std::string level = std::string(level_name);
        if (level == "DEBUG") {
            check_hip_error(hipDeviceSynchronize());
            hipError_t result = hipGetLastError();
            if (result) {
                throw std::runtime_error(std::string("[FT][ERROR] ROCM runtime error: ") + (_hipGetErrorEnum(result))
                                         + " " + file + ":" + std::to_string(line) + " \n");
            }
            FT_LOG_DEBUG(fmtstr("run syncAndCheck at %s:%d", file, line));
        }
    }
#define sync_check_hip_error() rocm::syncAndCheck(__FILE__, __LINE__)

#ifndef NDEBUG
    check_hip_error(hipDeviceSynchronize());
    hipError_t result = hipGetLastError();
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] ROCM runtime error: ") + (_hipGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
#endif
}

template<typename T>
void print_to_file(const T*           result,
                   const int          size,
                   const char*        file,
                   hipStream_t        stream    = 0,
                   std::ios::openmode open_mode = std::ios::out);

template<typename T>
void print_abs_mean(const T* buf, uint size, hipStream_t stream, std::string name = "");

template<typename T>
void print_to_screen(const T* result, const int size);

template<typename T>
void printMatrix(T* ptr, int m, int k, int stride, bool is_device_ptr);

void printMatrix(unsigned long long* ptr, int m, int k, int stride, bool is_device_ptr);
void printMatrix(int* ptr, int m, int k, int stride, bool is_device_ptr);
void printMatrix(size_t* ptr, int m, int k, int stride, bool is_device_ptr);

template<typename T>
void check_max_val(const T* result, const int size);

template<typename T>
void check_abs_mean_val(const T* result, const int size);

#define PRINT_FUNC_NAME_()                                                                                             \
    do {                                                                                                               \
        std::cout << "[FT][CALL] " << __FUNCTION__ << " " << std::endl;                                                \
    } while (0)

#define FT_CHECK_WITH_INFO(val, info, ...)                                                                             \
    do {                                                                                                               \
        bool is_valid_val = (val);                                                                                     \
        if (!is_valid_val) {                                                                                           \
            fastertransformer::myAssert(                                                                               \
                is_valid_val, __FILE__, __LINE__, fastertransformer::fmtstr(info, ##__VA_ARGS__));                     \
        }                                                                                                              \
    } while (0)

/*************Time Handling**************/
class HipTimer {
private:
    hipEvent_t  event_start_;
    hipEvent_t  event_stop_;
    hipStream_t stream_;

public:
    explicit HipTimer(hipStream_t stream = 0) {
        stream_ = stream;
    }
    void start() {
        check_hip_error(hipEventCreate(&event_start_));
        check_hip_error(hipEventCreate(&event_stop_));
        check_hip_error(hipEventRecord(event_start_, stream_));
    }
    float stop() {
        float time;
        check_hip_error(hipEventRecord(event_stop_, stream_));
        check_hip_error(hipEventSynchronize(event_stop_));
        check_hip_error(hipEventElapsedTime(&time, event_start_, event_stop_));
        check_hip_error(hipEventDestroy(event_start_));
        check_hip_error(hipEventDestroy(event_stop_));
        return time;
    }
    ~HipTimer() {}
};

static double diffTime(timeval start, timeval end) {
    return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}

/* ***************************** common utils ****************************** */

inline void print_mem_usage(std::string time = "after allocation") {
    size_t free_bytes, total_bytes;
    check_hip_error(hipMemGetInfo(&free_bytes, &total_bytes));
    float free  = static_cast<float>(free_bytes) / 1024.0 / 1024.0 / 1024.0;
    float total = static_cast<float>(total_bytes) / 1024.0 / 1024.0 / 1024.0;
    float used  = total - free;
    printf("%-20s: free: %5.2f GB, total: %5.2f GB, used: %5.2f GB\n", time.c_str(), free, total, used);
}

inline int getSMVersion() {
    int device{-1};
    check_hip_error(hipGetDevice(&device));
    int computeCapbilityMajor = 0;
    int computeCapbilityMinor = 0;
    check_hip_error(hipDeviceGetAttribute(&computeCapbilityMajor, hipDeviceAttributeComputeCapabilityMajor, device));
    check_hip_error(hipDeviceGetAttribute(&computeCapbilityMinor, hipDeviceAttributeComputeCapabilityMinor, device));
    return computeCapbilityMajor * 10 + computeCapbilityMinor;
}

inline int getMaxSharedMemoryPerBlock() {
    int device{-1};
    check_hip_error(hipGetDevice(&device));
    int max_shared_memory_size = 0;
    check_hip_error(hipDeviceGetAttribute(&max_shared_memory_size, hipDeviceAttributeMaxSharedMemoryPerBlock, device));
    return max_shared_memory_size;
}

inline std::string getDeviceName() {
    int device{-1};
    check_hip_error(hipGetDevice(&device));
    hipDeviceProp_t props;
    check_hip_error(hipGetDeviceProperties(&props, device));
    return std::string(props.name);
}

inline int div_up(int a, int n) {
    return (a + n - 1) / n;
}

hipError_t getSetDevice(int i_device, int* o_device = NULL);

inline int getDevice() {
    int current_dev_id = 0;
    check_hip_error(hipGetDevice(&current_dev_id));
    return current_dev_id;
}

inline int getDeviceCount() {
    int count = 0;
    check_hip_error(hipGetDeviceCount(&count));
    return count;
}

template<typename T>
HipblasDataType getHipblasDataType() {
    if (std::is_same<T, half>::value) {
        return HALF_DATATYPE;
    }
#if ENABLE_BF16
    else if (std::is_same<T, hip_bfloat16>::value) {
        return BFLOAT16_DATATYPE;
    }
#endif
    else if (std::is_same<T, float>::value) {
        return FLOAT_DATATYPE;
    } else {
        FT_CHECK(false);
        return FLOAT_DATATYPE;
    }
}

template<typename T>
hipblasDatatype_t getHipDataType() {
    if (std::is_same<T, half>::value) {
        return HIPBLAS_R_16F;
    }
#if ENABLE_BF16
    else if (std::is_same<T, hip_bfloat16>::value) {
        return HIPBLAS_R_16B;
    }
#endif
    else if (std::is_same<T, float>::value) {
        return HIPBLAS_R_32F;
    } else {
        FT_CHECK(false);
        return HIPBLAS_R_32F;
    }
}

template<HipblasDataType T>
struct getTypeFromHipDataType {
    using Type = float;
};

template<>
struct getTypeFromHipDataType<HALF_DATATYPE> {
    using Type = half;
};

#if ENABLE_BF16
template<>
struct getTypeFromHipDataType<BFLOAT16_DATATYPE> {
    using Type = hip_bfloat16;
};
#endif

FtHipDataType getModelFileType(std::string ini_file, std::string section_name);

template<typename T1, typename T2>
void compareTwoTensor(
    const T1* pred, const T2* ref, const int size, const int print_size = 0, const std::string filename = "") {
    T1* h_pred = new T1[size];
    T2* h_ref  = new T2[size];
    check_hip_error(hipMemcpy(h_pred, pred, size * sizeof(T1), hipMemcpyDeviceToHost));
    check_hip_error(hipMemcpy(h_ref, ref, size * sizeof(T2), hipMemcpyDeviceToHost));

    FILE* fd = nullptr;
    if (filename != "") {
        fd = fopen(filename.c_str(), "w");
        fprintf(fd, "| %10s | %10s | %10s | %10s | \n", "pred", "ref", "abs_diff", "rel_diff(%)");
    }

    if (print_size > 0) {
        FT_LOG_INFO("  id |   pred  |   ref   |abs diff | rel diff (%) |");
    }
    float mean_abs_diff = 0.0f;
    float mean_rel_diff = 0.0f;
    int   count         = 0;
    for (int i = 0; i < size; i++) {
        if (i < print_size) {
            FT_LOG_INFO("%4d | % 6.4f | % 6.4f | % 6.4f | % 7.4f |",
                        i,
                        (float)h_pred[i],
                        (float)h_ref[i],
                        abs((float)h_pred[i] - (float)h_ref[i]),
                        abs((float)h_pred[i] - (float)h_ref[i]) / (abs((float)h_ref[i]) + 1e-6f) * 100.f);
        }
        if ((float)h_pred[i] == 0) {
            continue;
        }
        count += 1;
        mean_abs_diff += abs((float)h_pred[i] - (float)h_ref[i]);
        mean_rel_diff += abs((float)h_pred[i] - (float)h_ref[i]) / (abs((float)h_ref[i]) + 1e-6f) * 100.f;

        if (fd != nullptr) {
            fprintf(fd,
                    "| %10.5f | %10.5f | %10.5f | %11.5f |\n",
                    (float)h_pred[i],
                    (float)h_ref[i],
                    abs((float)h_pred[i] - (float)h_ref[i]),
                    abs((float)h_pred[i] - (float)h_ref[i]) / (abs((float)h_ref[i]) + 1e-6f) * 100.f);
        }
    }
    mean_abs_diff = mean_abs_diff / (float)count;
    mean_rel_diff = mean_rel_diff / (float)count;
    FT_LOG_INFO("mean_abs_diff: % 6.4f, mean_rel_diff: % 6.4f (%%)", mean_abs_diff, mean_rel_diff);

    if (fd != nullptr) {
        fprintf(fd, "mean_abs_diff: % 6.4f, mean_rel_diff: % 6.4f (%%)", mean_abs_diff, mean_rel_diff);
        fclose(fd);
    }
    delete[] h_pred;
    delete[] h_ref;
}

/* ************************** end of common utils ************************** */
}  // namespace rocm

}  // namespace fastertransformer
