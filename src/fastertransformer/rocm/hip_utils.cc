#include "hip_utils.h"

namespace fastertransformer {
namespace rocm {

/* **************************** debug tools ********************************* */

template<typename T>
void print_to_file(const T* result, const int size, const char* file, hipStream_t stream, std::ios::openmode open_mode)
{
    check_hip_error(hipDeviceSynchronize());
    check_hip_error(hipGetLastError());
    printf("[INFO] file: %s with size %d.\n", file, size);
    std::ofstream outFile(file, open_mode);
    if (outFile) {
        T* tmp = new T[size];
        check_hip_error(hipMemcpyAsync(tmp, result, sizeof(T) * size, hipMemcpyDeviceToHost, stream));
        for (int i = 0; i < size; ++i) {
            float val = special_cast<float, T>(tmp[i]);
            outFile << val << std::endl;
        }
        delete[] tmp;
    }
    else {
        throw std::runtime_error(std::string("[FT][ERROR] Cannot open file: ") + file + "\n");
    }
    check_hip_error(hipDeviceSynchronize());
    check_hip_error(hipGetLastError());
}

template void
print_to_file(const float* result, const int size, const char* file, hipStream_t stream, std::ios::openmode open_mode);
template void
print_to_file(const half* result, const int size, const char* file, hipStream_t stream, std::ios::openmode open_mode);
#if ENABLE_BF16
template void print_to_file(
    const __hip_bfloat16* result, const int size, const char* file, hipStream_t stream, std::ios::openmode open_mode);
#endif

template<typename T>
void print_abs_mean(const T* buf, uint size, hipStream_t stream, std::string name)
{
    if (buf == nullptr) {
        FT_LOG_WARNING("It is an nullptr, skip!");
        return;
    }
    check_hip_error(hipDeviceSynchronize());
    check_hip_error(hipGetLastError());
    T* h_tmp = new T[size];
    check_hip_error(hipMemcpyAsync(h_tmp, buf, sizeof(T) * size, hipMemcpyDeviceToHost, stream));
    check_hip_error(hipDeviceSynchronize());
    check_hip_error(hipGetLastError());
    double   sum        = 0.0f;
    uint64_t zero_count = 0;
    float    max_val    = -1e10;
    bool     find_inf   = false;
    for (uint i = 0; i < size; i++) {
        if (std::isinf(special_cast<float, T>(h_tmp[i]))) {
            find_inf = true;
            continue;
        }
        sum += abs(special_cast<float, T>(h_tmp[i]));
        if (special_cast<float, T>(h_tmp[i]) == 0.0f) {
            zero_count++;
        }
        max_val = max_val > abs(special_cast<float, T>(h_tmp[i])) ? max_val : abs(special_cast<float, T>(h_tmp[i]));
    }
    printf("[INFO][FT] %20s size: %u, abs mean: %f, abs sum: %f, abs max: %f, find inf: %s",
           name.c_str(),
           size,
           sum / size,
           sum,
           max_val,
           find_inf ? "true" : "false");
    std::cout << std::endl;
    delete[] h_tmp;
    check_hip_error(hipDeviceSynchronize());
    check_hip_error(hipGetLastError());
}

template void print_abs_mean(const float* buf, uint size, hipStream_t stream, std::string name);
template void print_abs_mean(const half* buf, uint size, hipStream_t stream, std::string name);
#if ENABLE_BF16
template void print_abs_mean(const __hip_bfloat16* buf, uint size, hipStream_t stream, std::string name);
#endif
template void print_abs_mean(const int* buf, uint size, hipStream_t stream, std::string name);
template void print_abs_mean(const uint* buf, uint size, hipStream_t stream, std::string name);
template void print_abs_mean(const int8_t* buf, uint size, hipStream_t stream, std::string name);
#ifdef ENABLE_FP8
template void print_abs_mean(const hipblaslt_f8* buf, uint size, hipStream_t stream, std::string name);
#endif

template<typename T>
void print_to_screen(const T* result, const int size)
{
    if (result == nullptr) {
        FT_LOG_WARNING("It is an nullptr, skip! \n");
        return;
    }
    T* tmp = reinterpret_cast<T*>(malloc(sizeof(T) * size));
    check_hip_error(hipMemcpy(tmp, result, sizeof(T) * size, hipMemcpyDeviceToHost));
    for (int i = 0; i < size; ++i) {
        printf("%d, %f\n", i, special_cast<float, T>(tmp[i]));
    }
    free(tmp);
}

template void print_to_screen(const float* result, const int size);
template void print_to_screen(const half* result, const int size);
#if ENABLE_BF16
template void print_to_screen(const __hip_bfloat16* result, const int size);
#endif
template void print_to_screen(const int* result, const int size);
template void print_to_screen(const uint* result, const int size);
template void print_to_screen(const bool* result, const int size);
#ifdef ENABLE_FP8
template void print_to_screen(const hipblaslt_f8* result, const int size);
#endif

template<typename T>
void printMatrix(T* ptr, int m, int k, int stride, bool is_device_ptr)
{
    T* tmp;
    if (is_device_ptr) {
        // k < stride ; stride = col-dimension.
        tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
        check_hip_error(hipMemcpy(tmp, ptr, sizeof(T) * m * stride, hipMemcpyDeviceToHost));
        check_hip_error(hipDeviceSynchronize());
    }
    else {
        tmp = ptr;
    }

    for (int ii = -1; ii < m; ++ii) {
        if (ii >= 0) {
            printf("%02d ", ii);
        }
        else {
            printf("   ");
        }

        for (int jj = 0; jj < k; jj += 1) {
            if (ii >= 0) {
                printf("%7.3f ", special_cast<float, T>(tmp[ii * stride + jj]));
            }
            else {
                printf("%7d ", jj);
            }
        }
        printf("\n");
    }
    if (is_device_ptr) {
        free(tmp);
    }
}

template void printMatrix(float* ptr, int m, int k, int stride, bool is_device_ptr);
template void printMatrix(half* ptr, int m, int k, int stride, bool is_device_ptr);
#if ENABLE_BF16
template void printMatrix(__hip_bfloat16* ptr, int m, int k, int stride, bool is_device_ptr);
#endif

void printMatrix(unsigned long long* ptr, int m, int k, int stride, bool is_device_ptr)
{
    typedef unsigned long long T;
    T*                         tmp;
    if (is_device_ptr) {
        // k < stride ; stride = col-dimension.
        tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
        check_hip_error(hipMemcpy(tmp, ptr, sizeof(T) * m * stride, hipMemcpyDeviceToHost));
        check_hip_error(hipDeviceSynchronize());
    }
    else {
        tmp = ptr;
    }

    for (int ii = -1; ii < m; ++ii) {
        if (ii >= 0) {
            printf("%02d ", ii);
        }
        else {
            printf("   ");
        }

        for (int jj = 0; jj < k; jj += 1) {
            if (ii >= 0) {
                printf("%4llu ", tmp[ii * stride + jj]);
            }
            else {
                printf("%4d ", jj);
            }
        }
        printf("\n");
    }
    if (is_device_ptr) {
        free(tmp);
    }
}

void printMatrix(int* ptr, int m, int k, int stride, bool is_device_ptr)
{
    typedef int T;
    T*          tmp;
    if (is_device_ptr) {
        // k < stride ; stride = col-dimension.
        tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
        check_hip_error(hipMemcpy(tmp, ptr, sizeof(T) * m * stride, hipMemcpyDeviceToHost));
        check_hip_error(hipDeviceSynchronize());
    }
    else {
        tmp = ptr;
    }

    for (int ii = -1; ii < m; ++ii) {
        if (ii >= 0) {
            printf("%02d ", ii);
        }
        else {
            printf("   ");
        }

        for (int jj = 0; jj < k; jj += 1) {
            if (ii >= 0) {
                printf("%4d ", tmp[ii * stride + jj]);
            }
            else {
                printf("%4d ", jj);
            }
        }
        printf("\n");
    }
    if (is_device_ptr) {
        free(tmp);
    }
}

void printMatrix(size_t* ptr, int m, int k, int stride, bool is_device_ptr)
{
    typedef size_t T;
    T*             tmp;
    if (is_device_ptr) {
        // k < stride ; stride = col-dimension.
        tmp = reinterpret_cast<T*>(malloc(m * stride * sizeof(T)));
        check_hip_error(hipMemcpy(tmp, ptr, sizeof(T) * m * stride, hipMemcpyDeviceToHost));
        check_hip_error(hipDeviceSynchronize());
    }
    else {
        tmp = ptr;
    }

    for (int ii = -1; ii < m; ++ii) {
        if (ii >= 0) {
            printf("%02d ", ii);
        }
        else {
            printf("   ");
        }

        for (int jj = 0; jj < k; jj += 1) {
            if (ii >= 0) {
                printf("%4ld ", tmp[ii * stride + jj]);
            }
            else {
                printf("%4d ", jj);
            }
        }
        printf("\n");
    }
    if (is_device_ptr) {
        free(tmp);
    }
}

template<typename T>
void check_max_val(const T* result, const int size)
{
    T* tmp = new T[size];
    check_hip_error(hipMemcpy(tmp, result, sizeof(T) * size, hipMemcpyDeviceToHost));
    float max_val = -100000;
    for (int i = 0; i < size; i++) {
        float val = special_cast<float, T>(tmp[i]);
        if (val > max_val) {
            max_val = val;
        }
    }
    delete tmp;
    printf("[INFO][CUDA] addr %p max val: %f \n", result, max_val);
}

template void check_max_val(const float* result, const int size);
template void check_max_val(const half* result, const int size);
#if ENABLE_BF16
template void check_max_val(const __hip_bfloat16* result, const int size);
#endif

template<typename T>
void check_abs_mean_val(const T* result, const int size)
{
    T* tmp = new T[size];
    check_hip_error(hipMemcpy(tmp, result, sizeof(T) * size, hipMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += abs(special_cast<float, T>(tmp[i]));
    }
    delete tmp;
    printf("[INFO][CUDA] addr %p abs mean val: %f \n", result, sum / size);
}

template void check_abs_mean_val(const float* result, const int size);
template void check_abs_mean_val(const half* result, const int size);
#if ENABLE_BF16
template void check_abs_mean_val(const __hip_bfloat16* result, const int size);
#endif

/* ***************************** common utils ****************************** */

hipError_t getSetDevice(int i_device, int* o_device)
{
    int         current_dev_id = 0;
    hipError_t err            = hipSuccess;

    if (o_device != NULL) {
        err = hipGetDevice(&current_dev_id);
        if (err != hipSuccess) {
            return err;
        }
        if (current_dev_id == i_device) {
            *o_device = i_device;
        }
        else {
            err = hipSetDevice(i_device);
            if (err != hipSuccess) {
                return err;
            }
            *o_device = current_dev_id;
        }
    }
    else {
        err = hipSetDevice(i_device);
        if (err != hipSuccess) {
            return err;
        }
    }

    return hipSuccess;
}

FtCudaDataType getModelFileType(std::string ini_file, std::string section_name)
{
    FtCudaDataType model_file_type;
    INIReader      reader = INIReader(ini_file);
    if (reader.ParseError() < 0) {
        FT_LOG_WARNING("Can't load %s. Use FP32 as default", ini_file.c_str());
        model_file_type = FtCudaDataType::FP32;
    }
    else {
        std::string weight_data_type_str = std::string(reader.Get(section_name, "weight_data_type"));
        if (weight_data_type_str.find("fp32") != std::string::npos) {
            model_file_type = FtCudaDataType::FP32;
        }
        else if (weight_data_type_str.find("fp16") != std::string::npos) {
            model_file_type = FtCudaDataType::FP16;
        }
        else if (weight_data_type_str.find("bf16") != std::string::npos) {
            model_file_type = FtCudaDataType::BF16;
        }
        else {
            FT_LOG_WARNING("Invalid type %s. Use FP32 as default", weight_data_type_str.c_str());
            model_file_type = FtCudaDataType::FP32;
        }
    }
    return model_file_type;
}

/* ************************** end of common utils ************************** */
}  // namespace rocm
}  // namespace fastertransformer
