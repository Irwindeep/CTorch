/**
 * @file error.h
 * @brief Error Handling for CTorch
 *
 */

#ifndef CTORCH_ERROR_H
#define CTORCH_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

#if defined(__GNUC__) || defined(__clang__)
void ct_runtime_error_impl(int status_code, const char *status_msg,
                           const char *file_path, const char *func_name,
                           const char *error_msg_fmt, ...)
    __attribute__((format(printf, 5, 6)));
#else
void ct_runtime_error_impl(int status_code, const char *status_msg,
                           const char *file_path, const char *func_name,
                           const char *error_msg_fmt, ...);
#endif

#define ct_runtime_error(status_code, ...)                                     \
    do {                                                                       \
        ct_runtime_error_impl(status_code, #status_code, __FILE__, __func__,   \
                              __VA_ARGS__);                                    \
    } while (0);

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

#endif /* end of include guard: CTORCH_ERROR_H */
