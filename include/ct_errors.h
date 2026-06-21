#ifndef CT_ERRORS_H
#define CT_ERRORS_H

#if defined(__GNUC__) || defined(__clang__)
void __runtime_error_impl(int error_code, const char *code_name,
                          const char *file_path, const char *func_name,
                          const char *format, ...)
    __attribute__((format(printf, 5, 6)));
#else
void __runtime_error_impl(int error_code, const char *code_name,
                          const char *file_path, const char *func_name,
                          const char *format, ...);
#endif

#define ct_runtime_error(error_code, ...)                                      \
    do {                                                                       \
        __runtime_error_impl(error_code, #error_code, __FILE__, __func__,      \
                             __VA_ARGS__);                                     \
    } while (0);

#endif // !CT_ERRORS_H
