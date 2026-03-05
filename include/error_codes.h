#ifndef ERROR_CODES_H
#define ERROR_CODES_H

#define ERROR_MSG_MAX 256

typedef enum {
    /* array related error codes 10<x> */
    ARRAY_INIT_FAILURE = 101,
    INVALID_IDX = 102,
    NON_BROADCASTABLE_ARRAYS = 103,
    SHAPE_MISMATCH = 104,
    INVALID_ARRAY = 105,
    INVALID_DTYPE = 106,
    REPEATED_ARRAY_DIMS = 107,
    INVALID_DIM = 108,

    /* tensor related error codes 20<x> */
    TENSOR_INIT_FAILURE = 201,
    INVALID_GRAD = 202,
    DEPENDENCY_ARR_INIT_FAILURE = 203,
    ENV_INIT_FAILURE = 204,
    ENV_PUSH_FAILURE = 205,
    FILE_WRITE_FAILURE = 206,
    FILE_READ_FAILURE = 207,
    FILE_FORMAT_ERROR = 208,
    ENV_RESOLVE_FAILURE = 209,

    /* autograd related error codes 30<x> */
    BACKWARD_FN_INIT_FAILURE = 301,
    NEXT_FNS_INIT_FAILURE = 302,
    GRAD_INIT_FAILURE = 303,
    INVALID_BACKWARD_PASS = 304,
    INVALID_NUM_INPUTS_OUTPUTS = 305,

    /* random related error codes 40<x> */
    PRNG_INIT_FAILURE = 401,
    INVALID_LOW_HIGH = 402,

    /* neural nets related error codes 50<x> */
    MODULE_ALLOC_FAILURE = 501,
} ErrorCode;

extern const char *ErrorCodes[];

typedef struct {
    ErrorCode code;
    char message[ERROR_MSG_MAX];
    const char *file;
    int line;
} Error;

void print_error(const Error *err);
void exit_with_error(const Error *err);

#define RUNTIME_ERROR(code_, msg_)                                             \
    do {                                                                       \
        Error err = (Error){.code = code_,                                     \
                            .message = msg_,                                   \
                            .file = __FILE__,                                  \
                            .line = __LINE__};                                 \
        exit_with_error(&err);                                                 \
    } while (0)

#define RUNTIME_ERRORF(code_, fmt_, ...)                                       \
    do {                                                                       \
        Error err;                                                             \
        err.code = code_;                                                      \
        snprintf(err.message, ERROR_MSG_MAX, fmt_, __VA_ARGS__);               \
        err.file = __FILE__;                                                   \
        err.line = __LINE__;                                                   \
        exit_with_error(&err);                                                 \
    } while (0)

#endif // !ERROR_CODES_H
