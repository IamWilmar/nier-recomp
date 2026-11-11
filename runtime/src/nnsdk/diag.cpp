// nn::diag: aserciones y abortos del SDK. Se imprimen y se aborta, como en la consola.

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "nnsdk/types.h"

namespace nn::diag {

enum AssertionType {};

namespace detail {

bool NotNullImpl(uintptr_t value) { return value != 0; }

[[noreturn]] void AbortImpl(const char* condition, const char* function, const char* file, int line) {
    printf("[nn::diag] abort: %s en %s (%s:%d)\n", condition, function, file, line);
    fflush(stdout);
    abort();
}

[[noreturn]] void AbortImpl(const char* condition, const char* function, const char* file, int line,
                            const Result* result, const char* format, ...) {
    printf("[nn::diag] abort: %s en %s (%s:%d), result 0x%x: ", condition, function, file, line,
           result ? result->value : 0);
    va_list args;
    va_start(args, format);
    if (format) vprintf(format, args);
    va_end(args);
    printf("\n");
    fflush(stdout);
    abort();
}

[[noreturn]] void OnAssertionFailure(AssertionType type, const char* condition, const char* function,
                                     const char* file, int line) {
    printf("[nn::diag] aserción %d: %s en %s (%s:%d)\n", type, condition, function, file, line);
    fflush(stdout);
    abort();
}

}  // namespace detail
}  // namespace nn::diag
