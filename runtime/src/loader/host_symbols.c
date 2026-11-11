// Funciones de bionic, libm, compiler-rt y libc++abi con la misma ABI que la libc de
// Nintendo (musl, AArch64): el juego las llama directo. Lo que no coincide (errno,
// fpclassify, strdup con el malloc del juego…) lo define libnnsdk.so, que tiene prioridad.

#include <ctype.h>
#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/time.h>
#include <time.h>

#include "loader/host_symbols.h"

// compiler-rt: long double es de 128 bits en los dos lados.
void __addtf3(void), __subtf3(void), __multf3(void), __eqtf2(void), __netf2(void), __lttf2(void),
    __unordtf2(void), __extenddftf2(void), __trunctfdf2(void), __fixtfsi(void), __fixunstfsi(void),
    __floatditf(void), __floatsitf(void), __floatunsitf(void);
// libc++abi / libunwind (ABI Itanium).
void _Unwind_Resume(void), __gxx_personality_v0(void), __cxa_begin_catch(void), __cxa_guard_abort(void),
    __cxa_guard_acquire(void), __cxa_guard_release(void), __cxa_pure_virtual(void), _ZSt9terminatev(void);
extern char _ZTVN10__cxxabiv117__class_type_infoE[], _ZTVN10__cxxabiv120__si_class_type_infoE[],
    _ZTVN10__cxxabiv121__vmi_class_type_infoE[];

#define S(name) {#name, (void*)&name}

static const HostSymbol kSymbols[] = {
    S(__addtf3), S(__subtf3), S(__multf3), S(__eqtf2), S(__netf2), S(__lttf2), S(__unordtf2),
    S(__extenddftf2), S(__trunctfdf2), S(__fixtfsi), S(__fixunstfsi), S(__floatditf), S(__floatsitf),
    S(__floatunsitf),

    S(_Unwind_Resume), S(__gxx_personality_v0), S(__cxa_begin_catch), S(__cxa_guard_abort),
    S(__cxa_guard_acquire), S(__cxa_guard_release), S(__cxa_pure_virtual), S(_ZSt9terminatev),
    S(_ZTVN10__cxxabiv117__class_type_infoE), S(_ZTVN10__cxxabiv120__si_class_type_infoE),
    S(_ZTVN10__cxxabiv121__vmi_class_type_infoE),

    S(abort), S(exit), S(getenv), S(atof), S(atoi), S(atol), S(strtod), S(strtol), S(strtoul), S(strtoull),
    S(div), S(qsort), S(bsearch), S(rand), S(setjmp), S(longjmp),

    S(memchr), S(memcmp), S(memcpy), S(memmove), S(memset), S(strcat), S(strchr), S(strcmp), S(strcpy),
    S(strcspn), S(strerror), S(strlen), S(strncat), S(strncmp), S(strncpy), S(strnlen), S(strrchr), S(strspn),
    S(strstr), S(strcasecmp), S(strncasecmp), S(isalnum), S(tolower), S(toupper),

    S(stdin), S(stdout), S(stderr), S(fopen), S(fclose), S(fread), S(fwrite), S(fseek), S(ftell), S(feof),
    S(fgetc), S(fputc), S(putc), S(putchar), S(puts), S(printf), S(fprintf), S(sprintf), S(snprintf),
    S(vsprintf), S(vsnprintf), S(sscanf),

    S(time), S(gettimeofday), S(gmtime),

    S(acos), S(acosh), S(asin), S(asinh), S(atan), S(atan2), S(atanh), S(cbrt), S(cos), S(cosf), S(cosh),
    S(erf), S(erfc), S(exp), S(exp2), S(exp2f), S(expf), S(fmod), S(fmodf), S(frexp), S(hypot), S(ldexp),
    S(log), S(log10), S(log10f), S(log2), S(log2f), S(logf), S(modf), S(modff), S(pow), S(powf), S(sin),
    S(sinf), S(sinh), S(sqrt), S(sqrtf), S(tan), S(tanf), S(tanh),
};

const HostSymbol* host_symbols(size_t* count) {
    *count = sizeof kSymbols / sizeof kSymbols[0];
    return kSymbols;
}
