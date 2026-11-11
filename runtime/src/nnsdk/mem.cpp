// nn::mem::StandardAllocator sobre el malloc de bionic. El juego le da una región para
// gestionar; se ignora: le da igual de dónde salga la memoria mientras Free la acepte.

#include <cstddef>
#include <cstdlib>

namespace nn::mem {

class StandardAllocator {
public:
    StandardAllocator();
    void Initialize(void* address, size_t size);
    void Finalize();
    void* Allocate(size_t size);
    void* Allocate(size_t size, size_t alignment);
    void Free(void* p);
};

// Sin miembros: el objeto vive en memoria del juego y no se toca.
StandardAllocator::StandardAllocator() {}
void StandardAllocator::Initialize(void*, size_t) {}
void StandardAllocator::Finalize() {}

void* StandardAllocator::Allocate(size_t size) { return Allocate(size, 16); }

void* StandardAllocator::Allocate(size_t size, size_t alignment) {
    void* p = nullptr;
    if (alignment < sizeof(void*)) alignment = sizeof(void*);
    return posix_memalign(&p, alignment, size ? size : 1) == 0 ? p : nullptr;
}

void StandardAllocator::Free(void* p) { free(p); }

}  // namespace nn::mem
