// nn::account: el usuario de la consola. NieR pide el usuario preseleccionado al arrancar (sin él se
// queda en "Loading system data..." reintentando) y con su Uid monta la partida guardada. Hay un único
// usuario local; las funciones de cuenta de red (Nintendo Account) responden que no hay, y el juego
// sigue sin las funciones en línea.

#include <cstdint>
#include <cstring>

#include "nnsdk/os.h"
#include "nnsdk/types.h"

namespace nn::os {
class SystemEvent;  // su primer miembro es el SystemEventType: misma dirección
}

namespace nn::account {

struct Uid {
    uint64_t data[2];
};
struct UserHandle {
    uint64_t data[3];
};
struct Nickname {
    char name[33];
};
struct NetworkServiceAccountId {
    uint64_t id;
};

namespace {

constexpr Uid kUser{{0x0123456789abcdefULL, 0xfedcba9876543210ULL}};
// Resultado de "no hay cuenta de red" (módulo account, 124): cualquier error sirve, el juego
// solo distingue éxito de fallo.
constexpr Result kResultNoNetworkAccount{124 | (280 << 9)};

}  // namespace

class AsyncContext {
public:
    AsyncContext();
    ~AsyncContext();
    Result HasDone(bool* done);
    Result GetResult();
    Result Cancel();
    Result GetSystemEvent(os::SystemEvent* event);

private:
    uint8_t storage_[0x20];  // el juego reserva el objeto; no guardamos estado
};

void Initialize() {}

bool TryOpenPreselectedUser(UserHandle* out) {
    out->data[0] = kUser.data[0];
    out->data[1] = kUser.data[1];
    out->data[2] = 1;
    return true;
}

Result GetUserId(Uid* out, const UserHandle&) {
    *out = kUser;
    return ResultSuccess;
}

Result GetNickname(Nickname* out, const Uid&) {
    memset(out, 0, sizeof *out);
    strncpy(out->name, "Flip", sizeof out->name - 1);
    return ResultSuccess;
}

Result GetNetworkServiceAccountId(NetworkServiceAccountId* out, const UserHandle&) {
    out->id = 0;
    return kResultNoNetworkAccount;
}
Result EnsureNetworkServiceAccountAvailable(const UserHandle&) { return kResultNoNetworkAccount; }
Result EnsureNetworkServiceAccountIdTokenCacheAsync(AsyncContext*, const UserHandle&) { return kResultNoNetworkAccount; }
Result LoadNetworkServiceAccountIdTokenCache(unsigned long* size, char*, unsigned long, const UserHandle&) {
    *size = 0;
    return kResultNoNetworkAccount;
}

AsyncContext::AsyncContext() { memset(storage_, 0, sizeof storage_); }
AsyncContext::~AsyncContext() {}
Result AsyncContext::HasDone(bool* done) {
    *done = true;
    return ResultSuccess;
}
Result AsyncContext::GetResult() { return kResultNoNetworkAccount; }
Result AsyncContext::Cancel() { return ResultSuccess; }
Result AsyncContext::GetSystemEvent(os::SystemEvent* event) {
    nnsdk::signal_system_event(reinterpret_cast<os::SystemEventType*>(event));  // ya terminó
    return ResultSuccess;
}

}  // namespace nn::account
