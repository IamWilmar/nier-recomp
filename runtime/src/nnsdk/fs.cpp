// nn::fs sobre carpetas de la Flip. Cada punto de montaje del juego ("rom:", "save:"…) se
// traduce a una carpeta bajo NIER_DATA (por defecto /sdcard/NieR):
//   rom:  -> romfs/      (RomFS del juego con el update aplicado)
//   aoc:  -> aoc/<n>/    (DLC)
//   save: -> save/       (partidas)
// salvo que <datos>/rutas.txt diga otra cosa para rom: y aoc: (rutas.h).

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "nnsdk/rutas.h"
#include "nnsdk/types.h"

namespace nn {
namespace account {
struct Uid {
    uint64_t data[2];
};
}  // namespace account

namespace fs {

struct FileHandle {
    void* handle;
};
struct DirectoryHandle {
    void* handle;
};
struct DirectoryEntry {
    char name[769];
    char reserved0[3];
    int8_t type;
    char reserved1[3];
    int64_t size;
};
struct WriteOption {
    int flags;
};
enum DirectoryEntryType { DirectoryEntryType_Directory = 0, DirectoryEntryType_File = 1 };

}  // namespace fs
}  // namespace nn

using namespace nn;
using namespace nn::fs;

namespace {

// Códigos de nn::fs (módulo 2): PathNotFound, PathAlreadyExists, TargetNotFound.
constexpr Result kPathNotFound{0x202}, kPathAlreadyExists{0x402}, kTargetNotFound{0x7D402};
constexpr int kOpenWrite = 2, kOpenAppend = 4;
constexpr int kDirOpenDirs = 1, kDirOpenFiles = 2;

std::mutex g_mounts_m;
std::map<std::string, std::string> g_mounts;
int g_logged_opens = 0;

struct File {
    int fd;
    int mode;
};
struct Directory {
    std::vector<DirectoryEntry> entries;
    size_t next = 0;
};

std::string data_dir() {
    const char* env = getenv("NIER_DATA");
    return env ? env : "/sdcard/NieR";
}

Result from_errno() {
    if (errno == ENOENT || errno == ENOTDIR) return kPathNotFound;
    if (errno == EEXIST) return kPathAlreadyExists;
    return kTargetNotFound;
}

void mount(const char* name, const std::string& dir) {
    mkdir(dir.c_str(), 0775);
    std::lock_guard lock(g_mounts_m);
    g_mounts[name] = dir;
    printf("[fs] %s: -> %s\n", name, dir.c_str());
}

// "rom:/data/a.dat" -> "<carpeta de rom>/data/a.dat". Vacío si el punto de montaje no existe.
std::string host_path(const char* path) {
    const char* colon = strchr(path, ':');
    if (!colon) return {};
    std::lock_guard lock(g_mounts_m);
    auto it = g_mounts.find(std::string(path, colon - path));
    if (it == g_mounts.end()) return {};
    const char* rest = colon + 1;
    while (*rest == '/') rest++;
    return it->second + "/" + rest;
}

void log_open(const char* path, bool ok) {
    if (!ok)
        printf("[fs] no existe: %s\n", path);
    else if (g_logged_opens++ < 40)
        printf("[fs] abre %s\n", path);
}

}  // namespace

namespace nn::fs {

void SetAllocator(void* (*)(size_t), void (*)(void*, size_t)) {}

Result QueryMountRomCacheSize(size_t* out) {
    *out = 0;
    return ResultSuccess;
}
Result QueryMountAddOnContentCacheSize(size_t* out, int) {
    *out = 0;
    return ResultSuccess;
}

Result MountRom(const char* name, void*, size_t) {
    mount(name, rutas::get().romfs);
    return ResultSuccess;
}
Result MountAddOnContent(const char* name, int index, void*, size_t) {
    const auto& aoc = rutas::get().aoc;
    if (const auto it = aoc.find(index); it != aoc.end()) {
        mount(name, it->second);
    } else {
        mkdir((data_dir() + "/aoc").c_str(), 0775);
        mount(name, data_dir() + "/aoc/" + std::to_string(index));
    }
    return ResultSuccess;
}
Result EnsureSaveData(const account::Uid&) {
    mkdir(data_dir().c_str(), 0775);
    mkdir((data_dir() + "/save").c_str(), 0775);
    return ResultSuccess;
}
Result MountSaveData(const char* name, const account::Uid& uid) {
    EnsureSaveData(uid);
    mount(name, data_dir() + "/save");
    return ResultSuccess;
}
Result Commit(const char* name) {
    printf("[fs] commit %s\n", name);
    sync();
    return ResultSuccess;
}
void Unmount(const char* name) {
    std::lock_guard lock(g_mounts_m);
    g_mounts.erase(name);
}

Result OpenFile(FileHandle* out, const char* path, int mode) {
    std::string p = host_path(path);
    int flags = (mode & kOpenWrite) ? O_RDWR : O_RDONLY;
    int fd = p.empty() ? -1 : open(p.c_str(), flags | O_CLOEXEC);
    log_open(path, fd >= 0);
    if (fd < 0) return p.empty() ? kPathNotFound : from_errno();
    out->handle = new File{fd, mode};
    return ResultSuccess;
}
void CloseFile(FileHandle h) {
    auto* f = static_cast<File*>(h.handle);
    close(f->fd);
    delete f;
}
Result ReadFile(size_t* out, FileHandle h, int64_t offset, void* buffer, size_t size) {
    ssize_t n = pread(static_cast<File*>(h.handle)->fd, buffer, size, offset);
    if (n < 0) return from_errno();
    if (out) *out = n;
    return ResultSuccess;
}
Result ReadFile(FileHandle h, int64_t offset, void* buffer, size_t size) {
    return ReadFile(nullptr, h, offset, buffer, size);
}
Result WriteFile(FileHandle h, int64_t offset, const void* buffer, size_t size, const WriteOption&) {
    auto* f = static_cast<File*>(h.handle);
    if (!(f->mode & (kOpenWrite | kOpenAppend))) return kTargetNotFound;
    return pwrite(f->fd, buffer, size, offset) == static_cast<ssize_t>(size) ? ResultSuccess : from_errno();
}
Result FlushFile(FileHandle h) {
    fsync(static_cast<File*>(h.handle)->fd);
    return ResultSuccess;
}
Result GetFileSize(int64_t* out, FileHandle h) {
    struct stat st;
    if (fstat(static_cast<File*>(h.handle)->fd, &st) != 0) return from_errno();
    *out = st.st_size;
    return ResultSuccess;
}
Result SetFileSize(FileHandle h, int64_t size) {
    return ftruncate(static_cast<File*>(h.handle)->fd, size) == 0 ? ResultSuccess : from_errno();
}

Result CreateFile(const char* path, int64_t size) {
    std::string p = host_path(path);
    printf("[fs] crea %s (%lld bytes)\n", path, static_cast<long long>(size));
    int fd = p.empty() ? -1 : open(p.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0664);
    if (fd < 0) return p.empty() ? kPathNotFound : from_errno();
    int ok = ftruncate(fd, size);
    close(fd);
    return ok == 0 ? ResultSuccess : from_errno();
}
Result DeleteFile(const char* path) {
    std::string p = host_path(path);
    return !p.empty() && unlink(p.c_str()) == 0 ? ResultSuccess : from_errno();
}
Result RenameFile(const char* from, const char* to) {
    std::string a = host_path(from), b = host_path(to);
    return !a.empty() && !b.empty() && rename(a.c_str(), b.c_str()) == 0 ? ResultSuccess : from_errno();
}
Result GetEntryType(DirectoryEntryType* out, const char* path) {
    std::string p = host_path(path);
    struct stat st;
    if (p.empty() || stat(p.c_str(), &st) != 0) return kPathNotFound;
    *out = S_ISDIR(st.st_mode) ? DirectoryEntryType_Directory : DirectoryEntryType_File;
    return ResultSuccess;
}

Result OpenDirectory(DirectoryHandle* out, const char* path, int mode) {
    std::string p = host_path(path);
    DIR* d = p.empty() ? nullptr : opendir(p.c_str());
    if (!d) {
        printf("[fs] no existe la carpeta: %s\n", path);
        return kPathNotFound;
    }
    auto* dir = new Directory;
    while (dirent* e = readdir(d)) {
        if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
        struct stat st;
        if (stat((p + "/" + e->d_name).c_str(), &st) != 0) continue;
        bool is_dir = S_ISDIR(st.st_mode);
        if (!(mode & (is_dir ? kDirOpenDirs : kDirOpenFiles))) continue;
        DirectoryEntry entry{};
        snprintf(entry.name, sizeof entry.name, "%s", e->d_name);
        entry.type = is_dir ? DirectoryEntryType_Directory : DirectoryEntryType_File;
        entry.size = is_dir ? 0 : st.st_size;
        dir->entries.push_back(entry);
    }
    closedir(d);
    out->handle = dir;
    return ResultSuccess;
}
void CloseDirectory(DirectoryHandle h) { delete static_cast<Directory*>(h.handle); }
Result GetDirectoryEntryCount(int64_t* out, DirectoryHandle h) {
    *out = static_cast<Directory*>(h.handle)->entries.size();
    return ResultSuccess;
}
Result ReadDirectory(int64_t* out, DirectoryEntry* entries, DirectoryHandle h, int64_t max) {
    auto* dir = static_cast<Directory*>(h.handle);
    int64_t n = 0;
    while (n < max && dir->next < dir->entries.size()) entries[n++] = dir->entries[dir->next++];
    *out = n;
    return ResultSuccess;
}

}  // namespace nn::fs
