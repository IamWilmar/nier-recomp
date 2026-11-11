// nn::time: relojes del sistema y conversiones de calendario, sobre la hora de Android. NieR fecha
// las partidas guardadas con ellos (antes eran stubs y la fecha salía sin inicializar).

#include <time.h>

#include <cstdint>
#include <cstring>

#include "nnsdk/types.h"

namespace nn::time {

struct PosixTime {
    int64_t value;
};
struct CalendarTime {
    int16_t year;
    int8_t month, day, hour, minute, second;
    int8_t pad;
};
struct CalendarAdditionalInfo {
    int32_t day_of_week;  // DayOfWeek: domingo 0
    int32_t year_day;     // 0..365
    char time_zone[8];
    bool is_dst;
    int32_t utc_offset_seconds;
};
static_assert(sizeof(CalendarTime) == 8 && sizeof(CalendarAdditionalInfo) == 24);

class StandardUserSystemClock {
public:
    static Result GetCurrentTime(PosixTime* out);
};
class StandardNetworkSystemClock {
public:
    static Result GetCurrentTime(PosixTime* out);
};

namespace {

CalendarTime to_calendar(const tm& t) {
    return CalendarTime{static_cast<int16_t>(t.tm_year + 1900), static_cast<int8_t>(t.tm_mon + 1),
                        static_cast<int8_t>(t.tm_mday),         static_cast<int8_t>(t.tm_hour),
                        static_cast<int8_t>(t.tm_min),          static_cast<int8_t>(t.tm_sec), 0};
}

}  // namespace

Result Initialize() { return ResultSuccess; }
Result Finalize() { return ResultSuccess; }

Result StandardUserSystemClock::GetCurrentTime(PosixTime* out) {
    out->value = ::time(nullptr);
    return ResultSuccess;
}
Result StandardNetworkSystemClock::GetCurrentTime(PosixTime* out) {
    out->value = ::time(nullptr);
    return ResultSuccess;
}

Result ToCalendarTime(CalendarTime* out, CalendarAdditionalInfo* info, const PosixTime& posix) {
    time_t t = static_cast<time_t>(posix.value);
    tm local{};
    localtime_r(&t, &local);
    *out = to_calendar(local);
    if (info) {
        memset(info, 0, sizeof *info);
        info->day_of_week = local.tm_wday;
        info->year_day = local.tm_yday;
        strncpy(info->time_zone, local.tm_zone ? local.tm_zone : "UTC", sizeof info->time_zone - 1);
        info->is_dst = local.tm_isdst > 0;
        info->utc_offset_seconds = static_cast<int32_t>(local.tm_gmtoff);
    }
    return ResultSuccess;
}

CalendarTime ToCalendarTimeInUtc(const PosixTime& posix) {
    time_t t = static_cast<time_t>(posix.value);
    tm utc{};
    gmtime_r(&t, &utc);
    return to_calendar(utc);
}

PosixTime ToPosixTimeFromUtc(const CalendarTime& c) {
    tm t{};
    t.tm_year = c.year - 1900;
    t.tm_mon = c.month - 1;
    t.tm_mday = c.day;
    t.tm_hour = c.hour;
    t.tm_min = c.minute;
    t.tm_sec = c.second;
    return PosixTime{static_cast<int64_t>(timegm(&t))};
}

int GetDayOfWeek(int year, int month, int day) {
    tm t{};
    t.tm_year = year - 1900;
    t.tm_mon = month - 1;
    t.tm_mday = day;
    t.tm_hour = 12;
    time_t s = timegm(&t);
    gmtime_r(&s, &t);
    return t.tm_wday;
}

bool IsValidDate(int year, int month, int day) {
    if (month < 1 || month > 12 || day < 1) return false;
    static const int kDays[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    bool leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    return day <= kDays[month - 1] + (month == 2 && leap ? 1 : 0);
}

}  // namespace nn::time
