#pragma once
// Parte de nn::os que otros módulos de libnnsdk reutilizan (cxx.cpp monta std::mutex encima).

#include "nnsdk/types.h"

namespace nn::os {

void SignalEvent(EventType* e);
void LockMutex(MutexType* m);
void UnlockMutex(MutexType* m);
void FinalizeMutex(MutexType* m);
void SignalConditionVariable(ConditionVariableType* c);
void WaitConditionVariable(ConditionVariableType* c, MutexType* m);
ConditionVariableStatus TimedWaitConditionVariable(ConditionVariableType* c, MutexType* m, TimeSpan t);
void FinalizeConditionVariable(ConditionVariableType* c);

}  // namespace nn::os

namespace nnsdk {

// Señala un SystemEvent que crea el "sistema" (nosotros) y que el juego solo espera.
void signal_system_event(nn::os::SystemEventType* event);
// Crea (o rehace) el evento de sistema con su modo de borrado: el aviso de dispositivo de audio es AutoClear, como en el SDK.
void create_system_event(nn::os::SystemEventType* event, bool auto_clear, bool signaled);

}  // namespace nnsdk
