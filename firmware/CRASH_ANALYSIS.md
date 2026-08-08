# Dual-mode firmware crash: root cause + fixes

**Symptom:** device works for one voice interaction, then faults
(`Fault - Unknown`, core 1, idle-task PC); wake-word engine dead until reboot.

**Root cause:** `esp_websocket_client` invokes the event handler on its OWN
FreeRTOS task. The component does main-loop-only work from that task:

1. **[CRITICAL] Cross-thread speaker + queue access.** `process_received_audio_`
   (called from `WEBSOCKET_EVENT_DATA`) calls `speaker_->play/start/stop` and
   mutates `audio_queue_` on the websocket task, with no lock, while HA Assist
   drives the SAME speaker and `loop()` touches the same queue on the main loop.
   `std::queue` isn't thread-safe; concurrent push/pop across cores corrupts the
   heap → fault later (idle task), wake engine dead. → defer ALL audio/speaker
   work to `loop()`; hand bytes over via a mutex-guarded queue.
2. **[HIGH] Use-after-free on `audio_queue_.front()`** held across `pop()`/`push()`.
3. **[MED] `portMAX_DELAY`** on websocket send can block the main loop → task
   watchdog fault. → bounded timeout.
4. **[MED] `websocket_client_` lifecycle race**: destroyed on main loop while
   sender threads deref it. → guard with flag/mutex.
5. **[LOW] mic-callback reads shared `state_`/client unlocked**: fold into lock.

**Fix order:** (1)+(2) defer audio to loop() behind a mutex, almost certainly
resolves the crash; (3) bounded send timeout; (4) lifecycle guard.

Audit performed 2026-05-26.
