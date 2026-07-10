/*
 * midi_batch_send.c
 * Helper DLL for batched OmniMIDI KDMAPI event delivery.
 *
 * Instead of calling SendDirectData from Python once per MIDI event
 * (each call paying ctypes overhead), this runs the entire send loop
 * in native code — one Python-to-DLL roundtrip per batch.
 *
 * Compile (Developer Command Prompt):
 *   cl.exe /LD midi_batch_send.c /Fe:midi_batch_send.dll
 *
 * Or with gcc:
 *   gcc -shared -o midi_batch_send.dll midi_batch_send.c
 */

#include <stdint.h>

/* Matches OmniMIDI's KDMAPI SendDirectData signature (__cdecl, no return) */
typedef void (__cdecl *SendDirectDataFn)(uint32_t message);

/*
 * BatchSendDirectData — send an array of packed MIDI messages.
 *
 * send_fn  : pointer to OmniMIDI's SendDirectData function
 * messages : array of uint32_t packed messages, same format as SendDirectData
 * count    : number of messages in the array
 *
 * Returns the number of messages sent.
 */
__declspec(dllexport) int __cdecl BatchSendDirectData(SendDirectDataFn send_fn, uint32_t *messages, int count)
{
    int i;
    for (i = 0; i < count; i++) {
        send_fn(messages[i]);
    }
    return count;
}
