import numpy as np

RENDER_NOTE_DTYPE = np.dtype([
    ('on_time', 'f4'),
    ('off_time', 'f4'),
    ('pitch', 'u1'),
    ('velocity', 'u1'),
    ('track', 'u1'),
    ('padding', 'u1'),
    ('depth', 'f4'),
])


def _pack_render_notes(notes_array):
    packed = np.empty(len(notes_array), dtype=RENDER_NOTE_DTYPE)
    if len(notes_array) == 0:
        return packed
    packed['on_time'] = notes_array['on_time']
    packed['off_time'] = notes_array['off_time']
    packed['pitch'] = notes_array['pitch']
    packed['velocity'] = notes_array['velocity']
    packed['track'] = notes_array['track']
    packed['padding'] = notes_array['padding']
    packed['depth'] = 0.0
    return packed


SHARP_BITMASK = 0x54A


def _assign_layer_depth(notes, base_depth):
    pitch_mod = notes['pitch'].astype(np.uint32) % 12
    sharp_mask = ((np.uint32(1) << pitch_mod) & np.uint32(SHARP_BITMASK)) != 0
    notes['depth'] = np.where(sharp_mask, 0.5 + base_depth * 0.5, base_depth * 0.5)


def _build_base_render_data(all_notes_gpu):
    render_sort_idx = np.argsort(all_notes_gpu['on_time'], kind='stable')
    sorted_notes = all_notes_gpu[render_sort_idx]
    base_render_notes = _pack_render_notes(sorted_notes)
    return base_render_notes, base_render_notes['on_time']


def _build_render_data_for_mode(base_render_notes, mode):
    render_notes = np.ascontiguousarray(base_render_notes.copy())
    if len(render_notes) == 0:
        return render_notes, render_notes['on_time']

    if mode == 'track':
        pair_slots = np.zeros((256, 16), dtype=np.uint8)
        pair_present = np.zeros((256, 16), dtype=bool)
        pair_present[render_notes['track'], render_notes['padding']] = True

        slot = 0
        for track_idx in range(256):
            for channel_idx in range(16):
                if pair_present[track_idx, channel_idx]:
                    pair_slots[track_idx, channel_idx] = slot % 128
                    slot += 1

        render_notes['track'] = pair_slots[render_notes['track'], render_notes['padding']]
    else:
        render_notes['track'] = render_notes['padding']

    return render_notes, render_notes['on_time']
