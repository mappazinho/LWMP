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


def _order_visible_notes_for_draw(visible_slice, is_white_key_data):
    if visible_slice is None or len(visible_slice) <= 1 or is_white_key_data is None:
        return visible_slice
    white_mask = is_white_key_data[visible_slice['pitch']] == 1
    if np.all(white_mask) or not np.any(white_mask):
        return visible_slice
    return np.concatenate((visible_slice[white_mask], visible_slice[~white_mask]))


def _cull_overlapped_visible_notes(visible_slice, overlap_cull_duration_similarity, overlap_cull_coverage_threshold):
    if visible_slice is None or len(visible_slice) <= 1:
        return visible_slice

    keep_mask = np.ones(len(visible_slice), dtype=bool)
    kept_intervals_by_pitch = {}

    for idx in range(len(visible_slice) - 1, -1, -1):
        note = visible_slice[idx]
        pitch = int(note['pitch'])
        on_time = float(note['on_time'])
        off_time = float(note['off_time'])
        note_duration = max(0.0001, off_time - on_time)

        should_skip = False
        for kept_on, kept_off, kept_duration in kept_intervals_by_pitch.get(pitch, ()):
            overlap = min(off_time, kept_off) - max(on_time, kept_on)
            if overlap <= 0.0:
                continue

            duration_similarity = min(note_duration, kept_duration) / max(note_duration, kept_duration, 0.0001)
            if duration_similarity < overlap_cull_duration_similarity:
                continue

            overlap_coverage = overlap / min(note_duration, kept_duration)
            if overlap_coverage >= overlap_cull_coverage_threshold:
                should_skip = True
                break

        if should_skip:
            keep_mask[idx] = False
        else:
            kept_intervals_by_pitch.setdefault(pitch, []).append((on_time, off_time, note_duration))

    if np.all(keep_mask):
        return visible_slice
    return visible_slice[keep_mask]


def _precompute_overlap_cull_mask(notes_array, overlap_cull_duration_similarity, overlap_cull_coverage_threshold, overlap_cull_recent_candidates):
    if notes_array is None or len(notes_array) <= 1:
        return np.ones(len(notes_array), dtype=bool)

    keep_mask = np.ones(len(notes_array), dtype=bool)
    recent_kept_by_pitch = [[] for _ in range(128)]
    recent_limit = max(1, int(overlap_cull_recent_candidates))

    for idx in range(len(notes_array) - 1, -1, -1):
        note = notes_array[idx]
        pitch = int(note['pitch'])
        on_time = float(note['on_time'])
        off_time = float(note['off_time'])
        note_duration = max(0.0001, off_time - on_time)

        should_skip = False
        for kept_on, kept_off, kept_duration in recent_kept_by_pitch[pitch]:
            if kept_on >= off_time:
                continue
            overlap = min(off_time, kept_off) - max(on_time, kept_on)
            if overlap <= 0.0:
                continue

            duration_similarity = min(note_duration, kept_duration) / max(note_duration, kept_duration, 0.0001)
            if duration_similarity < overlap_cull_duration_similarity:
                continue

            overlap_coverage = overlap / min(note_duration, kept_duration)
            if overlap_coverage >= overlap_cull_coverage_threshold:
                should_skip = True
                break

        if should_skip:
            keep_mask[idx] = False
            continue

        recent_kept_by_pitch[pitch].insert(0, (on_time, off_time, note_duration))
        if len(recent_kept_by_pitch[pitch]) > recent_limit:
            del recent_kept_by_pitch[pitch][recent_limit:]

    return keep_mask


def _assign_note_depths(visible_notes):
    count = len(visible_notes)
    if count <= 0:
        return
    visible_notes['depth'] = (np.arange(1, count + 1, dtype=np.float32) / np.float32(count + 1))
